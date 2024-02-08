#!/usr/bin/env python3

# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Copyright 2023 Brno University of Technology (author: Federico Landini)
# Licensed under the MIT license.

from backend.models import (
    average_checkpoints,
    get_model,
)
from common_utils.diarization_dataset import KaldiDiarizationDataset
from os.path import join
from pathlib import Path
from torch.utils.data import DataLoader
from train import _convert
from types import SimpleNamespace
from typing import List, TextIO, Tuple
from safe_gpu import safe_gpu
from scipy.signal import medfilt
import h5py
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch
import yamlargparse


def get_infer_dataloader(args: SimpleNamespace) -> DataLoader:
    infer_set = KaldiDiarizationDataset(
        args.infer_data_dir,
        chunk_size=args.num_frames,
        context_size=args.context_size,
        feature_dim=args.feature_dim,
        frame_shift=args.frame_shift,
        frame_size=args.frame_size,
        input_transform=args.input_transform,
        n_speakers=args.num_speakers,
        sampling_rate=args.sampling_rate,
        shuffle=args.time_shuffle,
        subsampling=args.subsampling,
        use_last_samples=True,
        min_length=0,
        specaugment=args.specaugment,
    )
    infer_loader = DataLoader(
        infer_set,
        batch_size=1,
        collate_fn=_convert,
        num_workers=0,
        shuffle=False,
        worker_init_fn=_init_fn,
    )

    Y, _, _, _, _, _ = infer_set.__getitem__(0)
    assert Y.shape[1] == \
        (args.feature_dim * (1 + 2 * args.context_size)), \
        f"Expected feature dimensionality of \
        {args.feature_dim} but {Y.shape[1]} found."
    return infer_loader


def hard_labels_to_rttm(
    labels: np.ndarray,
    id_file: str,
    rttm_file: TextIO,
    frameshift: float = 10
) -> None:
    """
    Transform NfxNs matrix to an rttm file
    Nf is the number of frames
    Ns is the number of speakers
    The frameshift (in ms) determines how to interpret the frames in the array
    """
    if len(labels.shape) > 1:
        # Remove speakers that do not speak
        non_empty_speakers = np.where(labels.sum(axis=0) != 0)[0]
        labels = labels[:, non_empty_speakers]

    # Add 0's before first frame to use diff
    if len(labels.shape) > 1:
        labels = np.vstack([np.zeros((1, labels.shape[1])), labels])
    else:
        labels = np.vstack([np.zeros(1), labels])
    d = np.diff(labels, axis=0)

    spk_list = []
    ini_list = []
    end_list = []
    if len(labels.shape) > 1:
        n_spks = labels.shape[1]
    else:
        n_spks = 1
    for spk in range(n_spks):
        if n_spks > 1:
            ini_indices = np.where(d[:, spk] == 1)[0]
            end_indices = np.where(d[:, spk] == -1)[0]
        else:
            ini_indices = np.where(d[:] == 1)[0]
            end_indices = np.where(d[:] == -1)[0]
        # Add final mark if needed
        if len(ini_indices) == len(end_indices) + 1:
            end_indices = np.hstack([
                end_indices,
                labels.shape[0] - 1])
        assert len(ini_indices) == len(end_indices), \
            "Quantities of start and end of segments mismatch. \
            Are speaker labels correct?"
        n_segments = len(ini_indices)
        for index in range(n_segments):
            spk_list.append(spk)
            ini_list.append(ini_indices[index])
            end_list.append(end_indices[index])
    for ini, end, spk in sorted(zip(ini_list, end_list, spk_list)):
        rttm_file.write(
            f"SPEAKER {id_file} 1 " +
            f"{round(ini * frameshift / 1000, 3)} " +
            f"{round((end - ini) * frameshift / 1000, 3)} " +
            f"<NA> <NA> spk{spk} <NA> <NA>\n")


def rttm_to_hard_labels(
    rttm_path: str,
    precision: float,
    length: float = -1
) -> Tuple[np.ndarray, List[str]]:
    """
        reads the rttm and returns a NfxNs matrix encoding the segments in
        which each speaker is present (labels 1/0) at the given precision.
        Ns is the number of speakers and Nf is the resulting number of frames,
        according to the parameters given.
        Nf might be shorter than the real length of the utterance, as final
        silence parts cannot be recovered from the rttm.
        If length is defined (s), it is to account for that extra silence.
        In case of silence all speakers are labeled with 0.
        In case of overlap all speakers involved are marked with 1.
        The function assumes that the rttm only contains speaker turns (no
        silence segments).
        The overlaps are extracted from the speaker turn collisions.
    """
    # each row is a turn, columns denote beginning (s) and duration (s) of turn
    data = np.loadtxt(rttm_path, usecols=[3, 4])
    # speaker id of each turn
    spks = np.loadtxt(rttm_path, usecols=[7], dtype='str')
    spk_ids = np.unique(spks)
    Ns = len(spk_ids)
    if data.shape[0] == 2 and len(data.shape) < 2:  # if only one segment
        data = np.asarray([data])
        spks = np.asarray([spks])
    # length of the file (s) that can be recovered from the rttm,
    # there might be extra silence at the end
    len_file = data[-1][0]+data[-1][1]
    if length > len_file:
        len_file = length

    # matrix in given precision
    matrix = np.zeros([int(round(len_file*precision)), Ns])
    # ranges to mark each turn
    ranges = np.around((np.array([data[:, 0],
                        data[:, 0]+data[:, 1]]).T*precision)).astype(int)

    for s in range(Ns):  # loop over speakers
        # loop all the turns of the speaker
        for init_end in ranges[spks == spk_ids[s], :]:
            matrix[init_end[0]:init_end[1], s] = 1  # mark the spk
    return matrix, spk_ids


def _init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def estimate_diarization_outputs(
    model,
    inputs: torch.Tensor,
    args: SimpleNamespace
) -> List[torch.Tensor]:
    assert args.estimate_spk_qty_thr != -1 or \
        args.estimate_spk_qty != -1, \
        "Either 'estimate_spk_qty_thr' or 'estimate_spk_qty' \
        arguments have to be defined."
    (
        all_frame_embs,
        per_frameenclayer_ys_logits,
        per_frameenclayer_attractors_logits,
        per_frameenclayer_attractors,
        per_prcvblock_ys_logits,
        per_prcvblock_attractors_logits,
        per_prcvblock_attractors,
        per_prcvblock_l2a_entropy_term,
        per_prcvblock_latents
    ) = model.forward(inputs, args)

    ys_active = []
    existence_probs = torch.sigmoid(per_frameenclayer_attractors_logits[:, :, -1])
    ys = [torch.sigmoid(y) for y in per_frameenclayer_ys_logits[:, :, :, -1]]
    for p, y in zip(existence_probs, ys):
        if args.estimate_spk_qty != -1:
            _, order = torch.sort(p, descending=True)
            ys_active.append(y[:, order[:args.estimate_spk_qty]])
        elif args.estimate_spk_qty_thr != -1:
            active_speakers = torch.where(p >= args.estimate_spk_qty_thr)[0]
            ys_active.append(y[:, active_speakers])
        else:
            NotImplementedError(
                'estimate_spk_qty or estimate_spk_qty_thr needed.')
    return (
        ys_active, existence_probs, per_prcvblock_latents,
        per_prcvblock_attractors, torch.stack(ys))


def postprocess_output(
    probabilities,
    subsampling: int,
    threshold: float,
    median_window_length: int,
    normalize_probs: bool
) -> np.ndarray:
    """Threshold probabilities and apply median filter."""
    if normalize_probs:
        probabilities = (probabilities - probabilities.min(axis=0)[0]) / \
                probabilities.max(axis=0)[0]
    thresholded = probabilities.to("cpu") > threshold
    filtered = np.zeros(thresholded.shape)
    for spk in range(filtered.shape[1]):
        filtered[:, spk] = medfilt(
            thresholded[:, spk].to(float),
            kernel_size=median_window_length).astype(bool)
    probs_extended = np.repeat(filtered, subsampling, axis=0)
    return probs_extended


def parse_arguments() -> SimpleNamespace:
    """Parse arguments"""
    parser = yamlargparse.ArgumentParser(description='DiaPer inference')
    parser.add_argument('-c', '--config', help='config file path',
                        action=yamlargparse.ActionConfigFile)
    parser.add_argument('--attractor-existence-loss-weight', default=1.0, type=float,
                        help='weighting parameter')
    parser.add_argument('--attractor-frame-comparison', default='dotprod',
                        type=str, choices=['dotprod', 'xattention'],
                        help='how to compare attractors and frame embeddings')
    parser.add_argument('--att-qty-loss-weight', default=0.0, type=float)
    parser.add_argument('--att-qty-reg-loss-weight', default=0.0, type=float)
    parser.add_argument('--condition-frame-encoder', type=bool, default=True)
    parser.add_argument('--context-activations', type=bool, default=False)
    parser.add_argument('--context-size', type=int)
    parser.add_argument('--d-latents', type=int,
                        help='dimension of attractors')
    parser.add_argument('--detach-attractor-loss', default=False, type=bool,
                        help='If True, avoid backpropagation on attractor loss')
    parser.add_argument('--dropout_attractors', type=float,
                        help='attention dropout for attractors path')
    parser.add_argument('--dropout_frames', type=float,
                        help='attention dropout for frame embeddings path')
    parser.add_argument('--epochs', type=str,
                        help='epochs to average separated by commas \
                        or - for intervals.')
    parser.add_argument('--estimate-spk-qty', default=-1, type=int)
    parser.add_argument('--estimate-spk-qty-thr', default=-1, type=float)
    parser.add_argument('--feature-dim', type=int)
    parser.add_argument('--frame-encoder-heads', type=int)
    parser.add_argument('--frame-encoder-layers', type=int)
    parser.add_argument('--frame-encoder-units', type=int)
    parser.add_argument('--frame-size', type=int)
    parser.add_argument('--frame-shift', type=int)
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--hidden-size', type=int,
                        help='number of units in SA blocks')
    parser.add_argument('--infer-data-dir', help='inference data directory.')
    parser.add_argument('--input-transform', default='',
                        choices=['logmel', 'logmel_meannorm',
                                 'logmel_meanvarnorm'],
                        help='input normalization transform')
    parser.add_argument('--latents2attractors', type=str, default='linear')
    parser.add_argument('--length-normalize', default=False, type=bool)
    parser.add_argument('--log-report-batches-num', default=1, type=float)
    parser.add_argument('--median-window-length', default=11, type=int)
    parser.add_argument('--model-type', default='AttractorsPath',
                        help='Type of model (for now only AttractorsPath)')
    parser.add_argument('--models-path', type=str,
                        help='directory with model(s) to evaluate')
    parser.add_argument('--n-attractors', type=int,
                        help='Number of attractors to use')
    parser.add_argument('--n-blocks-attractors', type=int,
                        help='number of blocks in the transformer encoder')
    parser.add_argument('--n-internal-blocks-attractors', type=int, default=1,
                        help='number of Perceiver internal block, which \
                        repeats self-attention layers for attractors')
    parser.add_argument('--n-latents', type=int,
                        help='number of latents')
    parser.add_argument('--n-selfattends-attractors', type=int,
                        help='number of slef-attention layers per block')
    parser.add_argument('--n-sa-heads-attractors', type=int,
                        help='number of self-attention heads per layer')
    parser.add_argument('--n-xa-heads-attractors', type=int,
                        help='number of cross-attention heads per layer')
    parser.add_argument('--normalize-probs', default=False, type=bool)
    parser.add_argument('--num-frames', default=-1, type=int,
                        help='number of frames in one utterance')
    parser.add_argument('--num-speakers', type=int)
    parser.add_argument('--plot-output', default=False, type=bool)
    parser.add_argument('--posenc-maxlen', type=int, default=36000,
                        help="The maximum length allowed for the positional \
                        encoding. i.e. 36000 with 0.1s frames is 1 hour")
    parser.add_argument('--pre-xa-heads', type=int,
                        help='number of pre-Perceiver cross-attention heads')
    parser.add_argument('--ref-rttms-dir', type=str, default='',
                        help='directory with reference RTTMs, used for plots')
    parser.add_argument('--rttms-dir', type=str,
                        help='output directory for rttm files.')
    parser.add_argument('--sampling-rate', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--speakerid-loss', type=str, default='',
                        choices=['arcface', 'vanilla'])
    parser.add_argument('--speakerid-num-speakers', type=int, default=-1)
    parser.add_argument('--specaugment', type=bool, default=False)
    parser.add_argument('--subsampling', type=int)
    parser.add_argument('--threshold', default=0.5, type=float)
    parser.add_argument('--time-shuffle', action='store_true',
                        help='Shuffle time-axis order before input to the network')
    parser.add_argument('--use-frame-selfattention', default=False, type=bool)
    parser.add_argument('--use-posenc', default=False, type=bool)
    parser.add_argument('--use-pre-crossattention', default=False, type=bool)
    parser.add_argument('--vad-loss-weight', default=0.0, type=float)
    init_args = parser.parse_args()
    return init_args


if __name__ == '__main__':
    args = parse_arguments()

    # For reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    np.random.seed(args.seed)  # Numpy module.
    random.seed(args.seed)  # Python random module.
    torch.manual_seed(args.seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    logging.info(args)

    infer_loader = get_infer_dataloader(args)

    if args.gpu >= 1:
        safe_gpu.claim_gpus(nb_gpus=args.gpu)
        args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")

    assert args.estimate_spk_qty_thr != -1 or \
        args.estimate_spk_qty != -1, \
        ("Either 'estimate_spk_qty_thr' or 'estimate_spk_qty' "
         "arguments have to be defined.")
    if args.estimate_spk_qty != -1:
        out_dir = join(args.rttms_dir, f"spkqty{args.estimate_spk_qty}_\
            thr{args.threshold}_median{args.median_window_length}")
    elif args.estimate_spk_qty_thr != -1:
        out_dir = join(args.rttms_dir, f"spkqtythr{args.estimate_spk_qty_thr}_\
            thr{args.threshold}_median{args.median_window_length}")

    model = get_model(args)

    model = average_checkpoints(
        args.device, model, args.models_path, args.epochs)
    model.eval()

    out_dir = join(
        args.rttms_dir,
        f"epochs{args.epochs}",
        f"timeshuffle{args.time_shuffle}",
        (f"spk_qty{args.estimate_spk_qty}_"
            f"spk_qty_thr{args.estimate_spk_qty_thr}"),
        f"detection_thr{args.threshold}",
        f"median{args.median_window_length}",
        f"subsampling{args.subsampling}",
        "rttms"
    )
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    for i, batch in enumerate(infer_loader):
        input = torch.stack(batch['xs']).to(args.device)
        name = batch['names'][0]
        with torch.no_grad():
            (
                y_pred,
                existence_probs,
                per_prcvblock_latents,
                per_prcvblock_attractors,
                y_probs
            ) = estimate_diarization_outputs(model, input, args)
        # Each one has a single sequence
        y_pred = y_pred[0]
        existence_probs = existence_probs[0]
        y_probs = y_probs[0]
        per_prcvblock_attractors = torch.stack([a[0] for a in per_prcvblock_attractors])
        per_prcvblock_latents = torch.stack([lat[0] for lat in per_prcvblock_latents])
        post_y = postprocess_output(
            y_pred, args.subsampling,
            args.threshold, args.median_window_length,
            args.normalize_probs)
        rttm_filename = join(out_dir, f"{name}.rttm")
        with open(rttm_filename, 'w', encoding='UTF-8') as rttm_file:
            hard_labels_to_rttm(post_y, name, rttm_file)
        if args.plot_output:
            fig, axs = plt.subplots(y_probs.shape[1]+1)
            fig.set_figwidth(y_probs.shape[0]/100)
            for i in range(y_probs.shape[1]):
                y_probs_extended = y_probs[:, i].repeat_interleave(args.subsampling)
                y_probs_postprocessed = postprocess_output(
                    y_probs[:, i].unsqueeze(1),
                    args.subsampling,
                    args.threshold,
                    args.median_window_length,
                    args.normalize_probs)
                axs[i].set_ylim([-0.1, 1.1])
                axs[i].set_xticks([])
                axs[i].plot(range(
                    y_probs_extended.shape[0]),
                    y_probs_extended, linewidth=0.5)
                axs[i].plot(range(
                    y_probs_postprocessed.shape[0]),
                    y_probs_postprocessed, 'r', linewidth=0.2)
                axs[i].title.set_text(f'{existence_probs[i].item():.20f}')
                axs[i].title.set_size(6)
                for j in range(0, y_probs_extended.shape[0], 100):
                    axs[i].axvline(
                        x=j, ymin=-0.5, ymax=1.5, c='black',
                        lw=0.25, ls=':', clip_on=False)
            if args.ref_rttms_dir:
                ref_frames, ref_spks = rttm_to_hard_labels(join(
                    args.ref_rttms_dir, f"{name}.rttm"), 100)
            for i in range(ref_frames.shape[1]):
                axs[y_probs.shape[1]].set_ylim([-0.1, 1.1])
                axs[y_probs.shape[1]].plot(range(
                    ref_frames.shape[0]),
                    (1-0.1*i)*ref_frames[:, i], linewidth=0.5)
            axs[y_probs.shape[1]].title.set_text('Reference')
            axs[y_probs.shape[1]].title.set_size(6)
            for j in range(0, y_probs_extended.shape[0], 100):
                axs[y_probs.shape[1]].axvline(
                    x=j, ymin=-0.5, ymax=1.5, c='black',
                    lw=0.25, ls=':', clip_on=False)
            plt.subplots_adjust(hspace=1)
            png_filename = join(out_dir, f"{name}.png")
            fig.savefig(png_filename, dpi=300)
            plt.figure().clear()
            plt.close()
            plt.cla()
            plt.clf()
