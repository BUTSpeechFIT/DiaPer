#!/usr/bin/env python3

# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Copyright 2023 Brno University of Technology (author: Federico Landini)
# Licensed under the MIT license.

from backend.models import (
    average_checkpoints,
    get_model,
)
from common_utils.diarization_dataset import KaldiDiarizationDataset
from common_utils.features import stft, transform, splice, subsample
from infer import (
    estimate_diarization_outputs,
    get_infer_dataloader,
    hard_labels_to_rttm,
    postprocess_output,
    rttm_to_hard_labels,
)
from os.path import join
from pathlib import Path
from safe_gpu import safe_gpu
from scipy.signal import medfilt
from torch.utils.data import DataLoader
from train import _convert
from types import SimpleNamespace
from typing import List, TextIO, Tuple
import h5py
import librosa
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import soundfile as sf
import torch
import yamlargparse


def parse_arguments() -> SimpleNamespace:
    parser = yamlargparse.ArgumentParser(
        description='DiaPer inference of a single waveform')
    parser.add_argument('-c', '--config', help='config file path',
                        action=yamlargparse.ActionConfigFile)
    parser.add_argument('--attractor-existence-loss-weight', default=1.0,
                        type=float, help='weighting parameter')
    parser.add_argument('--attractor-frame-comparison', default='dotprod',
                        type=str, choices=['dotprod', 'xattention'],
                        help='how are attractors and frame embeddings compared')
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
    parser.add_argument('--infer-data-dir', help='inference data directory \
                        (it is ignored in this script).')
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
    parser.add_argument('--wav-dir', required=True, type=str)
    parser.add_argument('--wav-name', required=True, type=str)
    args = parser.parse_args()
    return args


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

    filepath = os.path.join(args.wav_dir, f"{args.wav_name}.wav")
    file_frames_length = int((100*librosa.get_duration(filename=filepath)) // 1)
    data, samplerate = sf.read(
        filepath, start=0, stop=(file_frames_length * args.frame_shift))
    Y = stft(data, args.frame_size, args.frame_shift)
    Y = transform(
        Y, args.sampling_rate, args.feature_dim, args.input_transform, False)
    Y_spliced = splice(Y, args.context_size)
    Y_ss, _ = subsample(Y_spliced, Y_spliced, args.subsampling)

    input = torch.from_numpy(np.asarray([Y_ss])).to(args.device)
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
    rttm_filename = join(out_dir, f"{args.wav_name}.rttm")
    with open(rttm_filename, 'w') as rttm_file:
        hard_labels_to_rttm(post_y, args.wav_name, rttm_file)
    if args.plot_output:
        fig, axs = plt.subplots(y_probs.shape[1]+1)
        fig.set_figwidth(y_probs.shape[0]/100)
        for i in range(y_probs.shape[1]):
            y_probs_extended = y_probs[:, i].repeat_interleave(args.subsampling)
            y_probs_postprocessed = postprocess_output(
                y_probs[:, i].unsqueeze(1),
                args.subsampling, args.threshold,
                args.median_window_length, args.normalize_probs)
            axs[i].set_ylim([-0.1, 1.1])
            axs[i].set_xticks([])
            axs[i].plot(range(
                y_probs_extended.shape[0]), y_probs_extended, linewidth=0.5)
            axs[i].plot(range(
                y_probs_postprocessed.shape[0]),
                y_probs_postprocessed, 'r', linewidth=0.2)
            axs[i].title.set_text('{:.20f}'.format(existence_probs[i].item()))
            axs[i].title.set_size(6)
            for j in range(0, y_probs_extended.shape[0], 100):
                axs[i].axvline(
                    x=j, ymin=-0.5, ymax=1.5, c='black',
                    lw=0.25, ls=':', clip_on=False)
        if args.ref_rttms_dir:
            ref_frames, ref_spks = rttm_to_hard_labels(
                join(args.ref_rttms_dir, f"{args.wav_name}.rttm"), 100)
        for i in range(ref_frames.shape[1]):
            axs[y_probs.shape[1]].set_ylim([-0.1, 1.1])
            axs[y_probs.shape[1]].plot(
                range(ref_frames.shape[0]),
                (1-0.1*i)*ref_frames[:, i], linewidth=0.5)
        axs[y_probs.shape[1]].title.set_text('Reference')
        axs[y_probs.shape[1]].title.set_size(6)
        for j in range(0, y_probs_extended.shape[0], 100):
            axs[y_probs.shape[1]].axvline(
                x=j, ymin=-0.5, ymax=1.5, c='black',
                lw=0.25, ls=':', clip_on=False)
        plt.subplots_adjust(hspace=1)
        png_filename = join(out_dir, f"{args.wav_name}.png")
        fig.savefig(png_filename, dpi=300)
        plt.figure().clear()
        plt.close()
        plt.cla()
        plt.clf()
