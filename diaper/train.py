#!/usr/bin/env python3

# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Copyright 2023 Brno University of Technology (author: Federico Landini)
# Licensed under the MIT license.

from backend.losses import (
    get_loss,
    pad_labels_zeros,
    pad_sequence
)
from backend.models import (
    average_checkpoints,
    get_model,
    load_checkpoint,
    save_checkpoint,
)
from backend.updater import setup_optimizer, get_rate
from common_utils.diarization_dataset import (
    KaldiDiarizationDataset,
    PrecomputedDiarizationDataset)
from common_utils.metrics import (
    calculate_metrics,
    new_metrics,
    reset_metrics,
    update_metrics,
)
from safe_gpu import safe_gpu
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple
import logging
import numpy as np
import os
import random
import torch
import yamlargparse


def _init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _convert(
    batch: List[Tuple[torch.Tensor, torch.Tensor, str]]
) -> Dict[str, Any]:
    return {'xs': [x for x, _, _, _, _, _ in batch],
            'ts': [t for _, t, _, _, _, _ in batch],
            'names': [r for _, _, r, _, _, _ in batch],
            'beg': [b for _, _, _, b, _, _ in batch],
            'end': [e for _, _, _, _, e, _ in batch],
            'spk_ids': [i for _, _, _, _, _, i in batch]}


def compute_loss_and_metrics(
    model: torch.nn.Module,
    labels: torch.Tensor,
    inputs: torch.Tensor,
    n_speakers: List[int],
    spkid_labels: torch.Tensor,
    acum_metrics: Dict[str, float],
    args: SimpleNamespace
) -> Tuple[torch.Tensor, Dict[str, float]]:
    in_size = args.feature_dim * (1 + 2 * args.context_size)
    # torch.autograd.set_detect_anomaly(True)
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

    y_probs = torch.sigmoid(per_frameenclayer_ys_logits[:, :, :, -1])
    (
        activation_loss_BCE,
        activation_loss_DER,
        attractor_existence_loss,
        att_qty_loss,
        vad_loss,
        osd_loss,
        spkid_loss
    ) = get_loss(
        per_frameenclayer_ys_logits[:, :, :, -1],
        labels,
        n_speakers,
        per_frameenclayer_attractors_logits[:, :, -1],
        model,
        per_frameenclayer_attractors[:, :, :, -1],
        args.speakerid_num_speakers,
        spkid_labels,
        args
    )

    l2a_entropy_term = per_prcvblock_l2a_entropy_term[-1]

    if args.intermediate_loss_frameencoder:
        intermediate_activation_losses_BCE = torch.zeros(per_frameenclayer_ys_logits.shape[-1] - 1)
        intermediate_activation_losses_DER = torch.zeros(per_frameenclayer_ys_logits.shape[-1] - 1)
        intermediate_attractor_existence_losses = torch.zeros(per_frameenclayer_ys_logits.shape[-1] - 1)
        intermediate_att_qty_losses = torch.zeros(per_frameenclayer_ys_logits.shape[-1] - 1)
        intermediate_vad_losses = torch.zeros(per_frameenclayer_ys_logits.shape[-1] - 1)
        intermediate_osd_losses = torch.zeros(per_frameenclayer_ys_logits.shape[-1] - 1)
        intermediate_spkid_losses = torch.zeros(per_frameenclayer_ys_logits.shape[-1] - 1)
        for j in range(per_frameenclayer_ys_logits.shape[-1] - 1):
            # we skip the last layer (already computed)
            (
                activation_loss_BCE_j,
                activation_loss_DER_j,
                attractor_existence_loss_j,
                att_qty_loss_j,
                vad_loss_j,
                osd_loss_j,
                spkid_loss_j
            ) = get_loss(
                per_frameenclayer_ys_logits[:, :, :, j],
                labels,
                n_speakers,
                per_frameenclayer_attractors_logits[:, :, j],
                model,
                per_frameenclayer_attractors[:, :, :, j],
                args.speakerid_num_speakers,
                spkid_labels,
                args
            )
            intermediate_activation_losses_BCE[j] = activation_loss_BCE_j
            intermediate_activation_losses_DER[j] = activation_loss_DER_j
            intermediate_attractor_existence_losses[j] = attractor_existence_loss_j
            intermediate_att_qty_losses[j] = att_qty_loss_j
            intermediate_vad_losses[j] = vad_loss_j
            intermediate_osd_losses[j] = osd_loss_j
            intermediate_spkid_losses[j] = spkid_loss_j
        activation_loss_BCE += torch.mean(intermediate_activation_losses_BCE)
        activation_loss_DER += torch.mean(intermediate_activation_losses_DER)
        attractor_existence_loss += torch.mean(intermediate_attractor_existence_losses)
        att_qty_loss += torch.mean(intermediate_att_qty_losses)
        vad_loss += torch.mean(intermediate_vad_losses)
        osd_loss += torch.mean(intermediate_osd_losses)
        spkid_loss += torch.mean(intermediate_spkid_losses)

    if args.intermediate_loss_perceiver:
        intermediate_activation_losses_BCE = torch.zeros(per_prcvblock_ys_logits.shape[-1] - 1)
        intermediate_activation_losses_DER = torch.zeros(per_prcvblock_ys_logits.shape[-1] - 1)
        intermediate_attractor_existence_losses = torch.zeros(per_prcvblock_ys_logits.shape[-1] - 1)
        intermediate_att_qty_losses = torch.zeros(per_prcvblock_ys_logits.shape[-1] - 1)
        intermediate_vad_losses = torch.zeros(per_prcvblock_ys_logits.shape[-1] - 1)
        intermediate_osd_losses = torch.zeros(per_prcvblock_ys_logits.shape[-1] - 1)
        intermediate_spkid_losses = torch.zeros(per_prcvblock_ys_logits.shape[-1] - 1)
        for i in range(per_prcvblock_ys_logits.shape[-1] - 1):
            # we skip the last layer (already computed)
            (
                activation_loss_BCE_i,
                activation_loss_DER_i,
                attractor_existence_loss_i,
                att_qty_loss_i,
                vad_loss_i,
                osd_loss_i,
                spkid_loss_i
            ) = get_loss(
                per_prcvblock_ys_logits[:, :, :, i],
                labels,
                n_speakers,
                per_prcvblock_attractors_logits[:, :, i],
                model,
                per_prcvblock_attractors[:, :, :, i],
                args.speakerid_num_speakers,
                spkid_labels,
                args
            )
            intermediate_activation_losses_BCE[i] = activation_loss_BCE_i
            intermediate_activation_losses_DER[i] = activation_loss_DER_i
            intermediate_attractor_existence_losses[i] = attractor_existence_loss_i
            intermediate_att_qty_losses[i] = att_qty_loss_i
            intermediate_vad_losses[i] = vad_loss_i
            intermediate_osd_losses[i] = osd_loss_i
            intermediate_spkid_losses[i] = spkid_loss_i
        activation_loss_BCE += torch.mean(intermediate_activation_losses_BCE)
        l2a_entropy_term += torch.mean(per_prcvblock_l2a_entropy_term[:-1])
        activation_loss_DER += torch.mean(intermediate_activation_losses_DER)
        attractor_existence_loss += torch.mean(intermediate_attractor_existence_losses)
        att_qty_loss += torch.mean(intermediate_att_qty_losses)
        vad_loss += torch.mean(intermediate_vad_losses)
        osd_loss += torch.mean(intermediate_osd_losses)
        spkid_loss += torch.mean(intermediate_spkid_losses)

    loss = activation_loss_BCE * args.activation_loss_BCE_weight + \
        l2a_entropy_term + \
        activation_loss_DER * args.activation_loss_DER_weight + \
        attractor_existence_loss * args.attractor_existence_loss_weight + \
        att_qty_loss * args.att_qty_loss_weight + \
        vad_loss * args.vad_loss_weight + \
        osd_loss * args.osd_loss_weight + \
        spkid_loss * args.speakerid_loss_weight

    metrics = calculate_metrics(
        labels.detach(), y_probs.detach(), threshold=0.5)

    acum_metrics = update_metrics(acum_metrics, metrics)
    acum_metrics['loss'] += loss.item()
    acum_metrics['activation_loss_BCE'] += activation_loss_BCE.item()
    acum_metrics['l2a_entropy_term'] += l2a_entropy_term.item()
    acum_metrics['activation_loss_DER'] += activation_loss_DER.item()
    acum_metrics['attractor_existence_loss'] += attractor_existence_loss.item()
    acum_metrics['att_qty_loss'] += att_qty_loss.item()
    acum_metrics['vad_loss'] += vad_loss.item()
    acum_metrics['osd_loss'] += osd_loss.item()
    acum_metrics['spkid_loss'] += spkid_loss.item()
    return loss, acum_metrics


def get_training_dataloaders(
    args: SimpleNamespace
) -> Tuple[DataLoader, DataLoader]:
    if args.gpu >= 1:
        train_batchsize = args.train_batchsize * args.gpu
        dev_batchsize = args.dev_batchsize * args.gpu
    else:
        train_batchsize = args.train_batchsize
        dev_batchsize = args.dev_batchsize

    if args.train_features_dir is None and args.valid_features_dir is None:
        assert not (args.train_data_dir is None) and \
            not (args.valid_data_dir is None), "--features-dir or \
            --train-data-dir and --valid-data-dir must be defined"
        train_set = KaldiDiarizationDataset(
            args.train_data_dir,
            chunk_size=args.num_frames,
            context_size=args.context_size,
            feature_dim=args.feature_dim,
            frame_shift=args.frame_shift,
            frame_size=args.frame_size,
            input_transform=args.input_transform,
            n_speakers=min(args.num_speakers, args.n_attractors),  # read up to n_attractors speakers
            sampling_rate=args.sampling_rate,
            shuffle=args.time_shuffle,
            subsampling=args.subsampling,
            use_last_samples=args.use_last_samples,
            min_length=args.min_length,
            specaugment=args.specaugment,
        )

        dev_set = KaldiDiarizationDataset(
            args.valid_data_dir,
            chunk_size=args.num_frames,
            context_size=args.context_size,
            feature_dim=args.feature_dim,
            frame_shift=args.frame_shift,
            frame_size=args.frame_size,
            input_transform=args.input_transform,
            n_speakers=min(args.num_speakers, args.n_attractors),  # read up to n_attractors speakers
            sampling_rate=args.sampling_rate,
            shuffle=args.time_shuffle,
            subsampling=args.subsampling,
            use_last_samples=args.use_last_samples,
            min_length=args.min_length,
            specaugment=args.specaugment,
        )

        train_loader = DataLoader(
            train_set,
            batch_size=train_batchsize,
            collate_fn=_convert,
            num_workers=args.num_workers,
            shuffle=True,
            worker_init_fn=_init_fn,
        )

        dev_loader = DataLoader(
            dev_set,
            batch_size=dev_batchsize,
            collate_fn=_convert,
            num_workers=1,
            shuffle=False,
            worker_init_fn=_init_fn,
        )

        Y_train, _, _, _, _, _ = train_set.__getitem__(0)
        Y_dev, _, _, _, _, _ = dev_set.__getitem__(0)
    else:
        assert not (args.train_features_dir is None) and \
            not (args.valid_features_dir is None), \
            "--train-features-dir and --valid-features-dir or \
            --train-data-dir and --valid-data-dir must be defined"
        train_set = PrecomputedDiarizationDataset(
            features_dir=args.train_features_dir,
            batch_size=args.train_batchsize)
        dev_set = PrecomputedDiarizationDataset(
            features_dir=args.valid_features_dir,
            batch_size=args.dev_batchsize)

        train_loader = DataLoader(
            train_set,
            batch_size=train_batchsize,
            collate_fn=_convert,
            num_workers=args.num_workers,
            worker_init_fn=_init_fn,
        )

        dev_loader = DataLoader(
            dev_set,
            batch_size=dev_batchsize,
            collate_fn=_convert,
            num_workers=1,
            worker_init_fn=_init_fn,
        )

    return train_loader, dev_loader


def optimizer_to(optim, device):
    for param in optim.state.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def parse_arguments() -> SimpleNamespace:
    parser = yamlargparse.ArgumentParser(description='DiaPer training')
    parser.add_argument('-c', '--config', help='config file path',
                        action=yamlargparse.ActionConfigFile)
    parser.add_argument('--activation-loss-BCE-weight', default=1.0, type=float,
                        help='weighting parameter for activation loss BCE')
    parser.add_argument('--activation-loss-DER-weight', default=0.0, type=float,
                        help='weighting parameter for activation loss DER')
    parser.add_argument('--attractor-existence-loss-weight', default=1.0,
                        type=float, help='weighting parameter')
    parser.add_argument('--attractor-frame-comparison', default='dotprod',
                        type=str, choices=['dotprod', 'xattention'],
                        help='how are attractors and frame embeddings compared')
    parser.add_argument('--att-qty-loss-weight', default=0.0, type=float)
    parser.add_argument('--condition-frame-encoder', type=bool, default=True)
    parser.add_argument('--context-activations', type=bool, default=False)
    parser.add_argument('--context-size', type=int)
    parser.add_argument('--d-latents', type=int, default=None,
                        help='dimension of latents')
    parser.add_argument('--detach-attractor-loss', type=bool,
                        help='If True, avoid backpropagation on attractor loss')
    parser.add_argument('--dev-batchsize', default=1, type=int,
                        help='number of utterances in one development batch')
    parser.add_argument('--dropout', type=float,
                        help='dropout for the rest of the model')
    parser.add_argument('--dropout_attractors', type=float,
                        help='attention dropout for attractors path')
    parser.add_argument('--dropout_frames', type=float,
                        help='attention dropout for frame embeddings path')
    parser.add_argument('--feature-dim', type=int)
    parser.add_argument('--frame-encoder-heads', type=int)
    parser.add_argument('--frame-encoder-layers', type=int)
    parser.add_argument('--frame-encoder-units', type=int)
    parser.add_argument('--frame-shift', type=int)
    parser.add_argument('--frame-size', type=int)
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--gradclip', default=-1, type=int,
                        help='gradient clipping. if < 0, no clipping')
    parser.add_argument('--init-epochs', type=str, default='',
                        help='Initialize model with average of epochs \
                        separated by commas or - for intervals.')
    parser.add_argument('--init-model-path', type=str, default='',
                        help='Initialize the model from the given directory')
    parser.add_argument('--input-transform', default='',
                        choices=['logmel', 'logmel_meannorm',
                                 'logmel_meanvarnorm'],
                        help='input normalization transform')
    parser.add_argument('--intermediate-loss-frameencoder', default=False, type=bool)
    parser.add_argument('--intermediate-loss-perceiver', default=False, type=bool)
    parser.add_argument('--length-normalize', default=False, type=bool)
    parser.add_argument('--log-report-batches-num', default=1, type=float)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--latents2attractors', type=str, default='dummy',
                        choices=['dummy', 'linear', 'weighted_average'])
    parser.add_argument('--max-epochs', type=int,
                        help='Max. number of epochs to train')
    parser.add_argument('--min-length', default=0, type=int,
                        help='Minimum number of frames for the sequences'
                             ' after downsampling.')
    parser.add_argument('--model-type', default='AttractorsPath',
                        help='Type of model (for now only AttractorsPath)')
    parser.add_argument('--n-attractors', type=int,
                        help='Number of attractors to use')
    parser.add_argument('--n-embeddings', type=int, default=1,
                        help='Number of embeddings to use')
    parser.add_argument('--n-blocks-attractors', type=int,
                        help='number of blocks in the attractors transformer encoder')
    parser.add_argument('--n-blocks-embeddings', type=int,
                        help='number of blocks in the embeddings transformer encoder')
    parser.add_argument('--n-internal-blocks-attractors', type=int, default=1,
                        help='number of Perceiver internal block, which \
                        repeats self-attention layers for attractors')
    parser.add_argument('--n-latents', type=int,
                        help='number of latents')
    parser.add_argument('--n-selfattends-attractors', type=int,
                        help='number of self-attention layers per attractors block')
    parser.add_argument('--n-selfattends-embeddings', type=int,
                        help='number of self-attention layers per embeddings block')
    parser.add_argument('--n-sa-heads-attractors', type=int,
                        help='number of self-attention heads per layer in \
                        attractors path')
    parser.add_argument('--n-sa-heads-frames', type=int,
                        help='number of self-attention heads per layer in \
                        frames self-attention layers')
    parser.add_argument('--n-sa-heads-embeddings', type=int,
                        help='number of self-attention heads per layer \
                        in embeddings path')
    parser.add_argument('--n-xa-heads-attractors', type=int,
                        help='number of cross-attention heads per layer \
                        in attractors path')
    parser.add_argument('--n-xa-heads-embeddings', type=int,
                        help='number of cross-attention heads per layer \
                        in embeddings path')
    parser.add_argument('--n-xa-heads-in', type=int, default=1,
                        help='number of input cross-attention heads')
    parser.add_argument('--n-xa-heads-out', type=int,
                        help='number of output cross-attention heads')
    parser.add_argument('--n-speakers-softmax', type=int, default=0,
                        help='number of speakers to train speaker loss')
    parser.add_argument('--noam-model-size', type=int)
    parser.add_argument('--noam-warmup-steps', type=float)
    parser.add_argument('--norm-loss-per-spk', type=bool, default=False)
    parser.add_argument('--num-frames', type=int,
                        help='number of frames in one utterance')
    parser.add_argument('--num-speakers', type=int,
                        help='maximum number of speakers allowed')
    parser.add_argument('--num-workers', default=1, type=int,
                        help='number of workers in train DataLoader')
    parser.add_argument('--optimizer', type=str)
    parser.add_argument('--osd-loss-weight', default=0.0, type=float)
    parser.add_argument('--output-path', type=str)
    parser.add_argument('--posenc-maxlen', type=int, default=36000,
                        help="The maximum length allowed for the positional \
                        encoding. i.e. 36000 with 0.1s frames is 1 hour")
    parser.add_argument('--pre-xa-heads', type=int,
                        help='number of pre-Perceiver cross-attention heads')
    parser.add_argument('--sampling-rate', type=int)
    parser.add_argument('--save-intermediate', type=int, default=-1,
                        help='save intermediate models every save_intermediate batches')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--shuffle-spk-order', type=bool, default=False)
    parser.add_argument('--speakerid-loss', type=str, default='',
                        choices=['arcface', 'vanilla'])
    parser.add_argument('--speakerid-loss-weight', default=0.0, type=float,
                        help='weighting parameter for speaker ID loss')
    parser.add_argument('--speakerid-num-speakers', type=int, default=-1)
    parser.add_argument('--specaugment', type=bool, default=False)
    parser.add_argument('--subsampling', type=int)
    parser.add_argument('--time-shuffle', action='store_true',
                        help='Shuffle time-axis order before input to the network')
    parser.add_argument('--train-batchsize', default=1, type=int,
                        help='number of utterances in one train batch')
    parser.add_argument('--train-data-dir', default=None,
                        help='kaldi-style data dir used for training.')
    parser.add_argument('--train-features-dir', default=None,
                        help='directory with pre-computed training features')
    parser.add_argument('--use-detection-error-rate', default=False, type=bool)
    parser.add_argument('--use-frame-selfattention', default=False, type=bool)
    parser.add_argument('--use-last-samples', default=True, type=bool)
    parser.add_argument('--use-posenc', default=False, type=bool)
    parser.add_argument('--use-pre-crossattention', default=False, type=bool)
    parser.add_argument('--vad-loss-weight', default=0.0, type=float)
    parser.add_argument('--valid-data-dir', default=None,
                        help='kaldi-style data dir used for validation.')
    parser.add_argument('--valid-features-dir', default=None,
                        help='directory with pre-computed validation features')
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

    torch.multiprocessing.set_sharing_strategy('file_system')

    logging.info(args)

    writer = SummaryWriter(f"{args.output_path}/tensorboard")

    if args.gpu >= 1:
        safe_gpu.claim_gpus(nb_gpus=args.gpu)
        args.device = torch.device("cuda")
        args.log_report_batches_num = int(args.log_report_batches_num / args.gpu)
        torch.multiprocessing.set_sharing_strategy('file_system')
    else:
        args.device = torch.device("cpu")

    if args.init_model_path == '':
        model = get_model(args)
        optimizer = setup_optimizer(args, model)
    else:
        model = get_model(args)
        model = average_checkpoints(
            args.device, model, args.init_model_path, args.init_epochs)
        optimizer = setup_optimizer(args, model)

    train_loader, dev_loader = get_training_dataloaders(args)

    acum_train_metrics = new_metrics()
    acum_dev_metrics = new_metrics()

    if os.path.isfile(os.path.join(
            args.output_path, 'models', 'checkpoint_0.tar')):
        # Load latest model and continue from there
        directory = os.path.join(args.output_path, 'models')
        checkpoints = os.listdir(directory)
        paths = [os.path.join(directory, basename) for
                 basename in checkpoints if basename.startswith("checkpoint_")]
        paths.sort(key=lambda x: os.path.getmtime(x))
        latest = paths[-1]
        epoch, model, optimizer, _ = load_checkpoint(args, latest)
        init_epoch = epoch
    else:
        init_epoch = 0
        # Save initial model
        save_checkpoint(args, init_epoch, model, optimizer, 0)

    model.to(args.device)
    train_batches_qty = 0

    for epoch in range(int(round(init_epoch)), args.max_epochs):
        model.train()
        for i, batch in enumerate(train_loader):
            train_batches_qty += 1
            features = batch['xs']
            labels = batch['ts']
            spkids = batch['spk_ids']

            n_speakers = np.asarray([max(torch.where(t.sum(0) != 0)[0]) + 1
                                     if t.sum() > 0 else 0 for t in labels])
            max_n_speakers = args.n_attractors
            features, labels = pad_sequence(features, labels, args.num_frames)
            labels = pad_labels_zeros(labels, max_n_speakers)
            features = torch.stack(features).to(args.device)
            labels = torch.stack(labels).to(args.device)
            loss, acum_train_metrics = compute_loss_and_metrics(
                model, labels, features, n_speakers,
                spkids, acum_train_metrics, args)
            if i % args.log_report_batches_num == \
                    (args.log_report_batches_num-1):
                for k in acum_train_metrics.keys():
                    writer.add_scalar(
                        f"train_{k}",
                        acum_train_metrics[k] / args.log_report_batches_num,
                        train_batches_qty)
                writer.add_scalar(
                    "lrate",
                    get_rate(optimizer), train_batches_qty)
                acum_train_metrics = reset_metrics(acum_train_metrics)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradclip)
            optimizer.step()

            if not (args.save_intermediate == -1):
                if i % args.save_intermediate == (args.save_intermediate-1):
                    save_checkpoint(args, epoch + (i / 100.0), model, optimizer, loss)

        save_checkpoint(args, epoch+1, model, optimizer, loss)

        dev_batches_qty = 0
        with torch.no_grad():
            model.eval()
            for i, batch in enumerate(dev_loader):
                dev_batches_qty += 1
                features = batch['xs']
                labels = batch['ts']
                spkids = batch['spk_ids']
                n_speakers = np.asarray([max(torch.where(t.sum(0) != 0)[0]) + 1
                                        if t.sum() > 0 else 0 for t in labels])
                max_n_speakers = args.n_attractors
                features, labels = pad_sequence(
                    features, labels, args.num_frames)
                labels = pad_labels_zeros(labels, max_n_speakers)
                features = torch.stack(features).to(args.device)
                labels = torch.stack(labels).to(args.device)
                _, acum_dev_metrics = compute_loss_and_metrics(
                    model, labels, features, n_speakers,
                    spkids, acum_dev_metrics, args)
        for k in acum_dev_metrics.keys():
            writer.add_scalar(
                f"dev_{k}", acum_dev_metrics[k] / dev_batches_qty,
                epoch * dev_batches_qty + i)
        acum_dev_metrics = reset_metrics(acum_dev_metrics)
