#!/usr/bin/env python3

# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Copyright 2023 Brno University of Technology (authors: Federico Landini)
# Licensed under the MIT license.


from common_utils.diarization_dataset import KaldiDiarizationDataset
from copy import deepcopy
from torch.utils.data import DataLoader
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple
import numpy as np
import os
import pathlib
import pickle
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


def get_training_dataloaders(
    args: SimpleNamespace
) -> Tuple[DataLoader, DataLoader]:
    train_set = KaldiDiarizationDataset(
        args.train_data_dir,
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
        use_last_samples=args.use_last_samples,
        min_length=args.min_length,
        specaugment=args.specaugment,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.macro_batchsize,
        collate_fn=_convert,
        num_workers=args.num_workers,
        shuffle=True,
        worker_init_fn=_init_fn,
    )

    dev_set = KaldiDiarizationDataset(
        args.valid_data_dir,
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
        use_last_samples=args.use_last_samples,
        min_length=args.min_length,
        specaugment=args.specaugment,
    )
    dev_loader = DataLoader(
        dev_set,
        batch_size=args.macro_batchsize,
        collate_fn=_convert,
        num_workers=1,
        shuffle=False,
        worker_init_fn=_init_fn,
    )

    Y_train, _, _, _, _, _ = train_set.__getitem__(0)
    Y_dev, _, _, _, _, _ = dev_set.__getitem__(0)
    assert Y_train.shape[1] == Y_dev.shape[1], \
        f"Train features dimensionality ({Y_train.shape[1]}) and \
        dev features dimensionality ({Y_dev.shape[1]}) differ."
    assert Y_train.shape[1] == (
        args.feature_dim * (1 + 2 * args.context_size)), \
        f"Expected feature dimensionality of {args.feature_dim} \
        but {Y_train.shape[1]} found."

    return train_loader, dev_loader


def parse_arguments() -> SimpleNamespace:
    parser = yamlargparse.ArgumentParser(
        description='Process features and save them')
    parser.add_argument('-c', '--config', help='config file path',
                        action=yamlargparse.ActionConfigFile)
    parser.add_argument('--context-size', type=int)
    parser.add_argument('--feature-dim', type=int)
    parser.add_argument('--frame-shift', type=int)
    parser.add_argument('--frame-size', type=int)
    parser.add_argument('--input-transform', default='',
                        choices=['logmel', 'logmel_meannorm',
                                 'logmel_meanvarnorm'],
                        help='input normalization transform')
    parser.add_argument('--macro-batchsize', default=1, type=int,
                        help='number of utterances in one macro batch')
    parser.add_argument('--min-length', default=0, type=int,
                        help='Minimum number of frames for the sequences'
                             ' after downsampling.')
    parser.add_argument('--num-frames', type=int,
                        help='number of frames in one utterance')
    parser.add_argument('--num-speakers', type=int,
                        help='maximum number of speakers allowed')
    parser.add_argument('--num-workers', default=1, type=int,
                        help='number of workers in train DataLoader')
    parser.add_argument('--features-output-path', type=str)
    parser.add_argument('--sampling-rate', type=int)
    parser.add_argument('--specaugment', type=bool, default=False)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--subsampling', type=int)
    parser.add_argument('--use-last-samples', default=True, type=bool)
    parser.add_argument('--time-shuffle', action='store_true',
                        help='Shuffle time-axis order before input to the network')
    parser.add_argument('--train-data-dir',
                        help='kaldi-style data dir used for training.')
    parser.add_argument('--valid-data-dir',
                        help='kaldi-style data dir used for validation.')
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

    train_loader, dev_loader = get_training_dataloaders(args)

    train_batches_qty = len(train_loader)
    dev_batches_qty = len(dev_loader)

    outpath = os.path.join(
        args.features_output_path, f"samplingrate{args.sampling_rate}_\
        numframes{args.num_frames}_minlength{args.min_length}_\
        contextsize{args.context_size}_featuredim{args.feature_dim}_\
        frameshift{args.frame_shift}_framesize{args.frame_size}_\
        inputtransform{args.input_transform}_subsampling{args.subsampling}_\
        timeshuffle{args.time_shuffle}")

    pathlib.Path(os.path.join(
        outpath, f"batchsize{args.macro_batchsize}", "train"
        )).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(
        outpath, f"batchsize{args.macro_batchsize}", "dev"
        )).mkdir(parents=True, exist_ok=True)

    sequences_qty = 0
    for i, batch in enumerate(train_loader):
        sequences_qty += len(batch['xs'])
        with open(os.path.join(
            outpath, f"batchsize{args.macro_batchsize}", "train",
                f"macrobatch_{i}.pkl"), 'wb') as f:
            pickle.dump(deepcopy(batch), f)
    print(f"Train sequences: {sequences_qty}")

    sequences_qty = 0
    for i, batch in enumerate(dev_loader):
        sequences_qty += len(batch['xs'])
        with open(os.path.join(
            outpath, f"batchsize{args.macro_batchsize}", "dev",
                f"macrobatch_{i}.pkl"), 'wb') as f:
            pickle.dump(deepcopy(batch), f)
    print(f"Validation sequences: {sequences_qty}")
