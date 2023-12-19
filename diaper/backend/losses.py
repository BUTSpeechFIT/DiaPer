#!/usr/bin/env python3

# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Copyright 2022 Brno University of Technology (authors: Federico Landini, Lukas Burget, Mireia Diez)
# Copyright 2022 AUDIAS Universidad Autonoma de Madrid (author: Alicia Lozano-Diez)
# Copyright 2023 Brno University of Technology (authors: Federico Landini)
# Licensed under the MIT license.

from itertools import permutations
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Beta
from typing import List, Tuple
from torch.nn.functional import logsigmoid
from types import SimpleNamespace
from scipy.optimize import linear_sum_assignment


def pad_labels(ts: torch.Tensor, out_size: int) -> torch.Tensor:
    # pad label's speaker-dim to be model's n_speakers
    ts_padded = []
    for _, t in enumerate(ts):
        if t.shape[1] < out_size:
            # padding
            ts_padded.append(torch.cat((t, -1 * torch.ones((
                t.shape[0], out_size - t.shape[1]))), dim=1))
        elif t.shape[1] > out_size:
            # truncate
            ts_padded.append(t[:, :out_size].float())
        else:
            ts_padded.append(t.float())
    return ts_padded


def pad_labels_zeros(ts: torch.Tensor, out_size: int) -> torch.Tensor:
    # pad label's speaker-dim to be model's n_speakers
    out_size = max(out_size, ts[0].shape[1])
    ts_padded = []
    for _, t in enumerate(ts):
        if t.shape[1] < out_size:
            # padding
            if t[-1, 0] == -1:
                boundary = min(torch.where(t[:, 0] == -1)[0])
                beg = torch.cat((t[:boundary, :], torch.zeros((
                    boundary, out_size - t.shape[1]))), dim=1)
                end = (-1 * torch.ones((t.shape[0] - boundary, out_size)))
                ts_padded.append(torch.cat((beg, end), dim=0))
            else:
                ts_padded.append(torch.cat((t, torch.zeros((
                    t.shape[0], out_size - t.shape[1]
                    ), device=t.device)), dim=1))
        elif t.shape[1] > out_size:
            # truncate
            ts_padded.append(t[:, :out_size].float())
        else:
            ts_padded.append(t.float())
    return ts_padded


def pad_sequence(
    features: List[torch.Tensor],
    labels: List[torch.Tensor],
    seq_len: int
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    features_padded = []
    labels_padded = []
    assert len(features) == len(labels), (
        f"Features and labels in batch were expected to match but got "
        "{len(features)} features and {len(labels)} labels.")
    for i, _ in enumerate(features):
        assert features[i].shape[0] == labels[i].shape[0], (
            f"Length of features and labels were expected to match but got "
            "{features[i].shape[0]} and {labels[i].shape[0]}")
        length = features[i].shape[0]
        if length < seq_len:
            extend = seq_len - length
            features_padded.append(torch.cat((features[i], -torch.ones((
                extend, features[i].shape[1]))), dim=0))
            labels_padded.append(torch.cat((labels[i], -torch.ones((
                extend, labels[i].shape[1]))), dim=0))
        elif length > seq_len:
            raise (f"Sequence of length {length} was received but only "
                   "{seq_len} was expected.")
        else:
            features_padded.append(features[i])
            labels_padded.append(labels[i])
    return features_padded, labels_padded


def pit_loss_multispk(
        logits: List[torch.Tensor],
        target: List[torch.Tensor],
        attractors_logits: torch.Tensor,
        n_speakers: np.ndarray,
        args: SimpleNamespace):
    logits_t = logits.detach().transpose(1, 2)

    permutations = []

    spk_labels = torch.from_numpy(np.asarray([
        [1.0] * n_spk + [0.0] * (target.shape[2] - n_spk)
        for n_spk in n_speakers]))
    if logits.device != torch.device("cpu"):
        spk_labels = spk_labels.to(torch.device("cuda"))

    for i in range(logits.shape[0]):
        if args.shuffle_spk_order:
            perm = torch.randperm(target.shape[2])
            target[i] = target[i, :, perm]
            spk_labels[i] = spk_labels[i, perm]

        # Remove -1 rows
        boundary = min(
            torch.cat((torch.where(target[i, :, 0] == -1)[0].to("cpu"),
                       torch.tensor([(target[i].shape[0])])))
        )
        if args.use_detection_error_rate:
            cost_mx = torch.matmul(1 - torch.sigmoid(logits_t[i, :, :boundary]), target[i, :boundary, :]) \
                    + torch.matmul(torch.sigmoid(logits_t[i, :, :boundary]), 1 - target[i, :boundary, :])
        else:
            cost_mx = (-logsigmoid(logits_t[i, :, :boundary].unsqueeze(0)).bmm(
                target[i, :boundary, :].unsqueeze(0)) -
                logsigmoid(-logits_t[i, :, :boundary].unsqueeze(0)).bmm(
                1-target[i, :boundary, :].unsqueeze(0)))[0]
        pred_alig, ref_alig = linear_sum_assignment(cost_mx.to("cpu"))
        assert (np.all(pred_alig == np.arange(logits.shape[-1])))
        permutations.append(torch.from_numpy(ref_alig))
        # take first columns, in case the model handles less speakers
        target[i, :, :ref_alig.shape[0]] = target[i, :, ref_alig]
        # take first columns, in case the model handles less speakers
        spk_labels[i, :ref_alig.shape[0]] = spk_labels[i, ref_alig]

    # Pick first dimensions, in case they mismatch because the data has more
    # speakers than the model has attractors
    logits_matched = logits[:, :, :min(logits.shape[2], target.shape[2])]
    target_matched = target[:, :, :min(logits.shape[2], target.shape[2])]
    activation_loss = F.binary_cross_entropy_with_logits(
        logits_matched, target_matched, reduction='none')
    activation_loss[torch.where(target_matched == -1)] = 0
    # normalize by sequence length
    activation_loss = torch.sum(activation_loss, axis=1) / (target_matched != -1).sum(axis=1)
    if args.norm_loss_per_spk:
        # normalize per speaker first
        activation_loss = torch.sum(activation_loss, axis=1) / torch.max(
            torch.sum(spk_labels, axis=1),
            torch.ones(spk_labels.shape[0], device=logits.device))
    # normalize in batch
    activation_loss = torch.mean(activation_loss)

    diff = torch.sigmoid(logits_matched) - target_matched
    diff[torch.where(target_matched == -1)] = 0
    # negative values will be misses and positives will be false alarms
    diff_sum = diff.sum(axis=2)
    miss = -diff_sum[torch.where(diff_sum < 0)].sum()
    fa = diff_sum[torch.where(diff_sum > 0)].sum()
    conf = ((torch.abs(diff).sum(axis=1) - torch.abs(diff.sum(axis=1)))/2).sum()
    activation_loss_DER = (miss + fa + conf) / target_matched[torch.where(target_matched > 0)].sum()
    # normalize by batch size
    activation_loss_DER = activation_loss_DER / target_matched.shape[0]

    # Pick first dimensions, in case they mismatch because the data has more
    # speakers than the model has attractors
    attractor_existence_loss = F.binary_cross_entropy_with_logits(
        attractors_logits[:, :min(attractors_logits.shape[1], spk_labels.shape[1])],
        spk_labels[:, :min(attractors_logits.shape[1], spk_labels.shape[1])],
        reduction='mean')

    return (activation_loss, activation_loss_DER, attractor_existence_loss,
            torch.stack(permutations).to(logits.device))


def get_silence_probs(ys_silence_probs: torch.Tensor):
    # The probability of silence in the frame is the product of the
    # probability that each speaker is silent
    silence_prob = torch.prod(ys_silence_probs, 2, keepdim=True)
    return silence_prob


def vad_loss(logits: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
    # Take from reference ts only the speakers that do not correspond to -1
    # (-1 are padded frames), if the sum of their values is >0 there is speech
    silence_ts = (torch.sum((ts != -1)*ts, 2, keepdim=True) == 0).float()
    silence_prob = get_silence_probs(1-torch.sigmoid(logits))
    # Estimate the loss. size=[batch_size, num_frames, 1]
    loss = F.binary_cross_entropy(silence_prob, silence_ts, reduction='none')
    # "torch.max(ts, 2, keepdim=True)[0]" keeps the maximum along speaker dim
    # Invalid frames in the sequence (padding) will be -1, replace those
    # invalid positions by 0 so that those losses do not count
    loss[torch.where(torch.max(ts, 2, keepdim=True)[0] < 0)] = 0
    # normalize by sequence length
    # "torch.sum(loss, axis=1)" gives a value per sequence
    # if torch.mean(ts,axis=2)==-1 then all speakers were invalid in the frame,
    # therefore we should not account for it
    # ts is size [batch_size, num_frames, num_spks]
    loss = torch.sum(loss, axis=1) / (torch.mean(ts, axis=2) != -1).sum(axis=1, keepdims=True)
    # normalize in batch for all speakers
    loss = torch.mean(loss)
    return loss


def get_nooverlap_probs(p_sil: torch.Tensor):
    # Imagine the case of 3 speakers for one specific frame,
    # probability of no overlap will be
    # p_sil1 * p_sil2 * p_sil3 +
    # p_sp1  * p_sil2 * p_sil3 +
    # p_sil1 * p_sp2  * p_sil3 +
    # p_sil1 * p_sil2 * p_sp3
    # Then, to construct such matrix, we use prob_mask and inv_prob_mask
    # which will look like (respectively)
    # 0 0 0        1 1 1
    # 1 0 0        0 1 1
    # 0 1 0        1 0 1
    # 0 0 1        1 1 0
    # Then, prob_mask * (1 - p_sil)[0,0] + inv_prob_mask * p_sil[0,0] will
    # give us the following
    # (note that the first two dimensions are batchsize and sequence_length
    # and the unsqueeze operators are there because of this)
    # p_sil1 p_sil2 p_sil3
    # p_sp1  p_sil2 p_sil3
    # p_sil1 p_sp2  p_sil3
    # p_sil1 p_sil2 p_sp3
    # So we only need to multiply along the last dimension and then sum along
    # the second to last dimension. Then, we will obtain p_nooverlap of
    # dimensions (batchsize, sequence_length)
    mask = torch.cat((torch.zeros(1, p_sil.shape[2], device=p_sil.device),
                      torch.eye(p_sil.shape[2], device=p_sil.device)), dim=0)
    prob_mask = mask.unsqueeze(0).unsqueeze(0).expand(
        p_sil.shape[0], p_sil.shape[1], -1, -1)
    inv_prob_mask = 1 - prob_mask
    p_nooverlap = torch.sum(torch.prod(
        prob_mask * (1 - p_sil).unsqueeze(2) +
        inv_prob_mask * p_sil.unsqueeze(2), dim=3), dim=2).unsqueeze(2)
    return p_nooverlap


def osd_loss(ys: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
    aux = torch.clone(ts)
    aux[aux < 0] = 0  # nullify -1 positions
    nooverlap_ts = (torch.sum(aux, 2, keepdim=True) < 2).float()
    p_nooverlap = get_nooverlap_probs(1 - torch.sigmoid(ys))
    # estimate the loss. size=[batch_size, num_framess, 1]
    loss = F.binary_cross_entropy(p_nooverlap, nooverlap_ts, reduction='none')

    # replace invalid positions by 0 so that those losses do not count
    loss[torch.where(torch.max(ts, 2, keepdim=True)[0] < 0)] = 0
    # normalize by sequence length (not counting padding positions)
    #  torch.sum(loss, axis=1) gives a value per batch.
    #  if torch.mean(ts,axis=2)==-1, all speakers were invalid in the frame,
    #  therefore we should not account for it.
    #  ts is size [batch_size, num_frames, num_spks]
    loss = torch.sum(loss, axis=1) / (torch.mean(ts, axis=2) != -1).sum(axis=1, keepdims=True)
    # normalize in batch so that all sequences count the same
    loss = torch.mean(loss)
    return loss


def get_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    n_speakers: List[int],
    attractors_logits: torch.Tensor,
    model: torch.nn.Module,
    attractors: torch.Tensor,
    total_n_speakers: int,
    spkid_labels: torch.Tensor,
    args: SimpleNamespace
) -> Tuple[torch.Tensor, torch.Tensor]:
    max_n_speakers = max(n_speakers)
    ts_padded = pad_labels_zeros(target, max_n_speakers)
    ts_padded = torch.stack(ts_padded)
    logits_padded = pad_labels_zeros(logits, max_n_speakers)
    logits_padded = torch.stack(logits_padded)

    spk_labels = torch.from_numpy(np.asarray([
        [1.0] * n_spk + [0.0] * (target.shape[2] - n_spk)
        for n_spk in n_speakers]))
    if attractors_logits.device != torch.device("cpu"):
        spk_labels = spk_labels.to(torch.device("cuda"))

    (activation_loss, activation_loss_DER,
        attractor_existence_loss, permutations) = pit_loss_multispk(
        logits_padded, ts_padded, attractors_logits, n_speakers, args)
    if not (args.speakerid_loss == ''):
        spkid_loss = speaker_identification_loss(
            spkid_labels, spk_labels, model, attractors,
            total_n_speakers, permutations)
    else:
        spkid_loss = 0
    if args.att_qty_loss_weight == 0:
        att_qty_loss = 0
    else:
        att_qty_loss = get_attractor_quantity_loss(attractors_logits, n_speakers)
    if args.vad_loss_weight == 0:
        vad_loss_value = 0
    else:
        vad_loss_value = vad_loss(logits, target)
    if args.osd_loss_weight == 0:
        osd_loss_value = 0
    else:
        osd_loss_value = osd_loss(logits, target)

    return (
        activation_loss,
        activation_loss_DER,
        attractor_existence_loss,
        att_qty_loss,
        vad_loss_value,
        osd_loss_value,
        spkid_loss
        )


def speaker_identification_loss(
    spkid_labels, spk_labels, model, attractors,
    total_n_speakers, permutations
):
    selected_att = []
    indices = []
    for i in range(spk_labels.shape[0]):
        assert sum(spk_labels[i]) <= len(spkid_labels[i])
        masked_spk_labels = (
            spk_labels[i].to(int) * torch.tensor(
                spkid_labels[i] + [0] *
                (spk_labels[i].shape[0] - len(spkid_labels[i]))
                ).to(int).to(spk_labels.device))
        masked_spk_labels = masked_spk_labels[permutations[i]]
        spk_labels_i = spk_labels[i, permutations[i]]
        for valid_pos in torch.where(spk_labels_i == 1)[0]:
            selected_att.append(attractors[i, valid_pos])
            indices.append(masked_spk_labels[valid_pos])
    loss_function = torch.nn.CrossEntropyLoss()
    selected_att = torch.stack(selected_att)
    indices = torch.stack(indices)
    if len(indices) > 0:
        return loss_function(
            model.module.get_speaker_logits(selected_att, indices),
            indices) / indices.shape[0]
    else:
        return 0


def get_attractor_quantity_loss(
    attractors_logits: torch.Tensor,
    n_speakers: List[int]
):
    criterion = torch.nn.MSELoss()
    attractor_loss = criterion(
        torch.sum(torch.sigmoid(attractors_logits.double()), dim=1),
        torch.from_numpy(n_speakers).to(attractors_logits.device).double()
    )
    return attractor_loss
