#!/usr/bin/env python3

# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Copyright 2023 Brno University of Technology (author: Federico Landini)
# Licensed under the MIT license.

from os.path import isfile, join

from backend.losses import (
    pad_labels,
    pad_sequence,
    pit_loss_multispk,
    vad_loss,
)
from backend.updater import (
    NoamOpt,
    setup_optimizer,
)
from pathlib import Path
from transformers.models.perceiver.configuration_perceiver import PerceiverConfig
from transformers.models.perceiver.modeling_perceiver import PerceiverEncoder
from types import SimpleNamespace
from typing import Dict, List, Tuple
import copy
import logging
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim


def save_checkpoint(
    args,
    epoch: float,
    model: torch.nn.Module,
    optimizer: NoamOpt,
    loss: torch.Tensor
) -> None:
    Path(f"{args.output_path}/models").mkdir(parents=True, exist_ok=True)

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss},
        f"{args.output_path}/models/checkpoint_{epoch}.tar"
    )


def load_checkpoint(args: SimpleNamespace, filename: str):
    model = get_model(args)
    optimizer = setup_optimizer(args, model)

    assert isfile(filename), \
        f"File {filename} does not exist."
    checkpoint = torch.load(filename, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch, model, optimizer, loss


def load_initmodel(args: SimpleNamespace):
    return load_checkpoint(args, args.initmodel)


def get_model(args: SimpleNamespace) -> torch.nn.Module:
    args.in_size = args.feature_dim * (1 + 2 * args.context_size)
    if args.model_type == 'AttractorPerceiver':
        model = AttractorPerceiver(args)
    else:
        raise ValueError('Possible model_type is "AttractorPerceiver"')
    return torch.nn.DataParallel(model)


def average_checkpoints(
    device: torch.device,
    model: torch.nn.Module,
    models_path: str,
    epochs: str
) -> torch.nn.Module:
    epochs = parse_epochs(epochs)
    states_dict_list = []
    for e in epochs:
        copy_model = copy.deepcopy(model)
        checkpoint = torch.load(join(
            models_path,
            f"checkpoint_{e}.tar"), map_location=device)
        copy_model.load_state_dict(checkpoint['model_state_dict'])
        states_dict_list.append(copy_model.state_dict())
    avg_state_dict = average_states(states_dict_list, device)
    avg_model = copy.deepcopy(model)
    avg_model.load_state_dict(avg_state_dict)
    if device == torch.device('cpu'):
        avg_model = avg_model.module
    return avg_model


def average_states(
    states_list: List[Dict[str, torch.Tensor]],
    device: torch.device,
) -> List[Dict[str, torch.Tensor]]:
    qty = len(states_list)
    avg_state = states_list[0]
    for i in range(1, qty):
        for key in avg_state:
            avg_state[key] = avg_state[key].to(device)
            avg_state[key] += states_list[i][key].to(device)

    for key in avg_state:
        avg_state[key] = avg_state[key] / qty
    return avg_state


def parse_epochs(string: str) -> List[int]:
    parts = string.split(',')
    res = []
    for p in parts:
        if '-' in p:
            interval = p.split('-')
            res.extend(range(int(interval[0])+1, int(interval[1])+1))
        else:
            res.append(int(p))
    return res


class VanillaSpeakerLayer(torch.nn.Module):

    def __init__(self, in_dim: int, out_dim: int, device: torch.device):
        self.device = device
        super(VanillaSpeakerLayer, self).__init__()
        self.weights = torch.nn.Parameter(
            torch.normal(torch.zeros((in_dim, out_dim), device=self.device),
                         0.01*torch.ones((in_dim, out_dim), device=self.device)
                         ), requires_grad=True)

    def forward(self, x: torch.Tensor,
                spkid_labels: torch.Tensor = None) -> torch.Tensor:
        return torch.matmul(x, self.weights)


class ArcfaceSpeakerLayer(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int,
                 device: torch.device, s=30., m=0.30):
        self.device = device
        super(ArcfaceSpeakerLayer, self).__init__()
        self.s = s
        self.m = m
        self.out_dim = out_dim
        self.weights = torch.nn.Parameter(
            torch.normal(torch.zeros((in_dim, out_dim), device=self.device),
                         0.01*torch.ones((in_dim, out_dim), device=self.device)
                         ), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.weights)

    def forward(self, x: torch.Tensor,
                spkid_labels: torch.Tensor) -> torch.Tensor:
        cos_m = math.cos(self.m)
        sin_m = math.sin(self.m)
        mm = sin_m * self.m
        threshold = math.cos(math.pi - self.m)
        # inputs and weights norm
        x_norm = torch.norm(x, dim=1)
        x = torch.div(x, x_norm.unsqueeze(1))
        weights_norm = torch.norm(self.weights, dim=0)
        weights = self.weights / weights_norm
        # cos(theta+m)
        cos_t = torch.matmul(x, weights)
        cos_t2 = torch.square(cos_t)
        sin_t2 = 1 - cos_t2
        sin_t = torch.sqrt(sin_t2)
        cos_mt = self.s * (cos_t * cos_m - sin_t * sin_m)

        # this condition controls the theta+m should in range [0, pi]
        #      0<=theta+m<=pi
        #     -m<=theta<=pi-m
        cond_v = cos_t - threshold
        relu = torch.nn.ReLU()
        cond = relu(cond_v).to(bool)

        keep_val = self.s * (cos_t - mm)
        cos_mt_temp = torch.where(cond, cos_mt, keep_val)

        mask = torch.nn.functional.one_hot(
            spkid_labels, num_classes=self.out_dim)
        inv_mask = 1 - mask

        s_cos_t = self.s * cos_t

        output = s_cos_t * inv_mask + cos_mt_temp * mask
        return output


class MultiHeadSelfAttention(torch.nn.Module):
    """ Multi head self-attention layer
    """
    def __init__(
        self,
        device: torch.device,
        n_units: int,
        h: int,
        dropout: float
    ) -> None:
        super(MultiHeadSelfAttention, self).__init__()
        self.device = device
        self.linearQ = torch.nn.Linear(n_units, n_units, device=self.device)
        self.linearK = torch.nn.Linear(n_units, n_units, device=self.device)
        self.linearV = torch.nn.Linear(n_units, n_units, device=self.device)
        self.linearO = torch.nn.Linear(n_units, n_units, device=self.device)
        self.d_k = n_units // h
        self.h = h
        self.dropout = dropout
        self.att = None  # attention for plot

    def __call__(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        # x: (BT, F)
        q = self.linearQ(x).reshape(batch_size, -1, self.h, self.d_k)
        k = self.linearK(x).reshape(batch_size, -1, self.h, self.d_k)
        v = self.linearV(x).reshape(batch_size, -1, self.h, self.d_k)
        scores = torch.matmul(q.permute(0, 2, 1, 3), k.permute(0, 2, 3, 1)) \
            / np.sqrt(self.d_k)
        # scores: (B, h, T, T)
        self.att = F.softmax(scores, dim=3)
        p_att = F.dropout(self.att, self.dropout, training=self.training)
        x = torch.matmul(p_att, v.permute(0, 2, 1, 3))
        x = x.permute(0, 2, 1, 3).reshape(-1, self.h * self.d_k)
        return self.linearO(x)


class PositionwiseFeedForward(torch.nn.Module):
    """ Positionwise feed-forward layer
    """
    def __init__(
        self,
        device: torch.device,
        n_units: int,
        d_units: int,
        dropout: float
    ) -> None:
        super(PositionwiseFeedForward, self).__init__()
        self.device = device
        self.linear1 = torch.nn.Linear(n_units, d_units, device=self.device)
        self.linear2 = torch.nn.Linear(d_units, n_units, device=self.device)
        self.dropout = dropout

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(F.dropout(F.relu(self.linear1(x)),
                                      self.dropout, training=self.training))


class PerceiverBlock(torch.nn.Module):

    def __init__(
        self,
        config: PerceiverConfig,
        device: torch.device,
        n_blocks: int,
        d_latents: int,
    ) -> None:
        """ Perceiver block where difference PerceiverEncoders are chained
        """
        self.device = device
        super(PerceiverBlock, self).__init__()
        self.encoder_layers = torch.nn.ModuleList()
        for _ in range(n_blocks):
            p = PerceiverEncoder(config, kv_dim=d_latents).to(self.device)
            self.encoder_layers.append(p)

    def __call__(
        self,
        latents: torch.Tensor,
        inputs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        per_prcvblock_latents = []
        for _, layer_module in enumerate(self.encoder_layers):
            output = layer_module.forward(latents, inputs=inputs)
            latents = output.last_hidden_state
            per_prcvblock_latents.append(latents)
        return torch.stack(per_prcvblock_latents)


class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model: int, max_len: int, dropout: float = 0.1):
        # maxlen 36000 steps at 0.1s per step is 1 hour
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             (-math.log(float(2*max_len)) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)[:, :pe[:, 0, 1::2].shape[1]]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = torch.swapaxes(x, 0, 1)
        x = x + self.pe[:x.size(0)].to(x.device)
        x = torch.swapaxes(x, 0, 1)
        return self.dropout(x)


class LinearCombination(torch.nn.Module):

    def __init__(self, in_dim: int, out_dim: int, device: torch.device):
        self.device = device
        super(LinearCombination, self).__init__()
        self.weights = torch.nn.Parameter(torch.normal(
            torch.zeros((in_dim, out_dim), device=self.device),
            0.01*torch.ones((in_dim, out_dim), device=self.device)),
            requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights_normalized = torch.nn.functional.softmax(self.weights, dim=0)
        entropy_term = (
            weights_normalized * torch.log(weights_normalized)
            ).mean(dim=0).sum()
        return torch.matmul(
            x.permute(0, 2, 1),
            weights_normalized).permute(0, 2, 1), entropy_term


class Dummy(torch.nn.Module):

    def __init__(self, device: torch.device):
        self.device = device
        super(Dummy, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 2, 1)


class AttractorPerceiver(torch.nn.Module):

    def __init__(
        self,
        args: SimpleNamespace,
    ) -> None:
        """ Perceiver latent space (path) used for attractors.
        Input are (optionally) stacked and downsampled acoustic frames
        """

        self.device = args.device
        super(AttractorPerceiver, self).__init__()
        self.use_pre_crossattention = args.use_pre_crossattention
        if self.use_pre_crossattention:
            self.pre_crossattention = torch.nn.MultiheadAttention(
                args.d_latents, args.pre_xa_heads,
                batch_first=True, device=self.device)
        else:
            self.pre_crossattention = None
        self.use_frame_selfattention = args.use_frame_selfattention
        if self.use_frame_selfattention:
            self.linear_in = torch.nn.Linear(
                args.in_size, args.d_latents, device=self.device)
            self.lnorm_in = torch.nn.LayerNorm(
                args.d_latents, device=self.device)
            self.n_layers = args.frame_encoder_layers
            self.dropout = args.dropout_frames
            for i in range(self.n_layers):
                setattr(
                    self,
                    '{}{:d}'.format("lnorm1_", i),
                    torch.nn.LayerNorm(args.d_latents, device=self.device)
                )
                setattr(
                    self,
                    '{}{:d}'.format("self_att_", i),
                    MultiHeadSelfAttention(self.device, args.d_latents,
                                           args.frame_encoder_heads,
                                           args.dropout_frames)
                )
                setattr(
                    self,
                    '{}{:d}'.format("lnorm2_", i),
                    torch.nn.LayerNorm(args.d_latents, device=self.device)
                )
                setattr(
                    self,
                    '{}{:d}'.format("ff_", i),
                    PositionwiseFeedForward(self.device, args.d_latents,
                                            args.frame_encoder_units,
                                            args.dropout_frames)
                )
            self.lnorm_out = torch.nn.LayerNorm(
                args.d_latents, device=self.device)
            self.condition_frame_encoder = args.condition_frame_encoder
            if self.condition_frame_encoder:
                self.W = torch.nn.Parameter(
                    torch.normal(torch.zeros((
                        args.d_latents, args.d_latents),
                        device=self.device), 0.01*torch.ones(
                        (args.d_latents, args.d_latents),
                        device=self.device)), requires_grad=True)
            else:
                self.condition_frame_encoder = None
        else:
            self.frame_encoder = torch.nn.Linear(
                args.in_size, args.d_latents, device=self.device)
        self.use_posenc = args.use_posenc
        if self.use_posenc:
            self.pos_encoder = PositionalEncoding(
                args.d_latents, args.posenc_maxlen)
        else:
            self.pos_encoder = None
        self.config = PerceiverConfig(
            qk_channels=args.d_latents*args.n_sa_heads_attractors,
            v_channels=args.d_latents,
            num_latents=args.n_latents,
            d_latents=args.d_latents,
            d_model=args.in_size,
            num_blocks=args.n_internal_blocks_attractors,
            num_self_attends_per_block=args.n_selfattends_attractors,
            num_self_attention_heads=args.n_sa_heads_attractors,
            num_cross_attention_heads=args.n_xa_heads_attractors,
            attention_probs_dropout_prob=args.dropout_attractors,
        )
        self.latent_attractors = torch.nn.Parameter(torch.normal(
            torch.zeros((args.n_latents, args.d_latents), device=self.device),
            0.01*torch.ones((args.n_latents, args.d_latents), device=self.device)
            ), requires_grad=True)
        self.encoder_attractors = PerceiverBlock(
            self.config, self.device, args.n_blocks_attractors, args.d_latents)
        self.lat2att = args.latents2attractors
        if self.lat2att == 'weighted_average':
            self.latents2attractors = LinearCombination(
                args.n_latents, args.n_attractors, device=self.device)
        elif self.lat2att == 'linear':
            self.latents2attractors = torch.nn.Linear(
                args.n_latents, args.n_attractors, device=self.device)
        else:
            assert args.n_latents == args.n_attractors, \
                f"The number of latents: {args.n_latents} and attractors: \
                {args.n_attractors} must match for this option of lat2att: \
                {self.lat2att}"
            self.latents2attractors = Dummy(device=self.device)
        # does not need to be d_latents, just a reasonable dimension
        self.counter = torch.nn.Linear(args.d_latents, 1, device=self.device)
        self.detach_attractor_loss = args.detach_attractor_loss
        self.attractor_frame_comparison = args.attractor_frame_comparison
        if self.attractor_frame_comparison == 'xattention':
            self.frame_activate = torch.nn.MultiheadAttention(
                args.d_latents, 1, batch_first=True, device=self.device)
        else:
            self.frame_activate = None
        if args.speakerid_loss == "arcface":
            self.speaker_layer = ArcfaceSpeakerLayer(
                args.d_latents, args.speakerid_num_speakers,
                device=self.device)
        elif args.speakerid_loss == "vanilla":
            self.speaker_layer = VanillaSpeakerLayer(
                args.d_latents, args.speakerid_num_speakers,
                device=self.device)
        else:
            self.speaker_layer = None
        self.context_activations = args.context_activations
        if self.context_activations:
            self.contextualizer_lstm = torch.nn.LSTM(
                input_size=args.n_attractors,
                hidden_size=args.d_latents,
                num_layers=1,
                dropout=args.dropout_attractors,
                batch_first=True,
                device=self.device)
            self.contextualizer_linear = torch.nn.Linear(
                args.d_latents,
                args.n_attractors,
                device=self.device)
        else:
            self.contextualizer_lstm = None
            self.contextualizer_linear = None
        self.count_parameters(args)

    def forward(
        self,
        inputs: torch.Tensor,
        args: SimpleNamespace
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        all_frame_embs = []
        per_frameenclayer_ys_logits = []
        per_frameenclayer_atts_logits = []
        per_frameenclayer_atts = []
        # inputs: (B, T, F)
        pad_shape = inputs.shape
        # emb: (B*T, E)

        # x: (B, T, F) ... batch, time, (mel)freq
        BT_size = inputs.shape[0] * inputs.shape[1]
        # e: (BT, F)
        e = self.linear_in(inputs.reshape(BT_size, -1))
        # Encoder stack
        for i in range(self.n_layers):
            # layer normalization
            e = getattr(self, '{}{:d}'.format("lnorm1_", i))(e)
            # self-attention
            s = getattr(self, '{}{:d}'.format("self_att_", i))(e, inputs.shape[0])
            # residual
            e = e + F.dropout(s, self.dropout, training=self.training)
            # layer normalization
            e = getattr(self, '{}{:d}'.format("lnorm2_", i))(e)
            # positionwise feed-forward
            s = getattr(self, '{}{:d}'.format("ff_", i))(e)
            # residual
            e = e + F.dropout(s, self.dropout, training=self.training)
            e = self.lnorm_out(e)
            e = e.reshape(pad_shape[0], pad_shape[1], -1)

            (
                per_prcvblock_latents,
                per_prcvblock_attractors,
                per_prcvblock_attractors_logits,
                per_prcvblock_l2a_entropy_term
            ) = self.get_attractors(e.clone())
            per_frameenclayer_atts_logits.append(
                per_prcvblock_attractors_logits[-1])
            per_frameenclayer_atts.append(per_prcvblock_attractors[-1])
            if self.condition_frame_encoder:
                if args.length_normalize:
                    atts = torch.nn.functional.normalize(
                        per_prcvblock_attractors[-1], dim=-1)
                    logits = 20 * torch.matmul(
                        torch.nn.functional.normalize(
                            e.clone(), dim=-1), atts.permute(0, 2, 1))
                else:
                    atts = per_prcvblock_attractors[-1]
                    logits = torch.matmul(e.clone(), atts.permute(0, 2, 1))
                if self.context_activations:
                    logits = self.contextualizer_lstm(logits)[0]
                    logits = self.contextualizer_linear(logits)
                per_frameenclayer_ys_logits.append(logits)
                condition = torch.matmul(torch.matmul(
                    torch.sigmoid(logits), atts), self.W)
                e += condition
            if (not self.condition_frame_encoder) and (i == self.n_layers - 1):
                if args.length_normalize:
                    atts = torch.nn.functional.normalize(
                        per_prcvblock_attractors[-1], dim=-1)
                    logits = 20 * torch.matmul(
                        torch.nn.functional.normalize(
                            e.clone(), dim=-1), atts.permute(0, 2, 1))
                else:
                    atts = per_prcvblock_attractors[-1]
                    logits = torch.matmul(e.clone(), atts.permute(0, 2, 1))
                if self.context_activations:
                    logits = self.contextualizer_lstm(logits)[0]
                    logits = self.contextualizer_linear(logits)
                per_frameenclayer_ys_logits.append(logits)
            e = e.reshape(BT_size, -1)
            all_frame_embs.append(e.reshape(pad_shape[0], pad_shape[1], -1))
        # output: (BT, F)

        # emb: [(T, E), ...]
        e = e.reshape(pad_shape[0], pad_shape[1], -1)
        if args.use_posenc:
            e = self.pos_encoder(e)

        per_prcvblock_ys_logits = []
        for i in range(len(per_prcvblock_latents)):
            if args.attractor_frame_comparison == 'xattention':
                ys_logits = self.frame_activate(
                    per_prcvblock_attractors[i],
                    all_frame_embs[-1],
                    all_frame_embs[-1])[1].transpose(2, 1)
            else:
                if args.length_normalize:
                    ys_logits = 20 * torch.matmul(
                        torch.nn.functional.normalize(e, dim=-1),
                        torch.nn.functional.normalize(
                            per_prcvblock_attractors[i].permute(0, 2, 1),
                            dim=1)
                        )
                else:
                    ys_logits = torch.matmul(
                        e, per_prcvblock_attractors[i].permute(0, 2, 1))
            if self.context_activations:
                ys_logits = self.contextualizer_lstm(ys_logits)[0]
                ys_logits = self.contextualizer_linear(ys_logits)
            per_prcvblock_ys_logits.append(ys_logits)

        all_frame_embs = torch.permute(
            torch.stack(all_frame_embs), (1, 2, 3, 0))
        per_frameenclayer_ys_logits = torch.permute(torch.stack(
            per_frameenclayer_ys_logits), (1, 2, 3, 0))
        per_frameenclayer_atts_logits = torch.permute(torch.stack(
            per_frameenclayer_atts_logits), (1, 2, 0))
        per_frameenclayer_atts = torch.permute(torch.stack(
            per_frameenclayer_atts), (1, 2, 3, 0))
        per_prcvblock_ys_logits = torch.permute(torch.stack(
            per_prcvblock_ys_logits), (1, 2, 3, 0))
        per_prcvblock_attractors_logits = torch.permute(torch.stack(
            per_prcvblock_attractors_logits), (1, 2, 0))
        per_prcvblock_attractors = torch.permute(torch.stack(
            per_prcvblock_attractors), (1, 2, 3, 0))
        per_prcvblock_l2a_entropy_term = torch.stack(
            per_prcvblock_l2a_entropy_term)
        per_prcvblock_latents = torch.permute(
            per_prcvblock_latents, (1, 2, 3, 0))

        return (
            all_frame_embs,
            per_frameenclayer_ys_logits,
            per_frameenclayer_atts_logits,
            per_frameenclayer_atts,
            per_prcvblock_ys_logits,
            per_prcvblock_attractors_logits,
            per_prcvblock_attractors,
            per_prcvblock_l2a_entropy_term,
            per_prcvblock_latents
            )

    def get_speaker_logits(
        self, attractors: torch.Tensor,
        spkid_labels: torch.Tensor
    ) -> torch.Tensor:
        if self.speaker_layer is None:
            return 0
        else:
            return self.speaker_layer(attractors, spkid_labels)

    def get_attractors(
        self,
        frame_embs: torch.Tensor
    ):

        per_prcvblock_attractors = []
        per_prcvblock_logits = []
        per_prcvblock_l2a_entropy_term = []
        latents = self.latent_attractors.repeat((frame_embs.shape[0], 1, 1))
        if self.use_pre_crossattention:
            latents = self.pre_crossattention(
                latents, frame_embs, frame_embs)[0]
        per_prcvblock_latents = self.encoder_attractors(
            latents, inputs=frame_embs)
        if self.device != torch.device("cpu"):
            self.encoder_attractors = self.encoder_attractors.to(
                torch.device("cuda"))
            per_prcvblock_latents = per_prcvblock_latents.to(
                torch.device("cuda"))
        for i in range(len(per_prcvblock_latents)):
            if type(self.latents2attractors) is LinearCombination:
                attractors_i, l2a_entropy_term_i = self.latents2attractors(
                    per_prcvblock_latents[i])
            else:
                attractors_i = self.latents2attractors(
                    per_prcvblock_latents[i]).transpose(1, 2)
                l2a_entropy_term_i = torch.zeros(1)[0]
            if self.detach_attractor_loss:
                attractors_i = attractors_i.detach()
            per_prcvblock_attractors.append(attractors_i)
            per_prcvblock_l2a_entropy_term.append(l2a_entropy_term_i)
            per_prcvblock_logits.append(torch.squeeze(
                self.counter(attractors_i), 2))
        return (per_prcvblock_latents, per_prcvblock_attractors,
                per_prcvblock_logits, per_prcvblock_l2a_entropy_term)

    def count_parameters(self, args: SimpleNamespace):
        total_params = 0
        frame_encoder_modules = (
            [self.linear_in, self.lnorm_in, self.lnorm_out] +
            [getattr(self, '{}{:d}'.format("lnorm1_", i)) for i in range(self.n_layers)] +
            [getattr(self, '{}{:d}'.format("self_att_", i)) for i in range(self.n_layers)] +
            [getattr(self, '{}{:d}'.format("lnorm2_", i)) for i in range(self.n_layers)] +
            [getattr(self, '{}{:d}'.format("ff_", i)) for i in range(self.n_layers)]
            )
        if self.condition_frame_encoder:
            frame_encoder_modules += [self.W]
        modules = [self.pre_crossattention, self.pos_encoder,
                   self.latent_attractors, self.encoder_attractors,
                   self.latents2attractors, self.counter, self.frame_activate,
                   self.speaker_layer, self.contextualizer_lstm,
                   self.contextualizer_linear]
        names = ['pre_crossattention', 'pos_encoder', 'latent_attractors',
                 'encoder_attractors', 'latents2attractors', 'counter',
                 'frame_activate', 'speaker_layer', 'contextualizer_lstm',
                 'contextualizer_linear']
        for i in range(len(names)):
            if not (modules[i] is None):
                params = 0
                if torch.is_tensor(modules[i]):
                    if not modules[i].requires_grad:
                        continue
                    params += modules[i].numel()
                else:
                    for name, parameter in modules[i].named_parameters():
                        if not parameter.requires_grad:
                            continue
                        params += parameter.numel()
                print(names[i], params)
                total_params += params
        frame_encoder_params = 0
        for i in range(len(frame_encoder_modules)):
            if not (frame_encoder_modules[i] is None):
                params = 0
                if torch.is_tensor(frame_encoder_modules[i]):
                    if not frame_encoder_modules[i].requires_grad:
                        continue
                    params += frame_encoder_modules[i].numel()
                else:
                    for name, parameter in frame_encoder_modules[i].named_parameters():
                        if not parameter.requires_grad:
                            continue
                        params += parameter.numel()
                frame_encoder_params += params
        print("frame_encoder", frame_encoder_params)
        print(f"Total trainable parameters: \
            {total_params+frame_encoder_params}")
