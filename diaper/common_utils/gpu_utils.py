#!/usr/bin/env python3

# Copyright 2023 Brno University of Technology (author: Federico Landini)
# Licensed under the MIT license.

from safe_gpu import safe_gpu


def use_gpus(gpus_qty: int) -> safe_gpu.GPUOwner:
    gpu_owner = safe_gpu.GPUOwner(nb_gpus=gpus_qty)
    return gpu_owner
