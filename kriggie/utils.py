
# -*- coding: utf-8 -*-

import torch
import numpy as np

TORCH_FLOAT_TYPE = torch.float32
NP_FLOAT_TYPE = np.float32

NUM_JITTER_TRIALS = 4
STARTING_JITTER = 1e-6
JITTER_INCREASER = 10

SMALL_CONSTANT = 1e-11

def try_func_with_increasing_jitter(func, input, exception_to_catch):

    try:
        return func(input)
    except exception_to_catch as ex:
        pass

    diag_last_two = torch.zeros_like(input)
    shape_m1 = diag_last_two.shape[-1]
    diag_last_two.view(-1)[::shape_m1 + 1] = 1.

    new_jitter = STARTING_JITTER * diag_last_two
    for _ in range(NUM_JITTER_TRIALS):
        try:
            return func(input + new_jitter)
        except exception_to_catch as ex:
            new_jitter = new_jitter * JITTER_INCREASER

    raise RuntimeError("Cholesky failed even with Jitter")







