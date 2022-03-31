from attention_xl.diengine_attention_xl import AttentionXL as diengine_attention
from attention_xl.sooftware_attention_xl import RelativeMultiHeadAttention as sooftware_attention
from attention_xl.labmlai_attention_xl import  RelativeMultiHeadAttention as labmlai_attention
from attention_xl.huggingface_attention_xl import RelPartialLearnableMultiHeadAttn as huggingface_attention
from attention_xl.labmlai_attention_xl_layer import TransformerXLLayer as labmlai_attention_layer
from easydict import EasyDict as edict
import torch
import timeit
import random
import numpy as np
from torch import nn


def benchmark_diengine(hparams, iters=100, warmup=10, device='cuda'):
    x = torch.rand((hparams.seq_len, hparams.bs, hparams.dim_size))
    memory_input = torch.rand((hparams.seq_len+hparams.memory_len, hparams.bs, hparams.dim_size))
    pos_embedding = torch.rand(hparams.seq_len+hparams.memory_len, 1, hparams.dim_size)
    u, v = (
        torch.nn.Parameter(torch.zeros(hparams.head_num, hparams.head_dim)),
        torch.nn.Parameter(torch.zeros(hparams.head_num, hparams.head_dim)),
    )
    attention = diengine_attention(hparams.dim_size, hparams.head_dim,
                                   hparams.head_num, nn.Dropout(0.))
    attention.to(device)
    input = [x, pos_embedding, memory_input, u, v]
    input = [item.to(device) for item in input]
    for i in range(iters + warmup):
        if i == warmup_iter:
            t0 = timeit.default_timer()
        out = attention.forward(*input)
    t1 = timeit.default_timer()
    return t1-t0


def benchmark_sooftware(hparams, iters=100, warmup=10, device='cuda'):
    q = torch.rand((hparams.bs, hparams.seq_len, hparams.head_num * hparams.head_dim))
    k = torch.rand((hparams.bs, hparams.seq_len, hparams.head_num * hparams.head_dim))
    v = torch.rand((hparams.bs, hparams.seq_len, hparams.head_num * hparams.head_dim))
    pos_embedding = torch.rand((hparams.bs, hparams.seq_len, hparams.head_num * hparams.head_dim))
    attention = sooftware_attention(hparams.dim_size, hparams.head_num, 0.)
    attention.to(device)
    input = [q, k, v, pos_embedding]
    input = [item.to(device) for item in input]
    for i in range(iters + warmup):
        if i == warmup_iter:
            t0 = timeit.default_timer()
        out = attention.forward(*input)
    t1 = timeit.default_timer()
    return t1-t0


def benchmark_labmlai(hparams, iters=100, warmup=10, device='cuda'):
    q = torch.rand((hparams.bs, hparams.seq_len, hparams.head_num, hparams.head_dim))
    k = torch.rand((hparams.bs, hparams.seq_len, hparams.head_num, hparams.head_dim))
    attention = labmlai_attention(hparams.head_num, hparams.dim_size, 0.)
    attention.to(device)
    input = [q, k]
    input = [item.to(device) for item in input]
    for i in range(iters + warmup):
        if i == warmup_iter:
            t0 = timeit.default_timer()
        out = attention.get_scores(*input)
    t1 = timeit.default_timer()
    return t1-t0


def benchmark_huggingface(hparams, iters=100, warmup=10, device='cuda'):
    x = torch.rand((hparams.seq_len, hparams.bs, hparams.dim_size))
    memory_input = torch.rand((hparams.memory_len, hparams.bs, hparams.dim_size))
    pos_embedding = torch.rand(hparams.seq_len+hparams.memory_len, hparams.dim_size)
    attention = huggingface_attention(hparams.head_num, hparams.dim_size, hparams.head_dim, 0.)
    attention.to(device)
    input = [x, pos_embedding]
    input = [item.to(device) for item in input]
    for i in range(iters + warmup):
        if i == warmup_iter:
            t0 = timeit.default_timer()
        out = attention.forward(*input, mems=memory_input.to(device))
    t1 = timeit.default_timer()
    return t1-t0


def benchmark_labmlai_layer(hparams, iters=100, warmup=10, device='cuda'):
    x = torch.rand((hparams.seq_len, hparams.bs, hparams.dim_size))
    memory_input = torch.rand((hparams.memory_len, hparams.bs, hparams.dim_size))
    attention = labmlai_attention(hparams.head_num, hparams.dim_size, 0.)
    attention_layer = labmlai_attention_layer(hparams.dim_size, attention, 0.)
    attention_layer.to(device)
    input = [x, memory_input]
    input = [item.to(device) for item in input]
    for i in range(iters + warmup):
        if i == warmup_iter:
            t0 = timeit.default_timer()
        out = attention_layer.forward(*input)
    t1 = timeit.default_timer()
    return t1-t0


if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    iterations = 10000
    warmup_iter = 10
    hparams = {
        'input_dim': 32,
        'dim_size': 128,
        'seq_len': 64,
        'bs': 32,
        'memory_len': 64,
        'head_num': 2,
        'head_dim': 64,
    }
    hparams = edict(hparams)
    device = ['cuda']

    for d in device:

        t = benchmark_diengine(hparams, iters=iterations)  # 10000: 6.6984s
        print('{} iterations of diengine attention-XL took {}'.format(iterations, t))

        # query and value given as input (full layer not provided)
        t = benchmark_sooftware(hparams, iters=iterations)  # 10000: 6.4019s
        print('{} iterations of sooftware attention-XL took {}'.format(iterations, t))

        t = benchmark_labmlai_layer(hparams, iters=iterations)  # 10000: 6.6384s
        print('{} iterations of labmlai attention-XL layer took {}'.format(iterations, t))

        t = benchmark_huggingface(hparams, iters=iterations)  # 10000: 7.4563s
        print('{} iterations of huggingface attention-XL took {}'.format(iterations, t))
