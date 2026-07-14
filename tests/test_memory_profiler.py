"""Tests for the training/inference memory profiler."""

import torch
import torch.nn as nn

from binary_optimizers.models.mnist import create_mnist_bit_mlp, create_mnist_swarm_mlp
from binary_optimizers.optimizers.signum import MomentumVotingOptimizer
from binary_optimizers.optimizers.swarm import SwarmOptimizer
from binary_optimizers.optimizers.voting import VotingOptimizer
from binary_optimizers.profiling.memory import (
    measure_inference_memory,
    measure_training_memory,
    profile_model_memory,
    tensor_nbytes,
)


def test_tensor_nbytes():
    t = torch.zeros(10, 20, dtype=torch.float32)
    assert tensor_nbytes(t) == 10 * 20 * 4
    t8 = torch.zeros(8, dtype=torch.uint8)
    assert tensor_nbytes(t8) == 8


def test_training_memory_params_only():
    model = create_mnist_bit_mlp(hidden_dim=32)
    report = measure_training_memory(model, optimizer=None)
    expected = sum(p.numel() * p.element_size() for p in model.parameters())
    assert report.param_bytes == expected
    assert report.optimizer_state_bytes == 0
    assert report.total_bytes == expected


def test_training_memory_with_optimizer_state():
    model = create_mnist_bit_mlp(hidden_dim=32)
    opt = MomentumVotingOptimizer(model.parameters(), lr=0.01)
    # Warm-up step to allocate momentum buffers
    x = torch.randn(4, 1, 28, 28)
    out = model(x)
    loss = out.sum()
    loss.backward()
    opt.step()

    report = measure_training_memory(model, opt)
    assert report.optimizer_state_bytes > 0
    assert report.optimizer_n_tensors > 0
    assert report.total_bytes == report.param_bytes + report.optimizer_state_bytes


def test_inference_memory_compression():
    model = create_mnist_bit_mlp(hidden_dim=64)
    inf = measure_inference_memory(model)
    assert inf.float_bytes > 0
    assert inf.n_weights > 0
    assert inf.int8_bytes == inf.n_weights
    # bitpacked should be ~1/8 of int8 (padding may add a little)
    assert inf.bitpacked_bytes <= inf.int8_bytes
    assert inf.bitpacked_bytes >= inf.n_weights // 8
    assert inf.compression_float_to_bitpacked > 1.0
    assert inf.compression_float_to_int8 > 1.0
    # float (4 bytes/weight for linear + bn) vs bitpacked (1 bit)
    assert inf.compression_float_to_bitpacked > inf.compression_float_to_int8


def test_swarm_majority_saves_memory():
    model = create_mnist_swarm_mlp(hidden_dim=16, swarm_size=16)
    maj = measure_inference_memory(model, use_swarm_majority=True)
    full = measure_inference_memory(model, use_swarm_majority=False)
    assert full.bitpacked_bytes > maj.bitpacked_bytes
    assert full.n_weights == maj.n_weights * 16


def test_profile_model_memory_dict():
    model = nn.Linear(10, 5)
    opt = torch.optim.Adam(model.parameters())
    x = torch.randn(2, 10)
    opt.zero_grad()
    model(x).sum().backward()
    opt.step()
    report = profile_model_memory(model, opt)
    d = report.to_dict()
    assert d["training"]["param_bytes"] > 0
    assert d["training"]["optimizer_state_bytes"] > 0
    assert d["inference"]["bitpacked_bytes"] > 0


def test_voting_optimizer_accumulator_counted():
    model = create_mnist_bit_mlp(hidden_dim=16)
    opt = VotingOptimizer(model.parameters(), lr=0.1)
    x = torch.randn(2, 1, 28, 28)
    model(x).sum().backward()
    opt.step()
    mem = measure_training_memory(model, opt)
    assert mem.optimizer_state_bytes > 0
