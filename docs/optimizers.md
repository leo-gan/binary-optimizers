# Optimizers (extracted)

Source: `experiments/Benchmarking Voting Optimizer vs STE on CIFAR-10.md`

This document summarizes the optimizer variants defined and discussed in the source note.

## STE Optimizer (SGD + clamp)

The note uses a "Classic STE" approach:

- **Core idea**: keep **latent float weights**, but use `sign()`-quantized weights during forward.
- **Update rule**: standard SGD update on the latent weights.
- **Stabilizer**: clamp weights to `[-1, 1]` after each step.

In the note this appears as `STEOptimizer(torch.optim.SGD)` which performs `super().step(...)` and then applies `p.data.clamp_(-1, 1)`.

## Voting Optimizer (reference, non-fused)

A Python reference implementation in the note:

- **State**: per-parameter accumulator `A`.
- **Vote**: `batch_consensus = -sign(grad)`.
- **Accumulate**: `A += lr * batch_consensus`, then `A = clamp(A, -1, 1)`.
- **Binary weight**: `p.data = sign(A)`.

This version writes binary weights directly from the accumulator sign.

## Voting Optimizer (STE-compatible accumulator + push)

In the note’s corrected CIFAR-10 setup ("v2"), Voting is rewritten to keep float storage compatible with STE-style forward binarization:

- **State**: per-parameter accumulator `A` clamped to `[-1, 1]`.
- **Vote**: `batch_consensus = -sign(grad)`.
- **Target sign**: `w_target = sign(A)`.
- **Push rule**: move the float weight toward `w_target` using `push_rate`:
  - `p.data += push_rate * (w_target - p.data)`
- **Bound**: clamp `p.data` to `[-1, 1]`.

## Voting Optimizer (simple SignSGD)

Later sections introduce a simplified "Voting" optimizer described as "Your Idea / SignSGD":

- **Vote**: `vote = sign(grad)`.
- **Update**: `p.data -= lr * vote`.
- **Note in the source**: no clipping in this initial version (weights can drift).

The note explicitly analyzes two issues with this baseline:

- **Drift problem**: large magnitude shadow weights become hard to flip.
- **No momentum**: per-batch vote noise causes back-and-forth updates.

## Momentum Voting Optimizer (Momentum SignSGD / Signum)

The note’s refinement adds a momentum buffer and clipping:

- **State**: `momentum_buffer` per parameter.
- **Momentum update**: `buf = momentum * buf + (1 - momentum) * grad`.
- **Vote**: `vote = sign(buf)`.
- **Update**: `p.data -= lr * vote`.
- **Bound**: `p.data.clamp_(...)` (the note uses bounds like `[-1.2, 1.2]` or `[-1.5, 1.5]`).
- **Optional**: weight decay variant appears in the memory-usage sections.

## Swarm Optimizer (population bit-flipping)

Introduced as a "biological" branch ("Swarm Descent") when weights are represented as a population of bits.

- **Applies to**: swarm parameters shaped like `[out, in, swarm_size]`.
- **Pressure**: `grad_pressure = grad.mean(dim=2)`.
- **Flip probability**: `probs = clamp(abs(grad_pressure) * recruit_rate, 0, 0.5)`.
- **Target**: `target = -sign(grad_pressure)`.
- **Flip rule**: flip bits that differ from target with probability `probs`.
- **Implementation note**: later fixed to handle non-swarm parameters (e.g. BatchNorm) by applying a standard SGD-style update for those.

## BitLogic Optimizer (stochastic comparator / "logic gate")

Used in the note’s "bit-physics" simulation framing:

- **Applies to**: swarm parameters (3D tensors).
- **Noise**: compare a random tensor against a flip probability.
- **Flip probability**: `abs(grad_pressure) * sensitivity`.
- **Flip rule**: flip bits that are not equal to `target_sign = -sign(grad_pressure)` when `noise < flip_probability`.
- **Non-swarm parameters**: updated with a simple sign-based step for thresholds.

## RankBasedBitOptimizer (deterministic top-k flipping)

The note’s deterministic alternative:

- **Wrongness score**: `wrongness_score = grad * p.data`.
- **Rank and flip**: select top-`k` entries (based on `flip_rate`) and flip those bits.
- **Non-swarm parameters**: updated with a sign-based step.

The note motivates this as ensuring updates happen even when gradient magnitudes are tiny.

## AdaptiveBitOptimizer (layer-normalized stochastic flips)

A stochastic bit-flip optimizer that normalizes gradients per layer:

- **Pressure**: `grad_pressure = grad.mean(dim=2)`.
- **Layer scale**: `mean(abs(grad_pressure)) + eps`.
- **Normalized pressure**: `norm_pressure = grad_pressure / layer_scale`.
- **Flip probability**: `clamp(abs(norm_pressure) * base_flip_rate, 0, 1)`.
- **Target**: `target_sign = -sign(grad_pressure)`.
- **Flip rule**: stochastic flips where bit differs from target and `rand < flip_prob`.

## MomentumRankOptimizer (momentum on wrongness + ranking) + annealing

A later refinement adds momentum to the wrongness score and schedules the flip rate:

- **State**: per-bit `wrongness_buffer`.
- **Current wrongness**: `current_wrongness = grad * p.data`.
- **Buffer update**: exponential moving average with `momentum`.
- **Rank and flip**: flip the top-`k` consistently-wrong bits.
- **Annealing**: the note includes an explicit exponential schedule for `flip_rate` from a high start rate to a lower end rate.

## IntegrateFireOptimizer (accumulator threshold)

A "refuse to flip unless consistently wrong" strategy:

- **State**: per-bit accumulator.
- **Pressure**: `pressure = grad * p.data * 100.0` (as written in the note).
- **Integrate**: `acc = decay * acc + pressure`.
- **Fire**: flip when `acc > threshold`, then reset accumulator.
- **Non-swarm parameters**: updated via clamped SGD.

## AdaptiveIntegrateFireOptimizer (self-tuning threshold)

Adds threshold adaptation based on observed flip rate:

- **Pressure**: computed as `grad * p.data` and then boosted (the note uses `* 1000.0`).
- **Integrate/fire**: as above.
- **Adaptive threshold**: per parameter-group adjustment based on whether flip rate is below/above a target range.

The note also discusses a failure mode where the adaptive threshold can "run away" after a spike.

## BoundedVoteOptimizer (bounded accumulator votes)

A later fix to avoid unbounded pressure from gradient magnitudes:

- **Vote**: `vote = sign(grad) * sign(p.data)` (bounded to `{-1, 0, +1}`).
- **Integrate**: `acc = decay * acc + vote`.
- **Fire**: flip if `acc > vote_threshold`.
- **Refractory reset**: accumulator set to `-threshold * 0.5` for fired bits (as written in the note), to reduce immediate re-flipping.

