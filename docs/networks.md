# Neural networks and layers (extracted)

Source: `experiments/Benchmarking Voting Optimizer vs STE on CIFAR-10.md`

This document summarizes the neural network architectures and building-block layers defined and discussed in the source note.

## CIFAR-10: `SmallConvNet` (initial version)

The note defines a small convnet for CIFAR-10 with:

- `conv1`: `Conv2d(3, 32, 3, padding=1, bias=False)`
- `conv2`: `Conv2d(32, 64, 3, padding=1, bias=False)`
- `fc1`: `Linear(64*8*8, 256, bias=False)`
- `fc2`: `Linear(256, 10, bias=False)`
- Activations: `ReLU`
- Pooling: `max_pool2d(..., 2)` after each conv

Binarization behavior in this initial version:

- When `binary=True`, it calls `_binarize()` which **mutates** parameters in-place using `p.data.sign()`.

## CIFAR-10: `SmallConvNet` (STE-style non-destructive binarization)

In the note’s corrected version ("v2"), `SmallConvNet` is rewritten so that binarization is applied in forward without mutating parameters.

### `ste_binarize(w)` helper

The note defines:

- `w_b = w + (w.sign() - w).detach()`

So forward sees `sign(w)` but gradients flow to the underlying float `w`.

### Functional conv/linear forward

Instead of using module calls that read `module.weight` implicitly, the forward path uses `F.conv2d` / `F.linear` so it can explicitly provide either:

- binarized weights (`ste_binarize(self.<layer>.weight)`) when `binary=True`, or
- raw float weights when `binary=False`.

The architecture stays the same shape (two conv + two FC), but avoids in-place `p.data` mutation.

## MNIST / MLP: `BitLinearSTE` + `create_model()`

The note introduces a 1-bit linear layer (STE) and builds a simple MNIST classifier.

### `BitLinearSTE` layer

- Stores float latent weights: `self.weight = nn.Parameter(...)`.
- Forward:
  - `w_q = sign(self.weight)`
  - `weight_proxy = self.weight + (w_q - self.weight).detach()`
  - `linear(x, weight_proxy)`

### Model (`create_model()`)

A sequential model:

- `Flatten()`
- `BitLinearSTE(28*28, hidden)`
- `BatchNorm1d(hidden)` (the note states BatchNorm is critical for convergence in this setup)
- `ReLU()`
- `BitLinearSTE(hidden, 10)`

Several sections use this same pattern with different hidden sizes (e.g. 128, 512).

### Larger model (`create_large_model()`)

Used in the note’s memory measurement:

- `Flatten()`
- `BitLinearSTE(28*28, 1024)`
- `BatchNorm1d(1024)`
- `ReLU()`
- `BitLinearSTE(1024, 1024)`
- `BatchNorm1d(1024)`
- `ReLU()`
- `BitLinearSTE(1024, 10)`

## Swarm-weight layers ("32 agents per weight")

The note introduces a swarm representation where a single logical weight is represented by a population of bits with `swarm_size=32`.

## `BitSwarmLinear`

Defined for the "equal memory" comparison:

- Parameter `population` shaped `[out_features, in_features, swarm_size]`.
- Forward:
  - `swarm_sum = population.sum(dim=2)` (ranges `[-32, 32]`)
  - `w_eff = sign(swarm_sum)` with ties forced to `+1`
  - `weight_proxy = swarm_sum + (w_eff - swarm_sum).detach()`
  - `linear(x, weight_proxy)`

This yields an effective 1-bit weight (`w_eff`) but keeps a sum-based proxy for gradient flow.

## `BitSwarmLogicLinear` (bit-physics / logic framing)

Multiple variants appear throughout the note:

- A version that stores `population` as `int8` (values `{-1, 1}`) to simulate "raw registers" but notes PyTorch restrictions around gradients for non-float tensors.
- Variants that compute `swarm_sum` and then either:
  - use a **majority vote proxy** (STE-style: `w_norm + (w_eff - w_norm).detach()`), or
  - use an **analog/normalized output** `w_normalized = swarm_sum / swarm_size` directly ("Analog Sum, Binary Storage"), producing 33 discrete output levels.

## Normalization / homeostasis modules

The note replaces or complements BatchNorm with simpler thresholding/centering mechanisms in several sections.

## `HomeostaticThreshold`

A module that maintains a learnable `threshold` and a running statistic:

- Forward subtracts a per-feature threshold.
- During training, updates internal statistics and nudges threshold values to control activity.

The note presents variants including:

- activity-based (targeting a firing rate around 50%), and
- mean-matching behavior using a running mean.

## `SimpleCentering`

A minimal centering layer:

- Learnable `bias`.
- Forward: `x - bias`.

Presented as a simpler alternative to BatchNorm / homeostasis.

## `HomeostaticScaleShift`

A scale-and-shift normalization module:

- Parameters: `threshold` and `gain`.
- Uses `gain` initialized using `1/sqrt(fan_in)` (as written in the note).
- Forward: `out = (x - threshold) * gain`.
- During training, updates `threshold` based on batch mean with a configurable learning rate.

## Example "equal information volume" architectures

The note includes an experiment runner that uses the same layer sizes for:

- a float/STE model (using `BitLinearSTE`), and
- a swarm model (using `BitSwarmLinear` with `swarm_size=32`),

motivated by:

- Standard model: 1 FP32 weight (32 bits) per connection.
- Swarm model: 32 binary agents (32 bits) per connection.
