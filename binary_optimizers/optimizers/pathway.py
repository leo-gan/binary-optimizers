"""Pathway optimizer: accept ±1 flips by task loss + assembly-index change."""

from __future__ import annotations

from typing import Any, Callable, Literal, Optional

import torch

from binary_optimizers.assembly import (
    approximate_assembly_index,
    mean_assembly_index,
    snap_binary_,
)

Ranking = Literal["ste_topk", "random"]
AssemblyOf = Literal["weights", "logits", "both"]
Mode = Literal["pathway", "hybrid"]
Closure = Callable[[], Any]


def _as_loss_logits(ret: Any) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    if isinstance(ret, tuple):
        if not ret:
            raise TypeError("closure returned an empty tuple")
        loss = ret[0]
        logits = ret[1] if len(ret) > 1 else None
        return loss, logits
    if isinstance(ret, torch.Tensor):
        return ret, None
    raise TypeError(
        "closure must return a loss Tensor or (loss, logits); "
        f"got {type(ret)!r}"
    )


def _unravel(flat_index: int, shape: torch.Size) -> tuple[int, ...]:
    """Unravel a C-order flat index."""
    idx: list[int] = []
    rem = int(flat_index)
    for dim in reversed(shape):
        idx.append(rem % dim)
        rem //= dim
    idx.reverse()
    return tuple(idx)


class PathwayOptimizer(torch.optim.Optimizer):
    """Sampled bit-flip search with an assembly-index accept rule.

    ``step(closure)`` is required. The closure must re-run the forward (and
    backward when ``ranking='ste_topk'`` or ``mode='hybrid'``) and return
    ``(loss, logits)`` or just ``loss``.

    2-D+ parameters are treated as binary weights and snapped to ±1 after
    every step. 1-D parameters (BN / LN affine) are not flipped; they take
    an optional SGD step of size ``bn_lr`` when they have a gradient.

    Modes:

    - ``pathway``: pick ``candidates`` coordinates (STE top-|grad| or random)
      and accept each flip iff ``ΔL + λ ΔA < 0``.
    - ``hybrid``: same accept rule, but the pool is only coordinates where
      the STE gradient wants a flip (``p * grad > 0``).
    """

    def __init__(
        self,
        params,
        *,
        candidates: int = 8,
        lambda_a: float = 0.1,
        ranking: Ranking = "ste_topk",
        assembly_of: AssemblyOf = "both",
        mode: Mode = "pathway",
        bn_lr: float = 1e-2,
    ):
        if candidates < 1:
            raise ValueError(f"candidates must be >= 1, got {candidates}")
        if ranking not in ("ste_topk", "random"):
            raise ValueError(f"unknown ranking: {ranking!r}")
        if assembly_of not in ("weights", "logits", "both"):
            raise ValueError(f"unknown assembly_of: {assembly_of!r}")
        if mode not in ("pathway", "hybrid"):
            raise ValueError(f"unknown mode: {mode!r}")
        defaults = dict(
            candidates=int(candidates),
            lambda_a=float(lambda_a),
            ranking=ranking,
            assembly_of=assembly_of,
            mode=mode,
            bn_lr=float(bn_lr),
        )
        super().__init__(params, defaults)
        self.last_proposed: int = 0
        self.last_accepted: int = 0
        self.last_delta_loss: float = 0.0
        self.last_delta_a: float = 0.0

    @property
    def ranking(self) -> Ranking:
        return self.param_groups[0]["ranking"]

    @property
    def mode(self) -> Mode:
        return self.param_groups[0]["mode"]

    def _binary_params(self) -> list[torch.nn.Parameter]:
        out: list[torch.nn.Parameter] = []
        for group in self.param_groups:
            for p in group["params"]:
                if p is None:
                    continue
                if p.dim() >= 2:
                    out.append(p)
        return out

    def _assembly(
        self,
        binary: list[torch.nn.Parameter],
        logits: Optional[torch.Tensor],
        assembly_of: AssemblyOf,
    ) -> float:
        use_w = assembly_of in ("weights", "both")
        use_y = assembly_of in ("logits", "both")
        a_w = mean_assembly_index(binary) if use_w else 0.0
        a_y = 0.0
        if use_y and logits is not None:
            a_y = approximate_assembly_index(logits)
        if use_w and use_y and logits is not None:
            return 0.5 * a_w + 0.5 * a_y
        if use_y and logits is not None:
            return a_y
        return a_w

    def _candidate_flat_indices(
        self,
        binary: list[torch.nn.Parameter],
        *,
        candidates: int,
        ranking: Ranking,
        mode: Mode,
    ) -> list[tuple[torch.nn.Parameter, int]]:
        if not binary:
            return []
        chunks: list[torch.Tensor] = []
        owners: list[tuple[torch.nn.Parameter, int]] = []
        offset = 0
        for p in binary:
            n = p.numel()
            if ranking == "ste_topk" and p.grad is not None:
                scores = p.grad.detach().abs().reshape(-1)
                if mode == "hybrid":
                    want = (p.data.reshape(-1) * p.grad.detach().reshape(-1)) > 0
                    scores = torch.where(want, scores, torch.zeros_like(scores))
            else:
                scores = torch.rand(n, device=p.device, dtype=torch.float32)
                if mode == "hybrid" and p.grad is not None:
                    want = (p.data.reshape(-1) * p.grad.detach().reshape(-1)) > 0
                    scores = torch.where(want, scores, torch.zeros_like(scores))
                elif mode == "hybrid":
                    scores = torch.zeros(n, device=p.device, dtype=torch.float32)
            chunks.append(scores)
            owners.append((p, offset))
            offset += n
        cat = torch.cat(chunks)
        if cat.numel() == 0:
            return []
        if mode == "hybrid":
            nonzero = (cat > 0).nonzero(as_tuple=False).view(-1)
            if nonzero.numel() == 0:
                return []
            k = min(candidates, int(nonzero.numel()))
            _, loc = torch.topk(cat[nonzero], k)
            picked = nonzero[loc]
        else:
            k = min(candidates, int(cat.numel()))
            _, picked = torch.topk(cat, k)

        picked_cpu = picked.detach().cpu().tolist()
        out: list[tuple[torch.nn.Parameter, int]] = []
        for gi in picked_cpu:
            for p, off in reversed(owners):
                if gi >= off:
                    out.append((p, int(gi - off)))
                    break
        return out

    @staticmethod
    def _flip_at(param: torch.nn.Parameter, flat_index: int) -> None:
        idx = _unravel(flat_index, param.shape)
        param.data[idx] = -param.data[idx]

    def step(self, closure: Optional[Closure] = None):  # type: ignore[override]
        if closure is None:
            raise TypeError(
                "PathwayOptimizer.step requires a closure returning "
                "(loss, logits) or loss"
            )

        group0 = self.param_groups[0]
        candidates: int = group0["candidates"]
        lambda_a: float = group0["lambda_a"]
        ranking: Ranking = group0["ranking"]
        assembly_of: AssemblyOf = group0["assembly_of"]
        mode: Mode = group0["mode"]

        binary = self._binary_params()
        for p in binary:
            snap_binary_(p.data)

        with torch.enable_grad():
            loss0, logits0 = _as_loss_logits(closure())
        loss0_val = float(loss0.detach())
        a0 = self._assembly(binary, logits0, assembly_of)

        picks = self._candidate_flat_indices(
            binary, candidates=candidates, ranking=ranking, mode=mode
        )
        accepted = 0
        loss_cur = loss0_val
        a_cur = a0

        for p, flat_i in picks:
            self._flip_at(p, flat_i)
            with torch.enable_grad():
                loss1, logits1 = _as_loss_logits(closure())
            loss1_val = float(loss1.detach())
            a1 = self._assembly(binary, logits1, assembly_of)
            score = (loss1_val - loss_cur) + lambda_a * (a1 - a_cur)
            if score < 0.0:
                accepted += 1
                loss_cur = loss1_val
                a_cur = a1
            else:
                self._flip_at(p, flat_i)

        for group in self.param_groups:
            step_bn = float(group["bn_lr"])
            for p in group["params"]:
                if p is None or p.grad is None:
                    continue
                if p.dim() < 2:
                    p.data.add_(p.grad.data, alpha=-step_bn)

        for p in binary:
            snap_binary_(p.data)

        self.last_proposed = len(picks)
        self.last_accepted = accepted
        self.last_delta_loss = loss_cur - loss0_val
        self.last_delta_a = a_cur - a0
        return loss0.new_tensor(loss_cur)
