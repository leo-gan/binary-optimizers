import torch


def ste_binarize(w: torch.Tensor) -> torch.Tensor:
    w_sign = w.sign()
    return w + (w_sign - w).detach()


def ste_binarize_weight(w: torch.Tensor, scale: bool = True, dim=None) -> torch.Tensor:
    if not scale:
        return ste_binarize(w)

    if dim is None:
        if w.dim() == 4:
            dim = (1, 2, 3)
        elif w.dim() == 2:
            dim = (1,)
        else:
            dim = None

    if dim is None:
        return ste_binarize(w)

    alpha = w.abs().mean(dim=dim, keepdim=True)
    w_q = w.sign() * alpha
    return w + (w_q - w).detach()
