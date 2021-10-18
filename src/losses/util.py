import torch


def logsumexp(x: torch.Tensor, keep_mask: torch.Tensor = None, add_one=True, dim=1):
    if keep_mask is not None:
        x = x.masked_fill(~keep_mask, torch.finfo(x.dtype).min)

    if add_one:
        zeros = torch.zeros(x.size(dim - 1), dtype=x.dtype, device=x.device).unsqueeze(
            dim
        )
        x = torch.cat([x, zeros], dim=dim)

    output = torch.logsumexp(x, dim=dim, keepdim=True)

    if keep_mask is not None:
        output = output.masked_fill(~torch.any(keep_mask, dim=dim, keepdim=True), 0)

    return output
