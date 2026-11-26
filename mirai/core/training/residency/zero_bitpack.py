"""Lossless bitmap storage for payloads with many all-zero elements."""

from __future__ import annotations

from math import prod

import torch


def pack_zero_bitmap(tensor):
    integer_dtype = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[tensor.element_size()]
    flat = tensor.contiguous().view(-1)
    # Read the bit pattern to preserve negative zero and NaN payloads.
    selected = flat.view(integer_dtype).ne(0)
    values = flat.masked_select(selected)
    if values.nbytes + (flat.numel() + 7) // 8 >= tensor.nbytes:
        return None
    padding = (-flat.numel()) % 8
    padded = torch.nn.functional.pad(selected, (0, padding)) if padding else selected
    # Each byte is 0 or 1. The multiply gathers one bit from each of the
    # eight little-endian byte lanes into the high byte without a lane reduction.
    lanes = padded.view(torch.int64)
    mask = ((lanes * 0x0102040810204080) >> 56).to(torch.uint8)
    return values, mask


def unpack_zero_bitmap(values, mask, shape, *, dtype):
    flat = torch.zeros(prod(shape), dtype=dtype, device=values.device)
    shifts = torch.arange(8, dtype=torch.uint8, device=mask.device)
    selected = ((mask[:, None] >> shifts) & 1).flatten()[:flat.numel()].bool()
    return flat.masked_scatter_(selected, values.to(dtype)).view(shape)
