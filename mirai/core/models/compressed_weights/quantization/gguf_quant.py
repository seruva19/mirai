"""GGUF k-quant / IQ sub-4-bit dequant primitives for frozen expert weights.

This is the owner seam for ``gguf_iq4`` / ``gguf_iq3`` / ``gguf_iq2`` quant
formats. It parallels the NF4 path in ``quant.py`` but stores each expert as the
canonical GGUF *packed block* byte layout (one ``uint8`` buffer per weight), so
bytes-per-expert equal the on-wire GGUF footprint exactly:

  * IQ4_XS  -> 136 bytes / 256 weights = 4.25 bits/weight
  * IQ3_XXS ->  98 bytes / 256 weights = 3.0625 bits/weight
  * IQ2_XS  ->  74 bytes / 256 weights = 2.3125 bits/weight

On-the-fly dequant reconstructs bf16 for the existing GEMM (no compute-path
change). Quantize runs offline (CPU) at export; dequant runs on any device.

PROVENANCE
----------
The IQ dequant math + constant tables (``kvalues_iq4nl``, the IQ3_XXS grid_map /
grid_hex, and ``ksigns_iq2xs``) are portable from the **MIT** reference
``gguf`` Python package (``gguf-py/gguf/quants.py``, ggml-org/llama.cpp), which is
the pure-python mirror of the ggml ``dequantize_row_iq4_xs`` /
``dequantize_row_iq3_xxs`` / ``dequantize_row_iq2_xs`` kernels. The code is
ported with attribution; no
GPL/AGPL code was copied. Block layouts additionally match the
public GGUF format spec (``block_iq4_xs`` / ``block_iq3_xxs`` /
``block_iq2_xs`` in ggml-common.h).

  reference: https://github.com/ggml-org/llama.cpp  (gguf-py, MIT)
  files:     gguf-py/gguf/quants.py, ggml/src/ggml-common.h

CALIBRATION CONTRACT
--------------------
The encoders use uniform code-assignment weighting. Per-tensor precision
calibration evaluates their decoded weights against routed input-square
evidence and may choose a different representation for each expert projection.
This is imatrix-weighted format selection, not a claim that IQ3_XXS/IQ2_XS code
assignment itself reproduces llama.cpp's imatrix optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


QK_K = 256  # GGUF super-block size (elements per super-block)

GGUF_FORMATS = ("gguf_iq4", "gguf_iq3", "gguf_iq2")

# type_size = bytes per super-block (block struct size in ggml-common.h).
#   block_iq4_xs  = d(fp16,2) + scales_h(u16,2) + scales_l[QK_K/64=4] + qs[QK_K/2=128] = 136
#   block_iq3_xxs = d(fp16,2) + qs[3*QK_K/8=96 -> 64 grid idx + 32 scale/sign]         = 98
GGUF_TYPE_SIZE = {"gguf_iq4": 136, "gguf_iq3": 98, "gguf_iq2": 74}
GGUF_BLOCK_FORMAT = {
    "gguf_iq4": "iq4_xs",
    "gguf_iq3": "iq3_xxs",
    "gguf_iq2": "iq2_xs",
}

# Canonical published bits-per-weight (format constants, imatrix-independent).
BITS_PER_WEIGHT = {
    "gguf_iq4": 4.25,
    "gguf_iq3": 3.0625,
    "gguf_iq2": 2.3125,
    "nf4": 4.5,  # canonical QLoRA NF4 (4-bit codes + fp32 absmax / 64-block)
    "nf4_double_quant": 4.0 + 8.0 / 64.0 + 32.0 / (64.0 * 256.0),  # ~4.127
}

# --- Constant codebook / grid / sign tables (verbatim from the Apache-2.0 ref) ---

# ggml kvalues_iq4nl (16-entry non-linear 4-bit codebook).
_KVALUES_IQ4NL = (
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
)

# IQ3_XXS grid: 256 rows x 4 magnitudes, packed as 3-bit indices into grid_map.
_IQ3XXS_GRID_MAP = (0x04, 0x0C, 0x14, 0x1C, 0x24, 0x2C, 0x34, 0x3E)
_IQ3XXS_GRID_HEX = (
    "0000020004001100130017002000220031004200730075000101030110011201"
    "2101250130013201410154017001000202020402110220022202310233023702"
    "5102570275020103070310031203250370031304370444045704730475040105"
    "0705320552053506640610071407160743076107011003101010121021102310"
    "3010321034104710501000110211111120112211011203121012121221123012"
    "7212001302132013311346136613011405145014201524154615711505162217"
    "4017002002201120132020202220262031204220012103210521102112212121"
    "3021632167217021002202221122172220222222372240225522012310231423"
    "7023742335245324032527254125742501270327162745270130103012302130"
    "2330503065307230003102312031313144314631013203321032253252327232"
    "1133333330344734723400350635223555351436363663363337603704401740"
    "3540374053405740744120423742404260426642074345430444514464442545"
    "4345704505471047124730471250415070500051065126515551145232527252"
    "0253535310542354275472540255315550562457425724604460466064602161"
    "6161176264623063366344640565526533660367216703700570077010703270"
    "5270267140711272457252720073157333736073217441740075027524753076"
)

# IQ2_XS canonical 512x8 positive grid, encoded with the gguf-py compact
# 2-bit map 0 -> 8, 1 -> 25, 2 -> 43 (MIT, ggml-org/llama.cpp).
_IQ2XS_GRID_HEX = (
    "00000200050008000a0011001400160019002000220025002800410044004600"
    "49005000520055005800610064008000820085008800910094009900a0000101"
    "04010601090110011201150118011a0121012401400142014501480151015401"
    "6001680181018401900100020202050208021102140220024102440250025502"
    "80028a0201040404060409041004120415041804210424044004420445044804"
    "5104540456046004810484049004000502050505080511051405200541054405"
    "500561058005010604061006260640064206840600080208050808080a081108"
    "14082008250841084408500858088008a008aa08010904091009400981098909"
    "000a200a280a960aa00a01100410061009101010121015101810211024104010"
    "4210451048105110541060106a10811084109010001102110511081111111411"
    "2011411144115011801194119611011204120612101240126012001402140514"
    "0814111414142014411444144914501464148014011504151015401500161416"
    "49160118041810181218401854188618001905196619511aa91a002002200520"
    "08200a201120142020204120442050208020a020012104211021402148216521"
    "002222228022a82201240424102429244024002541255225992501261a26a626"
    "002808280a28202855288828a22868299029082a202a822a882a8a2a01400440"
    "0640094010401240154018402140244040404240454048404a40514054406040"
    "6540814084409040004102410541084111411441204141414441504180418541"
    "a241014204421042124229424042004402440544084411441444194420444144"
    "4444504480449444014504451045244540459a4500460a464446504601480448"
    "1048404845485448624800491149444950496949044a00500250055008501150"
    "145020502850415044505050805001510451105115514051425100524452aa52"
    "0154045410542154405460548154a154005508558055885521566856a1560058"
    "14584158505899581a5940594259855a0160046010604060546062608660a960"
    "006124624a62926200641664106540654565a46501686a682569066a546a626a"
    "00800280058008801180148020802a8041804480508080808280a880aa800181"
    "0481068110814081518159810082208280828282a082a8820184048410841284"
    "158440846084898400854485a58518866a860088088825885a8880888288a888"
    "0689228a808a888a968aa88a0190049010904090569084900091229164915692"
    "89920094059444945094589429959095929541965198a6984999159a609a00a0"
    "02a008a00aa020a02aa0a0a051a159a1a6a100a202a208a22aa280a2a0a240a4"
    "95a465a698a60aa820a822a828a8a0a8a8a804a984a986a928aa2aaa91aaaaaa"
)

# ggml ksigns_iq2xs (128-entry): low 7 bits = index, bit 7 = parity so the decoded
# 8-sign pattern always has an even number of negatives. Generated deterministically
# (verified against the Apache-2.0 reference table: [0,129,130,3,132,5,6,135,...]).
def _build_ksigns() -> tuple[int, ...]:
    return tuple(i if bin(i).count("1") % 2 == 0 else i | 0x80 for i in range(128))


_KSIGNS_IQ2XS = _build_ksigns()

# Lazily-built device/dtype-cached torch tables.
_TABLE_CACHE: "dict[tuple[str, str], torch.Tensor]" = {}


def _decode_iq3xxs_grid() -> list[list[int]]:
    """Decode grid_hex (ASCII hex) -> 256 rows of 4 magnitudes via grid_map."""
    raw = bytes.fromhex(_IQ3XXS_GRID_HEX)  # 512 packed bytes
    flat: list[int] = []
    for byte in raw:  # 2 elements/byte (3-bit index at shift 0 and 4)
        flat.append(_IQ3XXS_GRID_MAP[byte & 0x7])
        flat.append(_IQ3XXS_GRID_MAP[(byte >> 4) & 0x7])
    return [flat[i * 4 : i * 4 + 4] for i in range(256)]


def _decode_iq2xs_grid() -> list[list[int]]:
    raw = bytes.fromhex(_IQ2XS_GRID_HEX)
    values = (8, 25, 43)
    flat: list[int] = []
    for byte in raw:
        for shift in (0, 2, 4, 6):
            index = (byte >> shift) & 0x3
            if index >= len(values):
                raise RuntimeError("IQ2_XS grid contains an invalid code.")
            flat.append(values[index])
    if len(flat) != 512 * 8:
        raise RuntimeError("IQ2_XS grid has an invalid size.")
    return [flat[i * 8 : i * 8 + 8] for i in range(512)]


def _table(name: str, device: "torch.device") -> "torch.Tensor":
    key = (name, str(device))
    cached = _TABLE_CACHE.get(key)
    if cached is not None:
        return cached
    if name == "kvalues":
        t = torch.tensor(_KVALUES_IQ4NL, dtype=torch.float32, device=device)
    elif name == "grid":
        t = torch.tensor(_decode_iq3xxs_grid(), dtype=torch.float32, device=device)
    elif name == "grid_iq2xs":
        t = torch.tensor(_decode_iq2xs_grid(), dtype=torch.float32, device=device)
    elif name == "ksigns":
        t = torch.tensor(_KSIGNS_IQ2XS, dtype=torch.int64, device=device)
    else:  # pragma: no cover - defensive
        raise KeyError(name)
    _TABLE_CACHE[key] = t
    return t


@dataclass(frozen=True)
class _GgufMeta:
    """Per-tensor-invariant GGUF block metadata (parallel to _Nf4Meta)."""

    block_format: str  # "iq4_xs" | "iq3_xxs" | "iq2_xs"
    blocksize: int  # QK_K (256)
    type_size: int  # bytes per super-block
    weight_dtype: str = "bfloat16"  # dtype dequant targets by default


def normalize_gguf_format(value: str | None) -> str:
    text = str(value or "").strip().lower()
    aliases = {
        "gguf_iq4_xs": "gguf_iq4",
        "iq4_xs": "gguf_iq4",
        "iq4": "gguf_iq4",
        "gguf_iq3_xxs": "gguf_iq3",
        "iq3_xxs": "gguf_iq3",
        "iq3": "gguf_iq3",
        "gguf_iq2_xs": "gguf_iq2",
        "iq2_xs": "gguf_iq2",
        "iq2": "gguf_iq2",
    }
    normalized = aliases.get(text, text)
    if normalized not in GGUF_FORMATS:
        raise ValueError(
            "gguf quant format must be one of: " + ", ".join(GGUF_FORMATS) + "."
        )
    return normalized


def gguf_meta_for(fmt: str) -> _GgufMeta:
    fmt = normalize_gguf_format(fmt)
    return _GgufMeta(
        block_format=GGUF_BLOCK_FORMAT[fmt],
        blocksize=QK_K,
        type_size=GGUF_TYPE_SIZE[fmt],
    )


def gguf_num_superblocks(numel: int) -> int:
    if int(numel) % QK_K != 0:
        raise ValueError(
            f"gguf quantization requires a multiple of {QK_K} elements, got {numel}."
        )
    return int(numel) // QK_K


def gguf_stored_bytes(fmt: str, numel: int) -> int:
    """Exact stored bytes for ``numel`` weights in ``fmt`` (== on-wire footprint)."""
    fmt = normalize_gguf_format(fmt)
    return gguf_num_superblocks(numel) * GGUF_TYPE_SIZE[fmt]


def nf4_stored_bytes(numel: int, *, blocksize: int = 64, double_quant: bool = True) -> int:
    """Analytic NF4 stored bytes for the same weight count (CPU-computable baseline).

    Mirrors the buffers bitsandbytes' quantize_4bit(compress_statistics=...) keeps,
    amortizing the shared 16-/256-entry codebooks (stored once per tensor, not per
    block). Lets the CPU test compare against NF4 without needing CUDA/bitsandbytes.
    """
    n = int(numel)
    if n % int(blocksize) != 0:
        raise ValueError(f"nf4 baseline needs a multiple of {blocksize} elements.")
    blocks = n // int(blocksize)
    packed = n // 2  # 4-bit codes
    if double_quant:
        absmax = blocks  # uint8 quantized absmax
        nested = 4 * ((blocks + 255) // 256)  # fp32 nested absmax / 256-block
        return packed + absmax + nested + 4  # + fp32 offset
    return packed + 4 * blocks  # fp32 absmax per block


# --------------------------------------------------------------------------- #
# IQ4_XS
# --------------------------------------------------------------------------- #

def quantize_iq4_xs(weight_2d: "torch.Tensor") -> "torch.Tensor":
    """Quantize a 2D fp/bf16 tensor to IQ4_XS packed blocks -> uint8 [N, 136].

    Nearest-codebook encoder: per super-block scale ``d`` (fp16) with per-32
    sub-block integer scale ``ls`` (6-bit), elements are the nearest of the 16
    non-linear codebook points. Faithful GGUF layout; not imatrix-optimized.
    """
    if weight_2d.ndim != 2:
        raise ValueError(f"quantize_iq4_xs expects a 2D tensor, got {tuple(weight_2d.shape)}.")
    device = weight_2d.device
    x = weight_2d.reshape(-1).to(torch.float32)
    n = gguf_num_superblocks(x.numel())
    xb = x.reshape(n, 8, 32)
    kv = _table("kvalues", device)

    amax = xb.abs().amax(dim=2)  # [n,8]
    sb_scale = amax / 127.0
    d = (sb_scale.amax(dim=1) / 31.0).clamp(min=1e-12)  # [n]
    ls = torch.round(sb_scale / d[:, None]).clamp(0, 31)  # (ls_stored-32) in [0,31]
    dl = d[:, None] * ls  # [n,8]
    dl_safe = torch.where(dl == 0, torch.ones_like(dl), dl)
    q = xb / dl_safe[:, :, None]
    idx = (q[..., None] - kv).abs().argmin(dim=-1)  # [n,8,32]
    zero_idx = int((0.0 - kv).abs().argmin().item())
    idx = torch.where((dl == 0)[:, :, None].expand_as(idx), torch.full_like(idx, zero_idx), idx)

    ls_store = (ls.to(torch.int64) + 32)  # [n,8] in [32,63]
    low = ls_store & 0xF
    high = (ls_store >> 4) & 0x3
    ib = torch.arange(8, device=device, dtype=torch.int64)
    scales_h = ((high << (2 * ib)).sum(dim=1)).to(torch.int64)  # [n]
    scales_l = (low[:, 0::2] | (low[:, 1::2] << 4)).to(torch.int64)  # [n,4]

    idx = idx.to(torch.int64).reshape(n, 8, 32)
    qs = (idx[:, :, 0:16] | (idx[:, :, 16:32] << 4)).reshape(n, 128)  # [n,128]

    blocks = torch.empty((n, 136), dtype=torch.uint8, device=device)
    blocks[:, 0:2] = d.to(torch.float16).reshape(n, 1).contiguous().view(torch.uint8)
    blocks[:, 2:4] = scales_h.to(torch.int16).reshape(n, 1).contiguous().view(torch.uint8)
    blocks[:, 4:8] = scales_l.to(torch.uint8)
    blocks[:, 8:136] = qs.to(torch.uint8)
    return blocks


def dequantize_iq4_xs(
    blocks: "torch.Tensor",
    *,
    shape: tuple[int, ...],
    dtype: "torch.dtype",
    device: "torch.device",
) -> "torch.Tensor":
    """Dequantize IQ4_XS packed blocks -> tensor of ``shape``/``dtype``."""
    validate_gguf_blocks("gguf_iq4", blocks)
    b = blocks.to(device=device)
    n = b.shape[0]
    d = b[:, 0:2].contiguous().view(torch.float16).to(torch.float32).reshape(n)
    scales_h = b[:, 2:4].contiguous().view(torch.int16).to(torch.int64).bitwise_and(0xFFFF).reshape(n)
    scales_l = b[:, 4:8].to(torch.int64)  # [n,4]
    qs = b[:, 8:136].to(torch.int64)  # [n,128]

    ib = torch.arange(8, device=device, dtype=torch.int64)
    low = (scales_l[:, ib // 2] >> (4 * (ib % 2))) & 0xF  # [n,8]
    high = (scales_h[:, None] >> (2 * ib)[None, :]) & 0x3  # [n,8]
    ls = (low | (high << 4)) - 32
    dl = d[:, None] * ls.to(torch.float32)  # [n,8]

    kv = _table("kvalues", device)
    qs3 = qs.reshape(n, 8, 16)
    low_n = qs3 & 0xF
    high_n = (qs3 >> 4) & 0xF
    y = torch.empty((n, 8, 32), dtype=torch.float32, device=device)
    y[:, :, 0:16] = dl[:, :, None] * kv[low_n]
    y[:, :, 16:32] = dl[:, :, None] * kv[high_n]
    return y.reshape(tuple(int(s) for s in shape)).to(dtype)


# --------------------------------------------------------------------------- #
# IQ3_XXS
# --------------------------------------------------------------------------- #

def quantize_iq3_xxs(weight_2d: "torch.Tensor") -> "torch.Tensor":
    """Quantize a 2D tensor to IQ3_XXS packed blocks -> uint8 [N, 98].

    Uniform-weight reference encoder. Per super-block scale ``d`` plus
    per-32-block 4-bit scale, groups of
    4 magnitudes matched to the 256-row grid, signs stored as 7-bit + parity.
    """
    if weight_2d.ndim != 2:
        raise ValueError(f"quantize_iq3_xxs expects a 2D tensor, got {tuple(weight_2d.shape)}.")
    device = weight_2d.device
    x = weight_2d.reshape(-1).to(torch.float32)
    n = gguf_num_superblocks(x.numel())
    xb = x.reshape(n, 8, 32)
    grid = _table("grid", device)  # [256,4]

    block_amax = xb.abs().amax(dim=2)  # [n,8]
    d = (block_amax.amax(dim=1) / (62.0 * 7.75)).clamp(min=1e-12)  # [n]
    sc = torch.round((block_amax / 62.0) / (0.5 * d[:, None]) - 0.5).clamp(0, 15)  # [n,8]
    db = d[:, None] * 0.5 * (0.5 + sc)  # [n,8]
    db_safe = torch.where(db == 0, torch.ones_like(db), db)

    # Grid match: 8 groups of 4 per 32-block -> nearest of 256 grid rows.
    tg = (xb.abs() / db_safe[:, :, None]).reshape(n, 8, 8, 4)
    dist = (tg[..., None, :] - grid[None, None, None, :, :]).pow(2).sum(-1)  # [n,8,8,256]
    gi = dist.argmin(-1)  # [n,8,8] grid indices

    # Signs: parity-constrained (decoded pattern always has even negatives).
    xg = xb.reshape(n, 8, 4, 8)
    neg = xg < 0
    cnt = neg.to(torch.int64).sum(-1)  # [n,8,4]
    odd = (cnt % 2 == 1)
    minpos = xg.abs().argmin(-1)  # [n,8,4]
    flip = torch.zeros_like(neg)
    flip.scatter_(-1, minpos.unsqueeze(-1), odd.unsqueeze(-1))
    bits = (neg ^ flip).to(torch.int64)  # [n,8,4,8], even popcount per group
    jb = torch.arange(7, device=device, dtype=torch.int64)
    idx7 = (bits[..., 0:7] << jb).sum(-1)  # [n,8,4]

    scales_u32 = (
        (sc.to(torch.int64) << 28)
        | idx7[:, :, 0]
        | (idx7[:, :, 1] << 7)
        | (idx7[:, :, 2] << 14)
        | (idx7[:, :, 3] << 21)
    )  # [n,8]

    blocks = torch.empty((n, 98), dtype=torch.uint8, device=device)
    blocks[:, 0:2] = d.to(torch.float16).reshape(n, 1).contiguous().view(torch.uint8)
    blocks[:, 2:66] = gi.reshape(n, 64).to(torch.uint8)
    sb = torch.stack(
        [scales_u32 & 0xFF, (scales_u32 >> 8) & 0xFF, (scales_u32 >> 16) & 0xFF, (scales_u32 >> 24) & 0xFF],
        dim=-1,
    )  # [n,8,4] little-endian
    blocks[:, 66:98] = sb.reshape(n, 32).to(torch.uint8)
    return blocks


def dequantize_iq3_xxs(
    blocks: "torch.Tensor",
    *,
    shape: tuple[int, ...],
    dtype: "torch.dtype",
    device: "torch.device",
) -> "torch.Tensor":
    """Dequantize IQ3_XXS packed blocks -> tensor of ``shape``/``dtype``."""
    validate_gguf_blocks("gguf_iq3", blocks)
    b = blocks.to(device=device)
    n = b.shape[0]
    d = b[:, 0:2].contiguous().view(torch.float16).to(torch.float32).reshape(n)
    qs = b[:, 2:66].to(torch.int64)  # [n,64] grid indices
    # clone(contiguous_format) forces a fresh offset-0 buffer: a plain
    # .contiguous() is a no-op on a single-block (n==1) slice at byte offset 66
    # (the size-1 leading dim reads as already-contiguous), which then makes
    # .view(torch.int32) raise "storage_offset must be divisible by 4".
    scales = (
        b[:, 66:98]
        .clone(memory_format=torch.contiguous_format)
        .view(torch.int32)
        .to(torch.int64)
        .bitwise_and(0xFFFFFFFF)
        .reshape(n, 8)
    )

    grid = _table("grid", device)  # [256,4]
    ksigns = _table("ksigns", device)  # [128]
    jb = torch.arange(8, device=device, dtype=torch.int64)
    y = torch.empty((n, 8, 4, 8), dtype=torch.float32, device=device)
    for ib32 in range(8):
        sc = scales[:, ib32]
        db = d * (0.5 + (sc >> 28).to(torch.float32)) * 0.5  # [n]
        base = 8 * ib32
        for l in range(4):
            sidx = (sc >> (7 * l)) & 0x7F
            signbyte = ksigns[sidx]  # [n]
            g1 = grid[qs[:, base + 2 * l]]  # [n,4]
            g2 = grid[qs[:, base + 2 * l + 1]]  # [n,4]
            g = torch.cat([g1, g2], dim=1)  # [n,8]
            signs = (signbyte[:, None] >> jb[None, :]) & 1
            signmul = torch.where(signs == 0, torch.ones_like(g), -torch.ones_like(g))
            y[:, ib32, l, :] = db[:, None] * g * signmul
    return y.reshape(n, 256).reshape(tuple(int(s) for s in shape)).to(dtype)


# --------------------------------------------------------------------------- #
# IQ2_XS
# --------------------------------------------------------------------------- #

def quantize_iq2_xs(weight_2d: "torch.Tensor") -> "torch.Tensor":
    """Encode canonical IQ2_XS blocks by exact legal-grid assignment.

    Code assignment uses uniform element weights. Mirai's imatrix calibration
    evaluates the decoded candidate before selecting it for any projection.
    """
    if weight_2d.ndim != 2:
        raise ValueError(
            f"quantize_iq2_xs expects a 2D tensor, got {tuple(weight_2d.shape)}."
        )
    device = weight_2d.device
    x = weight_2d.reshape(-1).to(torch.float32)
    n = gguf_num_superblocks(x.numel())
    groups = x.reshape(n, 16, 2, 8)
    magnitudes = groups.abs()
    negative = groups < 0
    odd = negative.sum(dim=-1).remainder(2).bool()
    minimum = magnitudes.argmin(dim=-1)
    parity_flip = torch.zeros_like(negative)
    parity_flip.scatter_(-1, minimum.unsqueeze(-1), odd.unsqueeze(-1))
    signs = negative ^ parity_flip

    desired = magnitudes.amax(dim=(-1, -2)) / 43.0
    max_desired = desired.amax(dim=1)
    d = (max_desired * (4.0 / 15.5)).clamp_min(0.0)
    d_safe = torch.where(d > 0, d, torch.ones_like(d))
    scales = torch.round(desired * 4.0 / d_safe[:, None] - 0.5).clamp(0, 15)
    scales = torch.where(desired > 0, scales, torch.zeros_like(scales))
    db = d[:, None] * (0.5 + scales) * 0.25
    db_safe = torch.where(db > 0, db, torch.ones_like(db))
    targets = magnitudes / db_safe[:, :, None, None]
    grid = _table("grid_iq2xs", device)
    grid_indices = torch.empty((n, 16, 2), dtype=torch.int64, device=device)
    for half in range(2):
        distance = (
            targets[:, :, half, None, :] - grid[None, None, :, :]
        ).square().sum(dim=-1)
        grid_indices[:, :, half] = distance.argmin(dim=-1)

    bit_positions = torch.arange(7, device=device, dtype=torch.int64)
    sign_indices = (
        signs[..., :7].to(torch.int64) << bit_positions
    ).sum(dim=-1)
    packed_qs = grid_indices | (sign_indices << 9)
    blocks = torch.zeros((n, 74), dtype=torch.uint8, device=device)
    blocks[:, :2] = d.to(torch.float16).reshape(n, 1).contiguous().view(torch.uint8)
    little_endian = torch.stack(
        (packed_qs & 0xFF, (packed_qs >> 8) & 0xFF), dim=-1
    )
    blocks[:, 2:66] = little_endian.reshape(n, 64).to(torch.uint8)
    packed_scales = scales[:, 0::2].to(torch.int64) | (
        scales[:, 1::2].to(torch.int64) << 4
    )
    blocks[:, 66:74] = packed_scales.to(torch.uint8)
    return blocks


def dequantize_iq2_xs(
    blocks: "torch.Tensor",
    *,
    shape: tuple[int, ...],
    dtype: "torch.dtype",
    device: "torch.device",
) -> "torch.Tensor":
    """Dequantize canonical IQ2_XS packed blocks."""
    validate_gguf_blocks("gguf_iq2", blocks)
    b = blocks.to(device=device)
    n = int(b.shape[0])
    d = b[:, :2].contiguous().view(torch.float16).float().reshape(n)
    qs_bytes = b[:, 2:66].to(torch.int64).reshape(n, 32, 2)
    qs = qs_bytes[..., 0] | (qs_bytes[..., 1] << 8)
    packed_scales = b[:, 66:74].to(torch.int64)
    scales = torch.stack(
        (packed_scales & 0xF, (packed_scales >> 4) & 0xF), dim=-1
    ).reshape(n, 16)
    db = d[:, None] * (0.5 + scales.float()) * 0.25
    grid = _table("grid_iq2xs", device)[qs & 0x1FF].reshape(n, 16, 2, 8)
    sign_bytes = _table("ksigns", device)[qs >> 9].reshape(n, 16, 2)
    bits = torch.arange(8, device=device, dtype=torch.int64)
    sign_mask = (sign_bytes[..., None] >> bits) & 1
    sign = torch.where(sign_mask == 0, 1.0, -1.0)
    decoded = db[:, :, None, None] * grid * sign
    return decoded.reshape(tuple(int(value) for value in shape)).to(dtype)


# --------------------------------------------------------------------------- #
# Dispatch + validation
# --------------------------------------------------------------------------- #

def validate_gguf_blocks(fmt: str, blocks: "torch.Tensor") -> None:
    """Fail fast on malformed packed blocks (wrong dtype / type_size / rank)."""
    fmt = normalize_gguf_format(fmt)
    expected = GGUF_TYPE_SIZE[fmt]
    if blocks.dtype != torch.uint8:
        raise ValueError(f"{fmt} blocks must be uint8, got {blocks.dtype}.")
    if blocks.ndim != 2 or int(blocks.shape[-1]) != expected:
        raise ValueError(
            f"{fmt} blocks must be [N, {expected}], got {tuple(blocks.shape)}."
        )


def quantize_gguf(fmt: str, weight_2d: "torch.Tensor") -> "torch.Tensor":
    fmt = normalize_gguf_format(fmt)
    if fmt == "gguf_iq4":
        return quantize_iq4_xs(weight_2d)
    if fmt == "gguf_iq3":
        return quantize_iq3_xxs(weight_2d)
    return quantize_iq2_xs(weight_2d)


def dequantize_gguf(
    fmt: str,
    blocks: "torch.Tensor",
    *,
    shape: tuple[int, ...],
    dtype: "torch.dtype",
    device: "torch.device",
) -> "torch.Tensor":
    fmt = normalize_gguf_format(fmt)
    if fmt == "gguf_iq4":
        return dequantize_iq4_xs(blocks, shape=shape, dtype=dtype, device=device)
    if fmt == "gguf_iq3":
        return dequantize_iq3_xxs(blocks, shape=shape, dtype=dtype, device=device)
    return dequantize_iq2_xs(blocks, shape=shape, dtype=dtype, device=device)
