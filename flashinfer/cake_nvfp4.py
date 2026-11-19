"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Cake FMHA kernels for Nemotron 3.5 Super with an NVFP4 KV cache (``sm_100a``).

Three kernels (decode ``"base"``, decode ``"mtp"`` and prefill) serve this
model's attention geometry:

* FP8 E4M3 query and output, NVFP4 K/V with FP8 E4M3 block scales;
* 32 query heads, 2 KV heads, head dimension 128;
* 128-token ``HND`` pages with the shared page-index layout;
* causal dense attention: no sliding window, sinks, LSE output, NVFP4 output,
  skip-softmax or block-sparse attention.

They are routes of the Cake FMHA product (``flashinfer.cake_fmha``):
``select_cake_fmha_decode_route`` and ``select_cake_fmha_context_route`` ask
:func:`select_decode_route` and :func:`select_prefill_route` first, and select
the ``decode_super_nvfp4`` or ``context_super_nvfp4`` route when they accept a
call. They accept every call with the geometry that a kernel serves:

* decode with 1 to 256 requests: ``"mtp"`` for 2, 4, 6 or 8 query tokens per
  request, ``"base"`` for 1, 3 or 5, and ``"base"`` for calls that pass
  ``cum_seq_lens_q`` (or whose requests differ in query length) with at most 6;
* prefill of a single request with at most 32768 query tokens and a
  ``workspace_buffer`` of at least :func:`prefill_workspace_bytes`.

They also check, before any launch, everything the kernels check (pointer
alignment, strides, page table, workspace size, scale range), so a selected
route never declines its call. Every other call, including all calls when the
kernel module cannot be built, keeps Cake's other routes.
"""

import collections
import functools
import logging
import threading
from typing import Dict, Optional, Tuple, Union

import torch

from .utils import get_compute_capability, get_device_sm_count, log2e

logger = logging.getLogger(__name__)

# Oldest toolkit the kernels are built and tested with. A built module runs on
# any driver that serves sm_100a.
MIN_CUDA_TOOLKIT = "13.0"
# From this toolkit on the decode kernels dequantize NVFP4 with one PTX
# instruction per four values. Older assemblers reject it and get a bit-identical
# sequence through FP16 instead (fp4_mul.cuh), which makes decode slower.
FUSED_DEQUANT_CUDA_TOOLKIT = "13.4"

NUM_QO_HEADS = 32
NUM_KV_HEADS = 2
HEAD_DIM = 128
PAGE_SIZE = 128

DECODE_MTP = "mtp"
DECODE_BASE = "base"
_DECODE_KERNEL_IDS = {DECODE_MTP: 0, DECODE_BASE: 1}
# What route_counts() calls a call this module left to Cake's other routes.
OTHER_CAKE_ROUTE = "other"

# Limits of the kernels (csrc/cake_nvfp4/host.h decides; these mirror it so
# that a call no kernel serves is not sent to one).
DECODE_MAX_BATCH_SIZE = 256
DECODE_MTP_Q_LENS = (2, 4, 6, 8)
DECODE_BASE_Q_LENS = (1, 3, 5)
# "base" runs a ragged call one query token at a time, up to this many per request.
DECODE_BASE_MAX_RAGGED_Q_LEN = 6

# The prefill kernel first dequantizes every visible NVFP4 page into the
# workspace, which costs 64 KiB per page plus a fixed 72 MiB. It serves one
# request per call, which keeps that at 584 MiB for a 1M-token KV.
PREFILL_MAX_BATCH_SIZE = 1
PREFILL_MAX_Q_LEN = 32768
# Mirror kPrefillFp8PageBytes and kPrefillSplitReserveBytes of
# csrc/cake_nvfp4/host.h, whose gate is the one that decides. The formula is
# repeated here because a server sizes its workspace before the kernel module
# is loaded.
_PREFILL_FP8_PAGE_BYTES = NUM_KV_HEADS * PAGE_SIZE * HEAD_DIM
_PREFILL_SPLIT_RESERVE_BYTES = 72 << 20

# Calls can arrive from several threads, and ``counter[key] += 1`` is not atomic.
# An uncontended lock costs far less than the kernel launch it accompanies.
_route_lock = threading.Lock()
_route_counts: Dict[Tuple[str, str], int] = collections.Counter()


def _count_route(stage: str, kernel: str) -> None:
    with _route_lock:
        _route_counts[(stage, kernel)] += 1


def route_counts() -> Dict[Tuple[str, str], int]:
    """Return how many geometry-matching calls each ``(stage, kernel)`` pair
    served in this process.

    ``stage`` is ``"decode"`` or ``"prefill"``; ``kernel`` is ``"mtp"``,
    ``"base"``, ``"prefill"`` or ``"other"`` (Cake's other routes). Calls
    outside the geometry are not counted.
    """
    with _route_lock:
        return dict(_route_counts)


def reset_route_counts() -> None:
    """Clear the counters returned by :func:`route_counts`."""
    with _route_lock:
        _route_counts.clear()


def select_decode_kernel(
    batch_size: int, q_len: int, ragged: bool = False
) -> Optional[str]:
    """Pick the decode kernel for a call inside the supported geometry.

    Parameters
    ----------
    batch_size : int
        Requests in the call.
    q_len : int
        Query tokens of the longest request (1 without speculative decoding).
    ragged : bool
        Whether the call passes ``cum_seq_lens_q``, so that its requests can
        differ in query length (also between replays of a CUDA graph).
        ``"mtp"`` does not serve such calls, and ``"base"`` runs them one query
        token at a time.

    Returns
    -------
    Optional[str]
        ``"mtp"``, ``"base"``, or ``None`` when no kernel serves the call.
    """
    if batch_size < 1 or batch_size > DECODE_MAX_BATCH_SIZE:
        return None
    if ragged:
        return DECODE_BASE if 1 <= q_len <= DECODE_BASE_MAX_RAGGED_Q_LEN else None
    if q_len in DECODE_MTP_Q_LENS:
        return DECODE_MTP
    if q_len in DECODE_BASE_Q_LENS:
        return DECODE_BASE
    return None


def select_prefill_kernel(batch_size: int, max_q_len: int) -> bool:
    """Return whether the prefill kernel serves a call inside the supported
    geometry.

    Parameters
    ----------
    batch_size : int
        Requests in the call.
    max_q_len : int
        Query tokens of the longest request.
    """
    return (
        1 <= batch_size <= PREFILL_MAX_BATCH_SIZE
        and 1 <= max_q_len <= PREFILL_MAX_Q_LEN
    )


def prefill_workspace_bytes(max_kv_len: int) -> int:
    """Return the ``workspace_buffer`` size the prefill kernel needs, in bytes.

    A prefill call whose workspace is smaller than this runs on Cake's other
    routes, so size the buffer once for the longest KV the server will see.
    This is a pure function: it does not load the kernel module.

    Parameters
    ----------
    max_kv_len : int
        Longest KV a request can attend to, in tokens: its cached prefix plus
        the query chunk. A model's maximum context length is an upper bound.
        The kernel dequantizes exactly this many tokens, so a call must never
        pass a ``max_kv_len`` below the true length of its request.
    """
    if max_kv_len < 1:
        raise ValueError("max_kv_len must be positive")
    pages = (max_kv_len + PAGE_SIZE - 1) // PAGE_SIZE
    return 2 * pages * _PREFILL_FP8_PAGE_BYTES + _PREFILL_SPLIT_RESERVE_BYTES


def is_supported_device(device: torch.device) -> bool:
    """Return whether the kernels run on ``device`` (compute capability 10.0).

    Parameters
    ----------
    device : torch.device
        A CUDA device.
    """
    return device.type == "cuda" and tuple(get_compute_capability(device)) == (10, 0)


@functools.cache
def _build_blocker() -> Optional[str]:
    """Return why the kernel module cannot be built here, or ``None`` when it
    is already built or its build can be attempted."""
    import os

    from .jit.cpp_ext import get_cuda_version, is_cuda_version_at_least
    from .jit.cake_nvfp4 import gen_cake_nvfp4_fmha_module

    # A module already built (AOT or in the JIT cache) needs no toolkit.
    # FLASHINFER_NVCC names another compiler than the toolkit's, so the toolkit
    # version says nothing about it.
    if gen_cake_nvfp4_fmha_module().is_compiled or "FLASHINFER_NVCC" in os.environ:
        return None
    try:
        if is_cuda_version_at_least(MIN_CUDA_TOOLKIT):
            return None
        found = str(get_cuda_version())
    except RuntimeError:
        found = "none"
    return (
        "its kernels are not built yet and compiling them needs the CUDA "
        f"{MIN_CUDA_TOOLKIT} toolkit or newer (found {found})"
    )


@functools.cache
def _module():
    """Return the kernel module, or ``None`` when it cannot be built here."""
    from .jit.cake_nvfp4 import load_cake_nvfp4_fmha_module

    # Do not start a build that cannot succeed: it would run inside the first
    # matching call and only then fail.
    blocker = _build_blocker()
    if blocker is not None:
        logger.warning(
            "The Cake Nemotron 3.5 Super NVFP4 kernels are unavailable, so "
            "backend='cake' runs those calls on its other routes: %s.",
            blocker,
        )
        return None
    try:
        module = load_cake_nvfp4_fmha_module()
    except RuntimeError as error:  # a failed JIT build
        logger.error(
            "The Cake Nemotron 3.5 Super NVFP4 kernels failed to build, so "
            "backend='cake' runs those calls on its other routes: %s",
            error,
        )
        return None
    if not module.cake_nvfp4_fused_fp4_mul():
        logger.info(
            "The Cake Nemotron 3.5 Super NVFP4 kernels were built with a CUDA "
            "toolkit older than %s: their decode kernels use the slower NVFP4 "
            "dequantization path.",
            FUSED_DEQUANT_CUDA_TOOLKIT,
        )
    return module


# ---------------------------------------------------------------------------
# Route selection. Each check below mirrors a gate of csrc/cake_nvfp4/
# (cake_nvfp4_fmha.cu: readCall; host.h: layoutReason, planDecode*,
# prefillReason), so that a selected route never declines its call.
# ---------------------------------------------------------------------------

_INT32_MAX = (1 << 31) - 1
_DECODE_BASE_MAX_CTAS = 512
_PREFILL_MAX_SMS = 256
_PREFILL_WORKSPACE_ALIGN = 128


def _aligned(tensor: torch.Tensor, alignment: int) -> bool:
    return tensor.data_ptr() != 0 and tensor.data_ptr() % alignment == 0


def _nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _is_token_tensor(tensor: torch.Tensor) -> bool:
    """A packed ``[rows, 32, 128]`` E4M3 tensor."""
    return (
        tensor.dtype == torch.float8_e4m3fn
        and tensor.dim() == 3
        and tuple(tensor.shape[1:]) == (NUM_QO_HEADS, HEAD_DIM)
        and tensor.stride() == (NUM_QO_HEADS * HEAD_DIM, HEAD_DIM, 1)
    )


def _is_page_tensor(tensor: torch.Tensor, dtype: torch.dtype, inner: int) -> bool:
    """An ``HND`` ``[pages, 2, 128, inner]`` tensor with dense tokens and a
    head and page stride the kernels' TMA descriptors accept."""
    return (
        tensor.dtype == dtype
        and tensor.dim() == 4
        and tuple(tensor.shape[1:]) == (NUM_KV_HEADS, PAGE_SIZE, inner)
        and tensor.stride(3) == 1
        and tensor.stride(2) == inner
        and all(stride > 0 and stride % 16 == 0 for stride in tensor.stride()[:2])
        # One KV head must not overlap the next.
        and tensor.stride(1) >= PAGE_SIZE * inner
        and _aligned(tensor, 16)
    )


def _is_index_tensor(tensor: torch.Tensor, elements: int) -> bool:
    return (
        tensor.dtype == torch.int32
        and tensor.is_contiguous()
        and tensor.numel() >= elements
    )


def _scales_supported(bmm1_scale, bmm2_scale, device: torch.device) -> bool:
    for scale in (bmm1_scale, bmm2_scale):
        if isinstance(scale, torch.Tensor):
            if scale.device != device or scale.dtype != torch.float32:
                return False
        elif not isinstance(scale, (int, float)):
            return False
    if isinstance(bmm1_scale, torch.Tensor):
        return True
    # The kernels mask with -FLT_MAX before scaling, so the log2 scale must
    # keep a masked score far below any visible one.
    return 1e-30 <= float(bmm1_scale) * log2e <= 1e30


def _has_geometry(
    *,
    out: torch.Tensor,
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_block_scales: Optional[torch.Tensor],
    v_block_scales: Optional[torch.Tensor],
) -> bool:
    """Whether a call has the model's attention geometry on an ``sm_100``
    device: E4M3 query and output, NVFP4 K/V of 128-token pages with E4M3
    block scales, 32 query heads, 2 KV heads, head dimension 128."""
    return (
        k_block_scales is not None
        and v_block_scales is not None
        and query.dtype == torch.float8_e4m3fn
        and out.dtype == torch.float8_e4m3fn
        and query.dim() == 3
        and tuple(query.shape[1:]) == (NUM_QO_HEADS, HEAD_DIM)
        and k_cache.dtype == torch.uint8
        and v_cache.dtype == torch.uint8
        and k_cache.dim() == 4
        and tuple(k_cache.shape[1:]) == (NUM_KV_HEADS, PAGE_SIZE, HEAD_DIM // 2)
        and is_supported_device(query.device)
    )


def _layout_supported(
    *,
    out: torch.Tensor,
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_block_scales: torch.Tensor,
    v_block_scales: torch.Tensor,
    workspace_buffer: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    cum_seq_lens_q: Optional[torch.Tensor],
    batch_size: int,
    max_q_len: int,
    max_kv_len: int,
    bmm1_scale,
    bmm2_scale,
) -> bool:
    """Every check the kernels share: devices, tensor layouts, page table and
    scales (``readCall`` and ``layoutReason``)."""
    device = query.device
    tensors = [out, k_cache, v_cache, k_block_scales, v_block_scales]
    tensors += [workspace_buffer, block_tables, seq_lens]
    if cum_seq_lens_q is not None:
        tensors.append(cum_seq_lens_q)
    if any(tensor.device != device for tensor in tensors):
        return False
    pages = k_cache.size(0)
    return (
        _is_token_tensor(query)
        and _is_token_tensor(out)
        and out.size(0) == query.size(0)
        and query.size(0) > 0
        and _aligned(query, 16)
        and _aligned(out, 4)
        and _is_page_tensor(k_cache, torch.uint8, HEAD_DIM // 2)
        and _is_page_tensor(v_cache, torch.uint8, HEAD_DIM // 2)
        and _is_page_tensor(k_block_scales, torch.float8_e4m3fn, HEAD_DIM // 16)
        and _is_page_tensor(v_block_scales, torch.float8_e4m3fn, HEAD_DIM // 16)
        and pages > 0
        and v_cache.size(0) == pages
        and k_block_scales.size(0) == pages
        and v_block_scales.size(0) == pages
        and 1 <= batch_size <= _INT32_MAX
        and 1 <= max_q_len <= _INT32_MAX
        and block_tables.dim() == 2
        and block_tables.size(0) == batch_size
        and block_tables.size(1) > 0
        and _is_index_tensor(block_tables, batch_size)
        and _is_index_tensor(seq_lens, batch_size)
        and (cum_seq_lens_q is None or _is_index_tensor(cum_seq_lens_q, batch_size + 1))
        and 0 <= max_kv_len <= block_tables.size(1) * PAGE_SIZE
        and _scales_supported(bmm1_scale, bmm2_scale, device)
    )


def _decode_fused_tokens(kernel: str, q_len: int, ragged: bool) -> int:
    """Query tokens a decode kernel fuses per virtual request, or 0 when it
    does not serve the call (``decodeMtpFusedTokens`` and
    ``decodeBaseFusedTokens`` in host.h)."""
    if kernel == DECODE_MTP:
        return 0 if ragged else {2: 2, 4: 4, 6: 2, 8: 8}.get(q_len, 0)
    if ragged:
        return 1 if 1 <= q_len <= DECODE_BASE_MAX_RAGGED_Q_LEN else 0
    return q_len if q_len in DECODE_BASE_Q_LENS else 0


def decode_workspace_bytes(
    kernel: str, batch_size: int, q_len: int, ragged: bool, sm_count: int
) -> int:
    """Return the ``workspace_buffer`` size a decode kernel needs, in bytes
    (``fillDecodePlan`` in ``csrc/cake_nvfp4/host.h``)."""
    fused = _decode_fused_tokens(kernel, q_len, ragged)
    if fused == 0:
        raise ValueError(f"decode kernel {kernel!r} does not serve q_len {q_len}")
    heads_per_kv = NUM_QO_HEADS // NUM_KV_HEADS
    ctas = sm_count & ~1
    virtual = batch_size * (q_len // fused)
    slots = (virtual + ctas // 2) * 2
    stats = slots * heads_per_kv * fused * HEAD_DIM
    ranges = stats + slots * heads_per_kv * fused * 2
    segments = (ranges + virtual * 4 + 3) // 4 * 4
    return (segments + ctas * (virtual + 1) * 4) * 4


def select_decode_route(
    *,
    out: torch.Tensor,
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_block_scales: Optional[torch.Tensor],
    v_block_scales: Optional[torch.Tensor],
    workspace_buffer: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    cum_seq_lens_q: Optional[torch.Tensor],
    batch_size: int,
    max_q_len: int,
    max_kv_len: int,
    bmm1_scale: Union[float, torch.Tensor],
    bmm2_scale: Union[float, torch.Tensor],
    plain: bool,
) -> Optional[str]:
    """Return the decode kernel (``"mtp"`` or ``"base"``) that serves a call,
    or ``None`` when the call keeps Cake's other routes.

    ``plain`` says the call uses no feature beyond causal dense attention with
    the shared page-index layout. A call with the geometry is counted in
    :func:`route_counts`, also when no kernel serves it.
    """
    if not (
        plain
        and _has_geometry(
            out=out,
            query=query,
            k_cache=k_cache,
            v_cache=v_cache,
            k_block_scales=k_block_scales,
            v_block_scales=v_block_scales,
        )
    ):
        return None
    # A call with cum_seq_lens_q runs as a ragged one even when its rows are
    # uniform now: under a CUDA graph the offsets can change between replays,
    # while the route stays the one chosen at capture.
    ragged = cum_seq_lens_q is not None or query.size(0) != batch_size * max_q_len
    kernel = select_decode_kernel(batch_size, max_q_len, ragged)
    if kernel is not None:
        sm_count = get_device_sm_count(query.device)
        ctas = sm_count & ~1
        if (
            not _layout_supported(
                out=out,
                query=query,
                k_cache=k_cache,
                v_cache=v_cache,
                k_block_scales=k_block_scales,
                v_block_scales=v_block_scales,
                workspace_buffer=workspace_buffer,
                block_tables=block_tables,
                seq_lens=seq_lens,
                cum_seq_lens_q=cum_seq_lens_q,
                batch_size=batch_size,
                max_q_len=max_q_len,
                max_kv_len=max_kv_len,
                bmm1_scale=bmm1_scale,
                bmm2_scale=bmm2_scale,
            )
            or _decode_fused_tokens(kernel, max_q_len, ragged) == 0
            # The "base" kernel reads a ragged call's rows through cum_seq_lens_q.
            or (ragged and cum_seq_lens_q is None)
            or ctas < 2
            or (kernel == DECODE_BASE and ctas > _DECODE_BASE_MAX_CTAS)
            or not _aligned(workspace_buffer, 16)
            or _nbytes(workspace_buffer)
            < decode_workspace_bytes(kernel, batch_size, max_q_len, ragged, sm_count)
            or _module() is None
        ):
            kernel = None
    _count_route("decode", kernel or OTHER_CAKE_ROUTE)
    return kernel


def select_prefill_route(
    *,
    out: torch.Tensor,
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_block_scales: Optional[torch.Tensor],
    v_block_scales: Optional[torch.Tensor],
    workspace_buffer: Optional[torch.Tensor],
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    cum_seq_lens_q: Optional[torch.Tensor],
    batch_size: int,
    max_q_len: int,
    max_kv_len: int,
    bmm1_scale: Union[float, torch.Tensor],
    bmm2_scale: Union[float, torch.Tensor],
    plain: bool,
) -> bool:
    """Return whether the prefill kernel serves a call; ``False`` keeps Cake's
    other routes. Same conventions as :func:`select_decode_route`."""
    if not (
        plain
        and _has_geometry(
            out=out,
            query=query,
            k_cache=k_cache,
            v_cache=v_cache,
            k_block_scales=k_block_scales,
            v_block_scales=v_block_scales,
        )
    ):
        return False
    served = select_prefill_kernel(batch_size, max_q_len)
    if served:
        served = (
            workspace_buffer is not None
            and cum_seq_lens_q is not None
            and 1 <= get_device_sm_count(query.device) <= _PREFILL_MAX_SMS
            and query.size(0) == max_q_len
            and max_kv_len >= max_q_len
            and _layout_supported(
                out=out,
                query=query,
                k_cache=k_cache,
                v_cache=v_cache,
                k_block_scales=k_block_scales,
                v_block_scales=v_block_scales,
                workspace_buffer=workspace_buffer,
                block_tables=block_tables,
                seq_lens=seq_lens,
                cum_seq_lens_q=cum_seq_lens_q,
                batch_size=batch_size,
                max_q_len=max_q_len,
                max_kv_len=max_kv_len,
                bmm1_scale=bmm1_scale,
                bmm2_scale=bmm2_scale,
            )
            # The conversion kernel reads K, V and their scales with one set of strides.
            and k_cache.stride()[:2] == v_cache.stride()[:2]
            and k_block_scales.stride()[:2] == v_block_scales.stride()[:2]
            and _aligned(out, 16)
            and _aligned(workspace_buffer, _PREFILL_WORKSPACE_ALIGN)
            and _nbytes(workspace_buffer) >= prefill_workspace_bytes(max_kv_len)
            and _module() is not None
        )
    _count_route("prefill", "prefill" if served else OTHER_CAKE_ROUTE)
    return served


# ---------------------------------------------------------------------------
# Launch. Cake calls these for a route the functions above selected.
# ---------------------------------------------------------------------------


def run_decode(
    kernel: str,
    *,
    out: torch.Tensor,
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_block_scales: torch.Tensor,
    v_block_scales: torch.Tensor,
    workspace_buffer: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    cum_seq_lens_q: Optional[torch.Tensor],
    batch_size: int,
    max_q_len: int,
    max_kv_len: int,
    bmm1_scale: Union[float, torch.Tensor],
    bmm2_scale: Union[float, torch.Tensor],
    enable_pdl: bool,
) -> None:
    """Launch a decode kernel on the current stream. ``bmm1_scale`` is a float
    or a log2 device tensor, as trtllm-gen takes it."""
    declined = _module().cake_nvfp4_decode(
        _DECODE_KERNEL_IDS[kernel],
        out,
        query,
        k_cache,
        v_cache,
        k_block_scales,
        v_block_scales,
        workspace_buffer,
        block_tables,
        seq_lens,
        batch_size,
        max_q_len,
        max_kv_len,
        bmm1_scale,
        bmm2_scale,
        enable_pdl,
        cum_seq_lens_q,
    )
    if declined:
        raise RuntimeError(
            f"Cake FMHA decode_super_nvfp4 {kernel!r} declined the call its route "
            f"selected: {declined}"
        )


def run_prefill(
    *,
    out: torch.Tensor,
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_block_scales: torch.Tensor,
    v_block_scales: torch.Tensor,
    workspace_buffer: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    cum_seq_lens_q: torch.Tensor,
    batch_size: int,
    max_q_len: int,
    max_kv_len: int,
    bmm1_scale: Union[float, torch.Tensor],
    bmm2_scale: Union[float, torch.Tensor],
) -> None:
    """Launch the prefill kernel on the current stream. A device
    ``bmm1_scale`` is the raw scale Cake's context API receives."""
    if isinstance(bmm1_scale, torch.Tensor):
        # The kernel reads the log2 form, as trtllm-gen does.
        bmm1_scale = bmm1_scale * log2e
    declined = _module().cake_nvfp4_prefill(
        out,
        query,
        k_cache,
        v_cache,
        k_block_scales,
        v_block_scales,
        workspace_buffer,
        block_tables,
        seq_lens,
        cum_seq_lens_q,
        batch_size,
        max_q_len,
        max_kv_len,
        bmm1_scale,
        bmm2_scale,
    )
    if declined:
        raise RuntimeError(
            "Cake FMHA context_super_nvfp4 declined the call its route selected: "
            f"{declined}"
        )
