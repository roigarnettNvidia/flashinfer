# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Routing of Cake's Nemotron 3.5 Super NVFP4 kernels behind ``backend="cake"``:
a call with their geometry runs on one of them whenever one serves its shape."""

import logging
import threading

import pytest
import torch

import flashinfer.cake_nvfp4 as cake_nvfp4

MTP, BASE = cake_nvfp4.DECODE_MTP, cake_nvfp4.DECODE_BASE
OTHER = cake_nvfp4.OTHER_CAKE_ROUTE
MIB = 1 << 20


@pytest.mark.parametrize(
    "batch_size,q_len,expected",
    [
        # 2, 4, 6 and 8 tokens on "mtp"; 1, 3 and 5 on "base"; at any batch size.
        (1, 1, BASE),
        (256, 1, BASE),
        (32, 3, BASE),
        (3, 5, BASE),
        (1, 2, MTP),
        (256, 4, MTP),
        (1, 6, MTP),
        (3, 6, MTP),
        (256, 6, MTP),
        (16, 8, MTP),
        # Outside what the kernels serve.
        (1, 7, None),
        (256, 7, None),
        (257, 1, None),
        (4, 9, None),
        (4, 0, None),
        (0, 1, None),
    ],
)
def test_select_decode_kernel(batch_size, q_len, expected):
    assert cake_nvfp4.select_decode_kernel(batch_size, q_len) == expected


@pytest.mark.parametrize(
    "q_len,expected",
    [(1, BASE), (2, BASE), (3, BASE), (4, BASE), (5, BASE), (6, BASE), (7, None)]
    + [(8, None), (0, None)],
)
def test_select_decode_kernel_ragged(q_len, expected):
    # "mtp" serves no ragged call; "base" runs them one token at a time, up to 6.
    for batch_size in (1, 2, 32, 256):
        kernel = cake_nvfp4.select_decode_kernel(batch_size, q_len, ragged=True)
        assert kernel == expected
    assert cake_nvfp4.select_decode_kernel(257, q_len, ragged=True) is None


def test_select_prefill_kernel():
    assert cake_nvfp4.select_prefill_kernel(1, 1)
    assert cake_nvfp4.select_prefill_kernel(1, 32768)
    assert not cake_nvfp4.select_prefill_kernel(1, 32769)
    assert not cake_nvfp4.select_prefill_kernel(2, 1024)
    assert not cake_nvfp4.select_prefill_kernel(0, 1024)
    assert not cake_nvfp4.select_prefill_kernel(1, 0)


def test_prefill_workspace_bytes():
    # 64 KiB per page, plus a 72 MiB reserve.
    assert cake_nvfp4.prefill_workspace_bytes(1) == 65536 + 72 * MIB
    assert cake_nvfp4.prefill_workspace_bytes(128) == 65536 + 72 * MIB
    assert cake_nvfp4.prefill_workspace_bytes(129) == 2 * 65536 + 72 * MIB
    assert cake_nvfp4.prefill_workspace_bytes(1 << 20) == 584 * MIB
    assert cake_nvfp4.prefill_workspace_bytes((1 << 20) + 32768) == 600 * MIB
    with pytest.raises(ValueError):
        cake_nvfp4.prefill_workspace_bytes(0)


def test_route_counts_are_thread_safe():
    cake_nvfp4.reset_route_counts()
    calls, threads = 20000, 8

    def count():
        for _ in range(calls):
            cake_nvfp4._count_route("decode", OTHER)

    workers = [threading.Thread(target=count) for _ in range(threads)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    assert cake_nvfp4.route_counts() == {("decode", OTHER): calls * threads}
    cake_nvfp4.reset_route_counts()
    assert cake_nvfp4.route_counts() == {}


SM_COUNT = 148


def _cpu_call(batch_size=2, max_q_len=1, query_dtype=torch.float8_e4m3fn):
    """A call with the geometry on CPU tensors (``_supported`` makes the CPU
    count as an sm_100 device with ``SM_COUNT`` SMs)."""
    rows = torch.zeros((batch_size * max_q_len, 32, 128), dtype=torch.uint8)
    pages = torch.zeros((4, 2, 128, 64), dtype=torch.uint8)
    scales = torch.zeros((4, 2, 128, 8), dtype=torch.uint8).view(torch.float8_e4m3fn)
    return dict(
        out=rows.clone().view(torch.float8_e4m3fn),
        query=rows.view(query_dtype),
        k_cache=pages,
        v_cache=pages.clone(),
        k_block_scales=scales,
        v_block_scales=scales.clone(),
        workspace_buffer=torch.empty(
            cake_nvfp4.prefill_workspace_bytes(10000), dtype=torch.uint8
        ),
        block_tables=torch.zeros((batch_size, 80), dtype=torch.int32),
        seq_lens=torch.full((batch_size,), 10000, dtype=torch.int32),
        cum_seq_lens_q=None,
        batch_size=batch_size,
        max_q_len=max_q_len,
        max_kv_len=10000,
        bmm1_scale=1.0,
        bmm2_scale=1.0,
    )


def _prefill_call(max_q_len=1):
    call = _cpu_call(batch_size=1, max_q_len=max_q_len)
    call["cum_seq_lens_q"] = torch.tensor([0, max_q_len], dtype=torch.int32)
    return call


class _Module:
    """Stands in for the kernel module: records launches, declines nothing."""

    def __init__(self):
        self.launches = []

    def cake_nvfp4_decode(self, kernel_id, *args):
        self.launches.append(("decode", kernel_id, args))
        return ""

    def cake_nvfp4_prefill(self, *args):
        self.launches.append(("prefill", None, args))
        return ""


@pytest.fixture
def module(monkeypatch):
    module = _Module()
    monkeypatch.setattr(cake_nvfp4, "_module", lambda: module)
    return module


@pytest.fixture
def _supported(monkeypatch, module):
    monkeypatch.setattr(cake_nvfp4, "is_supported_device", lambda device: True)
    monkeypatch.setattr(cake_nvfp4, "get_device_sm_count", lambda device: SM_COUNT)
    cake_nvfp4.reset_route_counts()


@pytest.mark.parametrize("reason", ["feature", "dtype", "no block scales", "device"])
def test_calls_outside_the_geometry_are_not_routed(monkeypatch, reason):
    # Such a call is turned away before the kernel module is looked at, and
    # is not counted: it is not one of the routes' calls.
    def unexpected():
        raise AssertionError("the kernel module must not be loaded")

    monkeypatch.setattr(cake_nvfp4, "_module", unexpected)
    if reason != "device":  # a CPU tensor is on no sm_100 device
        monkeypatch.setattr(cake_nvfp4, "is_supported_device", lambda device: True)
    call = _cpu_call()
    if reason == "dtype":
        call["query"] = call["query"].view(torch.int8)
    if reason == "no block scales":
        call["k_block_scales"] = None
    plain = reason != "feature"

    cake_nvfp4.reset_route_counts()
    assert cake_nvfp4.select_decode_route(**call, plain=plain) is None
    prefill = _prefill_call()
    if reason == "dtype":
        prefill["query"] = prefill["query"].view(torch.int8)
    if reason == "no block scales":
        prefill["k_block_scales"] = None
    assert not cake_nvfp4.select_prefill_route(**prefill, plain=plain)
    assert cake_nvfp4.route_counts() == {}


@pytest.mark.parametrize(
    "batch_size,max_q_len,expected",
    [(2, 1, BASE), (2, 3, BASE), (4, 2, MTP), (4, 6, MTP), (3, 8, MTP), (2, 7, None)],
)
def test_select_decode_route(_supported, batch_size, max_q_len, expected):
    call = _cpu_call(batch_size, max_q_len)
    assert cake_nvfp4.select_decode_route(**call, plain=True) == expected
    assert cake_nvfp4.route_counts() == {("decode", expected or OTHER): 1}


@pytest.mark.parametrize("max_kv_len", [1, 128, 8192, 10000])
def test_short_kv_is_routed_to_the_kernels(_supported, max_kv_len):
    # No KV length threshold: a call with the geometry reaches the kernels.
    call = _cpu_call()
    call["max_kv_len"] = max_kv_len
    assert cake_nvfp4.select_decode_route(**call, plain=True) == BASE
    prefill = _prefill_call()
    prefill["max_kv_len"] = max_kv_len
    assert cake_nvfp4.select_prefill_route(**prefill, plain=True)


@pytest.mark.parametrize("batch_size", [2, 4])
def test_prefill_with_several_requests_takes_other_cake_routes(_supported, batch_size):
    call = _cpu_call(batch_size=batch_size)
    call["cum_seq_lens_q"] = torch.arange(batch_size + 1, dtype=torch.int32)
    assert not cake_nvfp4.select_prefill_route(**call, plain=True)
    assert cake_nvfp4.route_counts() == {("prefill", OTHER): 1}


def test_prefill_with_too_many_query_tokens_takes_other_cake_routes(_supported):
    call = _prefill_call(max_q_len=cake_nvfp4.PREFILL_MAX_Q_LEN + 1)
    call["max_kv_len"] = call["max_q_len"]
    call["block_tables"] = torch.zeros((1, 300), dtype=torch.int32)
    assert not cake_nvfp4.select_prefill_route(**call, plain=True)


def _strided(tensor):
    spaced = tensor.new_zeros((2 * tensor.shape[0],) + tensor.shape[1:])
    spaced[::2] = tensor
    return spaced[::2]


def _offset(tensor, offset):
    size = tensor.numel() * tensor.element_size()
    buffer = torch.zeros(size + 256, dtype=torch.uint8)
    return buffer[offset : offset + size].view(tensor.dtype).view(tensor.shape)


_LAYOUTS = {
    "workspace not 16-byte aligned": lambda c: dict(
        workspace_buffer=c["workspace_buffer"][8:]
    ),
    "max_kv_len above the page table": lambda c: dict(max_kv_len=80 * 128 + 1),
    "block_tables not contiguous": lambda c: dict(
        block_tables=_strided(c["block_tables"].T).T
    ),
    "seq_lens not contiguous": lambda c: dict(seq_lens=_strided(c["seq_lens"])),
    "seq_lens not int32": lambda c: dict(seq_lens=c["seq_lens"].long()),
    "query not packed": lambda c: dict(query=_strided(c["query"])),
    "out not packed": lambda c: dict(out=_strided(c["out"])),
    "query not 16-byte aligned": lambda c: dict(query=_offset(c["query"], 8)),
    "k_cache not 16-byte aligned": lambda c: dict(k_cache=_offset(c["k_cache"], 8)),
    "NHD pages": lambda c: dict(k_cache=c["k_cache"].transpose(1, 2)),
    "page counts differ": lambda c: dict(v_cache=c["v_cache"][:3]),
    "bmm1_scale zero": lambda c: dict(bmm1_scale=0.0),
    "bmm1_scale zero in float": lambda c: dict(bmm1_scale=1e-300),
    "bmm1_scale negative": lambda c: dict(bmm1_scale=-1.0),
    "bmm2_scale tensor not float32": lambda c: dict(
        bmm2_scale=torch.ones(1, dtype=torch.float64)
    ),
}


@pytest.mark.parametrize("layout", sorted(_LAYOUTS))
def test_calls_a_kernel_would_decline_take_other_cake_routes(_supported, layout):
    # The routes check, before any launch, what the kernels check at launch.
    call = _cpu_call()
    call.update(_LAYOUTS[layout](call))
    assert cake_nvfp4.select_decode_route(**call, plain=True) is None
    prefill = _prefill_call()
    if layout not in ("block_tables not contiguous", "seq_lens not contiguous"):
        # (With one request, a strided view of either is still contiguous.)
        prefill.update(_LAYOUTS[layout](prefill))
        assert not cake_nvfp4.select_prefill_route(**prefill, plain=True)


@pytest.mark.parametrize("q_len,kernel", [(1, BASE), (2, MTP), (8, MTP)])
def test_decode_workspace_is_checked(_supported, q_len, kernel):
    required = cake_nvfp4.decode_workspace_bytes(kernel, 4, q_len, False, SM_COUNT)
    call = _cpu_call(batch_size=4, max_q_len=q_len)
    call["workspace_buffer"] = torch.empty(required, dtype=torch.uint8)
    assert cake_nvfp4.select_decode_route(**call, plain=True) == kernel
    call["workspace_buffer"] = torch.empty(required - 16, dtype=torch.uint8)
    assert cake_nvfp4.select_decode_route(**call, plain=True) is None


def test_prefill_workspace_is_checked(_supported):
    required = cake_nvfp4.prefill_workspace_bytes(10000)
    call = _prefill_call()
    call["workspace_buffer"] = torch.empty(required, dtype=torch.uint8)
    assert cake_nvfp4.select_prefill_route(**call, plain=True)
    call["workspace_buffer"] = torch.empty(required - 1, dtype=torch.uint8)
    assert not cake_nvfp4.select_prefill_route(**call, plain=True)
    call["workspace_buffer"] = torch.empty(required + 128, dtype=torch.uint8)[64:]
    assert not cake_nvfp4.select_prefill_route(**call, plain=True)


def test_prefill_needs_one_set_of_kv_strides(_supported):
    call = _prefill_call()
    pages = torch.zeros((8, 2, 128, 64), dtype=torch.uint8)
    call["k_cache"] = pages[::2]
    assert not cake_nvfp4.select_prefill_route(**call, plain=True)
    # Decode reads K and V with their own strides.
    decode = _cpu_call()
    decode["k_cache"] = pages[::2]
    assert cake_nvfp4.select_decode_route(**decode, plain=True) == BASE


def test_ragged_decode_without_offsets_takes_other_cake_routes(_supported):
    # Rows that are not batch * max_q_len need cum_seq_lens_q to be read.
    call = _cpu_call(batch_size=2, max_q_len=2)
    call["query"], call["out"] = call["query"][:3], call["out"][:3]
    assert cake_nvfp4.select_decode_route(**call, plain=True) is None
    call["cum_seq_lens_q"] = torch.tensor([0, 2, 3], dtype=torch.int32)
    assert cake_nvfp4.select_decode_route(**call, plain=True) == BASE


def test_base_needs_at_most_512_ctas(monkeypatch, _supported):
    monkeypatch.setattr(cake_nvfp4, "get_device_sm_count", lambda device: 514)
    call = _cpu_call()
    call["workspace_buffer"] = torch.empty(64 << 20, dtype=torch.uint8)
    assert cake_nvfp4.select_decode_route(**call, plain=True) is None
    call = _cpu_call(batch_size=2, max_q_len=2)
    call["workspace_buffer"] = torch.empty(64 << 20, dtype=torch.uint8)
    assert cake_nvfp4.select_decode_route(**call, plain=True) == MTP


def test_unbuildable_module_keeps_other_cake_routes(monkeypatch, _supported):
    monkeypatch.setattr(cake_nvfp4, "_module", lambda: None)
    assert cake_nvfp4.select_decode_route(**_cpu_call(), plain=True) is None
    assert not cake_nvfp4.select_prefill_route(**_prefill_call(), plain=True)
    assert cake_nvfp4.route_counts() == {("decode", OTHER): 1, ("prefill", OTHER): 1}


@pytest.mark.parametrize("offsets", [False, True])
def test_decode_with_query_offsets_is_routed_as_ragged(
    monkeypatch, _supported, offsets
):
    # Uniform query rows prove nothing when the call passes cum_seq_lens_q: a
    # CUDA graph replays whatever offsets it holds then.
    selected = []
    monkeypatch.setattr(
        cake_nvfp4, "select_decode_kernel", lambda *args: selected.append(args)
    )
    call = _cpu_call()
    if offsets:
        call["cum_seq_lens_q"] = torch.tensor([0, 1, 2], dtype=torch.int32)
    assert cake_nvfp4.select_decode_route(**call, plain=True) is None
    assert selected == [(2, 1, offsets)]


def test_run_prefill_converts_a_device_bmm1_scale_to_log2(module):
    # Cake's context API passes a device bmm1_scale raw; the kernel reads log2.
    call = _prefill_call()
    call["bmm1_scale"] = torch.tensor([0.5])
    cake_nvfp4.run_prefill(**call)
    ((_, _, args),) = module.launches
    assert torch.allclose(args[-2], torch.tensor([0.5 * cake_nvfp4.log2e]))


def test_run_decode_raises_when_the_kernel_declines(monkeypatch):
    class Declining:
        def cake_nvfp4_decode(self, *args):
            return "workspace too small"

    monkeypatch.setattr(cake_nvfp4, "_module", Declining)
    with pytest.raises(RuntimeError, match="workspace too small"):
        cake_nvfp4.run_decode(BASE, **_cpu_call(), enable_pdl=False)


class _Spec:
    def __init__(self, is_compiled):
        self.is_compiled = is_compiled


def _patch_toolkit(monkeypatch, is_compiled, new_enough):
    import flashinfer.jit.cpp_ext as cpp_ext
    import flashinfer.jit.cake_nvfp4 as jit_cake_nvfp4

    monkeypatch.setattr(
        jit_cake_nvfp4, "gen_cake_nvfp4_fmha_module", lambda: _Spec(is_compiled)
    )
    monkeypatch.setattr(cpp_ext, "is_cuda_version_at_least", lambda _: new_enough)
    monkeypatch.setattr(cpp_ext, "get_cuda_version", lambda: "12.9")


def test_build_is_blocked_only_by_an_old_toolkit(monkeypatch):
    blocker = cake_nvfp4._build_blocker.__wrapped__
    monkeypatch.delenv("FLASHINFER_NVCC", raising=False)

    _patch_toolkit(monkeypatch, is_compiled=False, new_enough=False)
    assert cake_nvfp4.MIN_CUDA_TOOLKIT == "13.0"
    assert "13.0 toolkit or newer (found 12.9)" in blocker()
    # Another compiler is named explicitly: its build is attempted.
    monkeypatch.setenv("FLASHINFER_NVCC", "/opt/cuda/bin/nvcc")
    assert blocker() is None
    monkeypatch.delenv("FLASHINFER_NVCC")

    _patch_toolkit(monkeypatch, is_compiled=True, new_enough=False)
    assert blocker() is None
    _patch_toolkit(monkeypatch, is_compiled=False, new_enough=True)
    assert blocker() is None


def test_package_import_is_lightweight():
    # Importing the routing module (as a server does to size its workspace)
    # must not import the JIT spec module.
    import subprocess
    import sys

    code = (
        "import sys\n"
        "from flashinfer.cake_nvfp4 import prefill_workspace_bytes\n"
        "assert prefill_workspace_bytes(1) > 0\n"
        "assert 'flashinfer.jit.cake_nvfp4' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_blocked_build_is_not_started(monkeypatch):
    import flashinfer.jit.cake_nvfp4 as jit_cake_nvfp4

    def unexpected():
        raise AssertionError("a build that cannot succeed must not start")

    monkeypatch.setattr(jit_cake_nvfp4, "load_cake_nvfp4_fmha_module", unexpected)
    monkeypatch.setattr(cake_nvfp4, "_build_blocker", lambda: "old toolkit")
    assert cake_nvfp4._module.__wrapped__() is None


def test_failed_build_falls_back(monkeypatch):
    import flashinfer.jit.cake_nvfp4 as jit_cake_nvfp4

    def failing():
        raise RuntimeError("nvcc failed")

    monkeypatch.setattr(jit_cake_nvfp4, "load_cake_nvfp4_fmha_module", failing)
    monkeypatch.setattr(cake_nvfp4, "_build_blocker", lambda: None)
    assert cake_nvfp4._module.__wrapped__() is None


@pytest.mark.parametrize("fused", [True, False])
def test_slower_dequantization_path_is_logged(monkeypatch, caplog, fused):
    import flashinfer.jit.cake_nvfp4 as jit_cake_nvfp4

    class Module:
        @staticmethod
        def cake_nvfp4_fused_fp4_mul():
            return fused

    monkeypatch.setattr(jit_cake_nvfp4, "load_cake_nvfp4_fmha_module", Module)
    monkeypatch.setattr(cake_nvfp4, "_build_blocker", lambda: None)
    with caplog.at_level(logging.INFO, logger=cake_nvfp4.__name__):
        assert cake_nvfp4._module.__wrapped__() is not None
    logged = any("slower NVFP4 dequantization" in m for m in caplog.messages)
    assert logged == (not fused)


def _context_abi_args(call, **changes):
    """``run_args`` as trtllm_batch_context_with_kv_cache passes them to Cake."""
    args = dict(
        out=call["out"],
        out_scale_factor=None,
        query=call["query"],
        key_cache=call["k_cache"],
        value_cache=call["v_cache"],
        workspace_buffer=call["workspace_buffer"],
        multi_ctas_kv_counter_buffer=torch.zeros(16, dtype=torch.uint8),
        block_tables=call["block_tables"],
        seq_lens=call["seq_lens"],
        max_q_len=call["max_q_len"],
        max_kv_len=call["max_kv_len"],
        bmm1_scale=call["bmm1_scale"],
        bmm2_scale=call["bmm2_scale"],
        o_sf_scale=-1.0,
        o_sf_vec_size=-1,
        o_sf_start_index=0,
        batch_size=call["batch_size"],
        window_left=-1,
        cum_seq_lens_q=call["cum_seq_lens_q"],
        cum_seq_lens_kv=torch.tensor([0, 80], dtype=torch.int32),
        sm_count=SM_COUNT,
        enable_pdl=False,
        workspace_size=call["workspace_buffer"].numel(),
        attention_sinks=None,
        key_block_scales=call["k_block_scales"],
        value_block_scales=call["v_block_scales"],
        skip_softmax_threshold_scale_factor=None,
        uses_shared_paged_kv_idx=True,
        use_fp16_softmax=False,
        uses_spcompress=False,
        is_causal=True,
        lse=None,
        lse_stride_tokens=0,
        lse_stride_heads=0,
    )
    args.update(changes)
    return list(args.values())


def _context_module(module):
    from flashinfer.cake_fmha import CakeFmhaContextRoute, CakeFmhaSuperNvfp4Module

    route = CakeFmhaContextRoute(
        target="sm100a",
        component="context_super_nvfp4",
        num_m_blocks=0,
        num_q_heads=32,
        num_kv_heads=2,
        pack_g=0,
        page_size=128,
        l2_swizzle=0,
        is_causal=True,
        return_lse=False,
        enable_sink=False,
        reads_device_scales=True,
    )
    return CakeFmhaSuperNvfp4Module(route=route, module=module)


def test_context_abi_launches_the_prefill_kernel(module):
    # Cake's context dispatch calls every module with the same 34 arguments.
    call = _prefill_call()
    _context_module(module).cake_paged_attention_context(*_context_abi_args(call))
    ((stage, _, args),) = module.launches
    assert stage == "prefill"
    expected = [call[name] for name in ("out", "query", "k_cache", "v_cache")]
    expected += [call[name] for name in ("k_block_scales", "v_block_scales")]
    expected += [call["workspace_buffer"], call["block_tables"], call["seq_lens"]]
    expected.append(call["cum_seq_lens_q"])
    assert all(a is b for a, b in zip(args[:10], expected, strict=True))
    assert args[10:] == (1, 1, 10000, 1.0, 1.0)


@pytest.mark.parametrize(
    "change",
    [
        dict(lse=torch.zeros((1, 32))),
        dict(attention_sinks=torch.zeros(32)),
        dict(is_causal=False),
        dict(window_left=128),
        dict(out_scale_factor=torch.zeros(1)),
        dict(use_fp16_softmax=True),
    ],
)
def test_context_abi_rejects_calls_outside_the_route(module, change):
    call = _prefill_call()
    with pytest.raises(RuntimeError, match="does not match its selected route"):
        _context_module(module).cake_paged_attention_context(
            *_context_abi_args(call, **change)
        )
    assert module.launches == []
