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

JIT spec of the Cake FMHA kernels for Nemotron 3.5 Super with an NVFP4 KV
cache. ``flashinfer.cake_nvfp4`` routes ``backend="cake"`` calls to them.
"""

import functools

from . import env as jit_env
from .core import JitSpec, gen_jit_spec, sm100a_nvcc_flags

# Directory under the CSRC dir that holds the sources and their headers.
OPERATOR_DIR = "cake_nvfp4"
# The kernels use tcgen05 and are tuned for sm_100a only, so the module is built
# for that target whatever else the compilation context lists.
SOURCES = (
    "cake_nvfp4_fmha.cu",
    "cake_nvfp4_decode_mtp.cu",
    "cake_nvfp4_decode_base.cu",
    "cake_nvfp4_prefill.cu",
)


@functools.cache
def gen_cake_nvfp4_fmha_module() -> JitSpec:
    """Generate the JIT spec of the Nemotron 3.5 Super NVFP4-KV attention kernels."""

    csrc_dir = jit_env.FLASHINFER_CSRC_DIR / OPERATOR_DIR
    return gen_jit_spec(
        "cake_nvfp4_fmha_sm100a",
        [csrc_dir / source for source in SOURCES],
        extra_cuda_cflags=sm100a_nvcc_flags + ["--expt-relaxed-constexpr"],
    )


@functools.cache
def load_cake_nvfp4_fmha_module():
    """Build or load the module."""

    return gen_cake_nvfp4_fmha_module().build_and_load()


__all__ = [
    "gen_cake_nvfp4_fmha_module",
    "load_cake_nvfp4_fmha_module",
]
