# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runtime glue between the LeapQuant state ops and the GDN attention path.

``state.py`` holds the device-agnostic numerics; this module owns the
integration concerns: deriving window-boundary rows from per-request sequence
lengths and applying the flush to rows of a state cache pool.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm_ascend.quantization.methods.mamba_state.leapquant.state import leap_window_flush

if TYPE_CHECKING:
    from vllm_ascend.ascend_config import LeapQuantConfig

__all__ = [
    "leapquant_flush_state_rows",
]

_device_gate_checked = False


def _check_device_once() -> None:
    """Fail fast on non-A3 devices at the first flush (worker process).

    The worker has already initialized its NPU context at the first flush.
    """
    global _device_gate_checked
    if _device_gate_checked:
        return
    from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

    try:
        device_type = get_ascend_device_type()
    except Exception:
        # No resolvable Ascend device (e.g. CPU-only CI running unit tests);
        # leave the gate unset and retry the lookup on the next flush.
        return
    if device_type != AscendDeviceType.A3:
        raise RuntimeError(
            "leapquant_config is validated on Atlas A3 only in this milestone; "
            "please disable leapquant_config on this device."
        )
    _device_gate_checked = True


def leapquant_flush_state_rows(
    state: torch.Tensor,
    state_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    config: LeapQuantConfig,
) -> None:
    """Re-quantize the state rows whose sequence just crossed a window boundary.

    The scheduler orders requests decodes-first. The post-step sequence
    length must be positive and divisible by the window size. The flush
    keeps the state pool in its original floating-point layout.

    The prefill kernel only materializes the state at the end of each
    chunk. Prefill flushes therefore occur at boundary-aligned chunk ends.
    Decode processes one token per request and flushes at each boundary.

    Args:
        state: state pool ``[slots, H, V, K]`` (may have a padded batch stride).
        state_indices: per-request state slot ids aligned with ``seq_lens``.
        seq_lens: CPU int tensor of per-request post-step sequence lengths.
        config: enabled LeapQuantConfig.
    """
    if state_indices.numel() == 0:
        return
    if state.device.type != "cpu":
        _check_device_once()
    due = (seq_lens > 0) & (seq_lens % config.window == 0)
    due_rows = torch.nonzero(due, as_tuple=False).flatten()
    if due_rows.numel() == 0:
        return

    rows = due_rows.to(state_indices.device)
    slot_ids = state_indices.index_select(0, rows)
    # Use the same indexed state reads and writes as v0.23.0 GDN prefill.
    flushed = leap_window_flush(state[slot_ids], compensator_tokens=config.compensator_tokens)
    state[slot_ids] = flushed.to(state.dtype)
