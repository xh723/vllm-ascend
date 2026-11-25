# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Atlas A2 smoke test with Qwen3.8-27B's per-rank GDN dimensions at TP=4."""

import torch

from vllm_ascend.ascend_config import LeapQuantConfig
from vllm_ascend.quantization.methods.mamba_state.leapquant import leap_window_flush, leapquant_flush_state_rows
from vllm_ascend.utils import AscendDeviceType, enable_custom_op, get_ascend_device_type

KEY_HEADS, VALUE_HEADS = 4, 12
HEAD_DIM = 128
SLOTS, BATCH_SIZE, WINDOW = 6, 2, 16
COMPENSATOR_TOKENS = 4


@torch.inference_mode()
def test_a2_gdn_recurrent_state_flush():
    torch.manual_seed(42)
    device = torch.device("npu:0")
    torch.npu.set_device(device)
    assert get_ascend_device_type() == AscendDeviceType.A2
    assert enable_custom_op()

    steps = 2 * WINDOW + 1
    qk_shape = (steps, BATCH_SIZE, KEY_HEADS, HEAD_DIM)
    value_shape = (steps, BATCH_SIZE, VALUE_HEADS, HEAD_DIM)
    query = torch.nn.functional.normalize(torch.randn(qk_shape, device=device), dim=-1).to(torch.bfloat16)
    key = torch.nn.functional.normalize(torch.randn(qk_shape, device=device), dim=-1).to(torch.bfloat16)
    value = torch.randn(value_shape, device=device, dtype=torch.bfloat16)
    beta = (0.5 + 0.5 * torch.rand(value_shape[:-1], device=device)).to(torch.bfloat16)
    decay = -0.1 * torch.rand(value_shape[:-1], device=device, dtype=torch.float32)
    state = torch.zeros(SLOTS, VALUE_HEADS, HEAD_DIM, HEAD_DIM, device=device, dtype=torch.float32)
    live_slots = [1, 3]
    slot_ids = torch.tensor(live_slots, device=device, dtype=torch.int32)
    query_lengths = torch.tensor([0] + [1] * BATCH_SIZE, device=device, dtype=torch.int32)
    seq_lens = torch.zeros(BATCH_SIZE, device="cpu", dtype=torch.int64)
    config = LeapQuantConfig(
        enabled=True, allow_experimental_a2=True, window=WINDOW, compensator_tokens=COMPENSATOR_TOKENS
    )

    for step in range(steps):
        output = torch.ops._C_ascend.npu_recurrent_gated_delta_rule(
            query=query[step],
            key=key[step],
            value=value[step],
            state=state,
            g=decay[step],
            beta=beta[step],
            scale=HEAD_DIM**-0.5,
            actual_seq_lengths=query_lengths,
            ssm_state_indices=slot_ids,
        )
        seq_lens += 1
        at_boundary = (step + 1) % WINDOW == 0
        if at_boundary:
            before = state.cpu()[live_slots]
        leapquant_flush_state_rows(state, slot_ids, seq_lens, config=config)
        if at_boundary:
            after = state.cpu()[live_slots]
            expected = leap_window_flush(before, compensator_tokens=COMPENSATOR_TOKENS)
            assert not torch.equal(after, before)
            assert (after - expected).norm() / expected.norm() < 0.01

    assert torch.isfinite(output.cpu()).all()
    assert torch.count_nonzero(state.cpu()[[0, 2, 4, 5]]) == 0
