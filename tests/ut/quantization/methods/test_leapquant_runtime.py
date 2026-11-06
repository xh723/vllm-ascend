# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the LeapQuant runtime glue (row selection + pool flush)."""

import unittest
from typing import Any
from unittest.mock import MagicMock, patch

import torch

from vllm_ascend.ascend_config import LeapQuantConfig
from vllm_ascend.quantization.methods.mamba_state.leapquant import (
    leapquant_config_or_none,
    leapquant_flush_state_rows,
    runtime,
)


class TestLeapQuantRuntime(unittest.TestCase):
    def setUp(self):
        self.config = LeapQuantConfig(enabled=True, window=16, compensator_tokens=4)
        torch.manual_seed(11)

    def test_only_due_rows_flushed(self):
        pool = torch.randn(8, 4, 32, 48)
        indices = torch.tensor([3, 5, 6])
        seq_lens = torch.tensor([15, 16, 32])  # rows 1 (slot 5) and 2 (slot 6) due
        before = pool.clone()
        leapquant_flush_state_rows(pool, indices, seq_lens, config=self.config)
        self.assertFalse(bool((pool[3] != before[3]).any()))
        self.assertTrue(bool((pool[5] != before[5]).any()))
        self.assertTrue(bool((pool[6] != before[6]).any()))
        untouched = [0, 1, 2, 4, 7]
        self.assertTrue(bool((pool[untouched] == before[untouched]).all()))

    def test_no_due_rows_is_noop(self):
        pool = torch.randn(4, 2, 16, 16)
        before = pool.clone()
        leapquant_flush_state_rows(pool, torch.tensor([0, 1]), torch.tensor([5, 33]), config=self.config)
        self.assertTrue(torch.equal(pool, before))

    def test_empty_indices(self):
        pool = torch.randn(4, 2, 16, 16)
        before = pool.clone()
        leapquant_flush_state_rows(pool, torch.zeros(0, dtype=torch.long), torch.zeros(0), config=self.config)
        self.assertTrue(torch.equal(pool, before))

    def test_zero_length_does_not_flush_padding(self):
        pool = torch.randn(4, 2, 16, 16)
        before = pool.clone()
        leapquant_flush_state_rows(pool, torch.tensor([0, 1]), torch.tensor([0, 16]), config=self.config)
        self.assertTrue(torch.equal(pool[0], before[0]))
        self.assertFalse(torch.equal(pool[1], before[1]))

    def test_padded_stride_pool(self):
        # Slots embedded as a non-contiguous view (hybrid padded pools).
        wide = torch.randn(8, 4, 32, 64)
        view = wide[:, :, :, :48]
        indices = torch.tensor([2, 4])
        seq_lens = torch.tensor([31, 48])
        before = view.clone()
        leapquant_flush_state_rows(view, indices, seq_lens, config=self.config)
        self.assertFalse(bool((view[2] != before[2]).any()))  # 31 % 16 != 0
        self.assertTrue(bool((view[4] != before[4]).any()))  # 48 % 16 == 0

    def test_device_gate_retries_until_a3_is_confirmed(self):
        devices = MagicMock(A2=object(), A3=object())
        detect = MagicMock(side_effect=[RuntimeError("device unavailable"), devices.A2, devices.A2, devices.A3])
        utils = MagicMock(AscendDeviceType=devices, get_ascend_device_type=detect)
        with (
            patch.dict("sys.modules", {"vllm_ascend.utils": utils}),
            patch.object(runtime, "_device_gate_checked", False),
        ):
            runtime._check_device_once()
            for _ in range(2):
                with self.assertRaisesRegex(RuntimeError, "Atlas A3 only"):
                    runtime._check_device_once()
            runtime._check_device_once()
            runtime._check_device_once()
        self.assertEqual(detect.call_count, 4)

    def test_config_or_none_degrades_without_singleton(self):
        with patch(
            "vllm_ascend.ascend_config.get_ascend_config",
            side_effect=RuntimeError("Ascend config is not initialized"),
        ):
            self.assertIsNone(leapquant_config_or_none())

    def test_config_or_none_disabled(self):
        mock = MagicMock()
        mock.leapquant_config = LeapQuantConfig(enabled=False)
        with patch("vllm_ascend.ascend_config.get_ascend_config", return_value=mock):
            self.assertIsNone(leapquant_config_or_none())

    def test_config_or_none_enabled(self):
        mock = MagicMock()
        mock.leapquant_config = self.config
        with patch("vllm_ascend.ascend_config.get_ascend_config", return_value=mock):
            resolved: Any = leapquant_config_or_none()
            self.assertIs(resolved, self.config)


if __name__ == "__main__":
    unittest.main()
