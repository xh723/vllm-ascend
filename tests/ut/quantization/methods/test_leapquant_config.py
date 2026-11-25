# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the LeapQuantConfig additional_config schema."""

import runpy
import unittest
from types import ModuleType
from unittest.mock import patch

import pytest

from vllm_ascend import ascend_config
from vllm_ascend.ascend_config import LeapQuantConfig


class TestLeapQuantConfig(unittest.TestCase):
    def test_import_while_vllm_config_is_initializing(self):
        with patch.dict("sys.modules", {"vllm.config": ModuleType("vllm.config")}):
            module = runpy.run_path(ascend_config.__file__)
            config = module["LeapQuantConfig"](enabled=True, allow_experimental_a2=True)
        self.assertTrue(config.enabled)
        self.assertTrue(config.allow_experimental_a2)

    def test_defaults(self):
        config = LeapQuantConfig()
        self.assertFalse(config.enabled)
        self.assertFalse(config.allow_experimental_a2)
        self.assertEqual(config.window, 16)
        self.assertEqual(config.compensator_tokens, 4)
        self.assertEqual(config.bits, 8)

    def test_valid_config(self):
        config = LeapQuantConfig(enabled=True, window=32, compensator_tokens=2)
        self.assertTrue(config.enabled)
        self.assertEqual(config.window, 32)
        self.assertEqual(config.compensator_tokens, 2)
        self.assertIn("window=32", config.summary)

    def test_atlas_a2_opt_in(self):
        config = LeapQuantConfig(enabled=True, allow_experimental_a2=True)
        self.assertTrue(config.allow_experimental_a2)
        self.assertIn("allow_experimental_a2=True", config.summary)

    def test_window_choices(self):
        for window in (8, 16, 32, 64):
            LeapQuantConfig(window=window)  # accepted
        for window in (0, 12, 128, -16):
            with pytest.raises(ValueError):
                LeapQuantConfig(window=window)

    def test_compensator_tokens_range(self):
        for rank in (1, 4, 8):
            LeapQuantConfig(compensator_tokens=rank)
        for rank in (0, 9, -1):
            with pytest.raises(ValueError):
                LeapQuantConfig(compensator_tokens=rank)

    def test_bits_only_int8(self):
        LeapQuantConfig(bits=8)
        with pytest.raises(ValueError):
            LeapQuantConfig(bits=4)

    def test_unknown_key_rejected(self):
        # extra="forbid" must fail fast on typos like "windows".
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            LeapQuantConfig(windows=16)


if __name__ == "__main__":
    unittest.main()
