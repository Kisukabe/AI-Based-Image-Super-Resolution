#!/usr/bin/env python3
"""Unit tests cho Bit-Accurate Python Golden Model."""

import unittest
import numpy as np
import sys
from pathlib import Path

# Đảm bảo import được module golden_model khi chạy từ bất kỳ thư mục nào
sys.path.insert(0, str(Path(__file__).resolve().parent))

from golden_model import (
    load_hardware_weights,
    conv_layer1_rtl,
    conv_layer2_rtl,
    conv_layer3_rtl,
    CompactSRCNN_GoldenModel,
)


class TestGoldenModel(unittest.TestCase):
    """Bộ kiểm thử xác thực bit-exact cho Python Golden Model."""

    @classmethod
    def setUpClass(cls):
        cls.weights_dir = Path(__file__).resolve().parent.parent / "weights_fixed_point"
        cls.trained_weights = cls.weights_dir / "weights_hex_clean.txt"
        cls.trained_biases = cls.weights_dir / "biases_hex_clean.txt"
        cls.func_weights = cls.weights_dir / "functional_weights.txt"
        cls.func_biases = cls.weights_dir / "functional_biases.txt"
        cls.smoke_weights = cls.weights_dir / "smoke_weights.txt"
        cls.smoke_biases = cls.weights_dir / "smoke_biases.txt"

    def test_weight_parsing(self):
        """Kiểm tra việc nạp trọng số huấn luyện đủ 1.624 trọng số và 25 bias."""
        hw = load_hardware_weights(self.trained_weights, self.trained_biases)
        self.assertEqual(hw.w1.shape, (16, 1, 9, 9))
        self.assertEqual(hw.b1.shape, (16,))
        self.assertEqual(hw.w2.shape, (8, 16, 1, 1))
        self.assertEqual(hw.b2.shape, (8,))
        self.assertEqual(hw.w3.shape, (1, 8, 5, 5))
        self.assertEqual(hw.b3.shape, (1,))

    def test_tb_axis_functional_reproduction(self):
        """Kiểm tra tái lập đúng 100% logic testbench RTL tb_axis_functional.v."""
        hw = load_hardware_weights(self.func_weights, self.func_biases)
        input_s7 = np.arange(16, dtype=np.int8).reshape(4, 4)

        l1 = conv_layer1_rtl(input_s7, hw.w1, hw.b1)
        l2 = conv_layer2_rtl(l1, hw.w2, hw.b2)
        out = conv_layer3_rtl(l2, hw.w3, int(hw.b3[0]))

        expected = np.clip(128 + (input_s7.astype(int) // 8), 0, 255).astype(np.uint8)
        max_error = np.max(np.abs(out.astype(int) - expected.astype(int)))
        self.assertEqual(max_error, 0, "Sai số phải tuyệt đối bằng 0 đối với tb_axis_functional.")

    def test_smoke_zero_weights(self):
        """Kiểm tra với trọng số bằng 0, output luôn luôn bằng đúng 128."""
        hw = load_hardware_weights(self.smoke_weights, self.smoke_biases)
        dummy_input = np.random.randint(-128, 128, (16, 16), dtype=np.int8)

        l1 = conv_layer1_rtl(dummy_input, hw.w1, hw.b1)
        l2 = conv_layer2_rtl(l1, hw.w2, hw.b2)
        out = conv_layer3_rtl(l2, hw.w3, int(hw.b3[0]))

        self.assertTrue(np.all(out == 128), "Output với trọng số 0 phải là 128.")

    def test_golden_model_full_image(self):
        """Kiểm tra chạy ảnh ngẫu nhiên 64x64 và bảo toàn shape và dtypes."""
        model = CompactSRCNN_GoldenModel(self.trained_weights, self.trained_biases)
        dummy_img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        out = model.process_image(dummy_img)

        self.assertEqual(out.shape, (64, 64))
        self.assertEqual(out.dtype, np.uint8)
        self.assertTrue(np.all((out >= 0) & (out <= 255)))


if __name__ == "__main__":
    unittest.main()
