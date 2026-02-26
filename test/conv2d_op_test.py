# Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.
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
# ==============================================================================
"""Tests for MUSA Conv2D operator."""

import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase


class Conv2DOpTest(MUSATestCase):
  """Tests for MUSA Conv2D operator."""

  def _make_input_and_filter(self,
                             input_shape,
                             filter_shape,
                             dtype,
                             value_range=(-1.0, 1.0),
                             seed=None):
    """Generate Conv2D input/filter tensors."""
    if seed is not None:
      np.random.seed(seed)

    np_dtype = np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype

    low, high = value_range
    x_np = np.random.uniform(low, high, size=input_shape).astype(np_dtype)
    w_np = np.random.uniform(low, high, size=filter_shape).astype(np_dtype)

    x_tf = tf.constant(x_np, dtype=dtype)
    w_tf = tf.constant(w_np, dtype=dtype)
    return x_tf, w_tf

  def _test_conv2d(self,
                   input_shape,
                   filter_shape,
                   dtype,
                   strides,
                   padding,
                   data_format="NHWC",
                   dilations=None,
                   rtol=1e-5,
                   atol=1e-8,
                   seed=123):
    """Compare tf.nn.conv2d on CPU vs MUSA."""
    if dilations is None:
      dilations = [1, 1, 1, 1]

    x_tf, w_tf = self._make_input_and_filter(
        input_shape=input_shape,
        filter_shape=filter_shape,
        dtype=dtype,
        seed=seed)

    def conv2d_wrapper(x, w):
      return tf.nn.conv2d(
          x,
          w,
          strides=strides,
          padding=padding,
          data_format=data_format,
          dilations=dilations)

    self._compare_cpu_musa_results(
        conv2d_wrapper, [x_tf, w_tf], dtype, rtol=rtol, atol=atol)

  def _tolerance_for_dtype(self, dtype):
    if dtype in [tf.float16, tf.bfloat16]:
      return 1e-2, 1e-2
    return 1e-5, 1e-8

  # ---------------------------------------------------------------------------
  # Basic correctness tests (NHWC)
  # ---------------------------------------------------------------------------

  def testConv2DValidNHWCStride1(self):
    """Test Conv2D NHWC + VALID + stride=1."""
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      rtol, atol = self._tolerance_for_dtype(dtype)

      # Small shape
      self._test_conv2d(
          input_shape=[1, 8, 8, 3],
          filter_shape=[3, 3, 3, 4],  # HWIO
          dtype=dtype,
          strides=[1, 1, 1, 1],
          padding="VALID",
          data_format="NHWC",
          dilations=[1, 1, 1, 1],
          rtol=rtol,
          atol=atol)

      # Medium shape
      self._test_conv2d(
          input_shape=[2, 16, 16, 8],
          filter_shape=[3, 3, 8, 16],
          dtype=dtype,
          strides=[1, 1, 1, 1],
          padding="VALID",
          data_format="NHWC",
          dilations=[1, 1, 1, 1],
          rtol=rtol,
          atol=atol)

  def testConv2DSameNHWCStride1(self):
    """Test Conv2D NHWC + SAME + stride=1 (usually symmetric padding)."""
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      rtol, atol = self._tolerance_for_dtype(dtype)

      self._test_conv2d(
          input_shape=[1, 8, 8, 3],
          filter_shape=[3, 3, 3, 4],
          dtype=dtype,
          strides=[1, 1, 1, 1],
          padding="SAME",
          data_format="NHWC",
          dilations=[1, 1, 1, 1],
          rtol=rtol,
          atol=atol)

      self._test_conv2d(
          input_shape=[2, 15, 15, 4],
          filter_shape=[5, 5, 4, 6],
          dtype=dtype,
          strides=[1, 1, 1, 1],
          padding="SAME",
          data_format="NHWC",
          dilations=[1, 1, 1, 1],
          rtol=rtol,
          atol=atol)

  def testConv2DStride2ValidNHWC(self):
    """Test Conv2D NHWC + VALID + stride=2."""
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      rtol, atol = self._tolerance_for_dtype(dtype)

      self._test_conv2d(
          input_shape=[1, 9, 9, 3],
          filter_shape=[3, 3, 3, 8],
          dtype=dtype,
          strides=[1, 2, 2, 1],
          padding="VALID",
          data_format="NHWC",
          dilations=[1, 1, 1, 1],
          rtol=rtol,
          atol=atol)

  def testConv2DDilationValidNHWC(self):
    """Test Conv2D NHWC + VALID + dilation > 1."""
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      rtol, atol = self._tolerance_for_dtype(dtype)

      self._test_conv2d(
          input_shape=[1, 12, 12, 3],
          filter_shape=[3, 3, 3, 5],
          dtype=dtype,
          strides=[1, 1, 1, 1],
          padding="VALID",
          data_format="NHWC",
          dilations=[1, 2, 2, 1],
          rtol=rtol,
          atol=atol)

  def testConv2DPointwise1x1NHWC(self):
    """Test 1x1 Conv2D (pointwise conv)."""
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      rtol, atol = self._tolerance_for_dtype(dtype)

      self._test_conv2d(
          input_shape=[4, 16, 16, 7],
          filter_shape=[1, 1, 7, 13],
          dtype=dtype,
          strides=[1, 1, 1, 1],
          padding="SAME",
          data_format="NHWC",
          dilations=[1, 1, 1, 1],
          rtol=rtol,
          atol=atol)

  # ---------------------------------------------------------------------------
  # Optional NCHW tests (enable once your filter layout / wrapper path is stable)
  # ---------------------------------------------------------------------------

  def testConv2DValidNCHW(self):
    """Test Conv2D NCHW + VALID + stride=1."""
    for dtype in [tf.float32, tf.float16]:
      rtol, atol = self._tolerance_for_dtype(dtype)

      self._test_conv2d(
          input_shape=[1, 3, 8, 8],      # NCHW
          filter_shape=[3, 3, 3, 4],     # HWIO (tf.nn.conv2d filter is still HWIO)
          dtype=dtype,
          strides=[1, 1, 1, 1],          # NCHW format: [N, C, H, W]
          padding="VALID",
          data_format="NCHW",
          dilations=[1, 1, 1, 1],
          rtol=rtol,
          atol=atol)

  # ---------------------------------------------------------------------------
  # Edge cases / special cases
  # ---------------------------------------------------------------------------

  def testConv2DEmptyBatch(self):
    """Test Conv2D with empty batch dimension."""
    for dtype in [tf.float32]:
      rtol, atol = self._tolerance_for_dtype(dtype)

      self._test_conv2d(
          input_shape=[0, 8, 8, 3],      # empty batch
          filter_shape=[3, 3, 3, 4],
          dtype=dtype,
          strides=[1, 1, 1, 1],
          padding="SAME",
          data_format="NHWC",
          dilations=[1, 1, 1, 1],
          rtol=rtol,
          atol=atol)

  def testConv2DZeroInput(self):
    """Test Conv2D with all-zero input for numerical sanity."""
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      rtol, atol = self._tolerance_for_dtype(dtype)
      np_dtype = np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype

      x_np = np.zeros([1, 8, 8, 3], dtype=np_dtype)
      w_np = np.random.uniform(-1, 1, size=[3, 3, 3, 4]).astype(np_dtype)

      x_tf = tf.constant(x_np, dtype=dtype)
      w_tf = tf.constant(w_np, dtype=dtype)

      def conv2d_wrapper(x, w):
        return tf.nn.conv2d(
            x, w, strides=[1, 1, 1, 1], padding="SAME", data_format="NHWC")

      self._compare_cpu_musa_results(
          conv2d_wrapper, [x_tf, w_tf], dtype, rtol=rtol, atol=atol)

  def testConv2DZeroFilter(self):
    """Test Conv2D with all-zero filter (output should be zeros)."""
    for dtype in [tf.float32, tf.float16, tf.bfloat16]:
      rtol, atol = self._tolerance_for_dtype(dtype)
      np_dtype = np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype

      x_np = np.random.uniform(-1, 1, size=[1, 8, 8, 3]).astype(np_dtype)
      w_np = np.zeros([3, 3, 3, 4], dtype=np_dtype)

      x_tf = tf.constant(x_np, dtype=dtype)
      w_tf = tf.constant(w_np, dtype=dtype)

      def conv2d_wrapper(x, w):
        return tf.nn.conv2d(
            x, w, strides=[1, 1, 1, 1], padding="VALID", data_format="NHWC")

      self._compare_cpu_musa_results(
          conv2d_wrapper, [x_tf, w_tf], dtype, rtol=rtol, atol=atol)

  # ---------------------------------------------------------------------------
  # Validation / expected-error tests (important for your current pure muDNN impl)
  # ---------------------------------------------------------------------------

  def testConv2DChannelMismatch(self):
    """Test Conv2D input/filter channel mismatch raises error."""
    x = tf.constant(np.random.uniform(-1, 1, [1, 8, 8, 3]).astype(np.float32))
    w = tf.constant(np.random.uniform(-1, 1, [3, 3, 4, 8]).astype(np.float32))  # IC=4 mismatches input C=3

    # CPU path itself should already reject; this tests TF-level validation behavior.
    with self.assertRaises((tf.errors.InvalidArgumentError, ValueError)):
      tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME", data_format="NHWC")

  def testConv2DInvalidStridesOnBatchOrChannel(self):
    """Test Conv2D rejects strides on batch/channel dims."""
    x = tf.constant(np.random.uniform(-1, 1, [1, 8, 8, 3]).astype(np.float32))
    w = tf.constant(np.random.uniform(-1, 1, [3, 3, 3, 4]).astype(np.float32))

    with self.assertRaises((tf.errors.InvalidArgumentError, ValueError)):
      tf.nn.conv2d(x, w, strides=[2, 1, 1, 1], padding="SAME", data_format="NHWC")

    with self.assertRaises((tf.errors.InvalidArgumentError, ValueError)):
      tf.nn.conv2d(x, w, strides=[1, 1, 1, 2], padding="SAME", data_format="NHWC")

  def testConv2DAsymmetricSamePaddingMayBeUnsupportedInCurrentMuDNNPath(self):
    """Test a case that may trigger asymmetric SAME padding in current pure muDNN path.

    Your current pure muDNN implementation rejects asymmetric padding because
    mudnn_xmma.h Convolution::SetNdInfo appears to accept symmetric pad only.
    This test is written to validate that behavior explicitly on MUSA.
    """
    x = tf.constant(np.random.uniform(-1, 1, [1, 4, 4, 3]).astype(np.float32))
    w = tf.constant(np.random.uniform(-1, 1, [3, 3, 3, 4]).astype(np.float32))

    # CPU should succeed.
    with tf.device('/CPU:0'):
      _ = tf.nn.conv2d(x, w, strides=[1, 2, 2, 1], padding="SAME", data_format="NHWC")

    # MUSA may raise UnimplementedError in current pure muDNN implementation.
    with self.assertRaises((tf.errors.UnimplementedError, tf.errors.InvalidArgumentError)):
      with tf.device('/device:MUSA:0'):
        _ = tf.nn.conv2d(x, w, strides=[1, 2, 2, 1], padding="SAME", data_format="NHWC")

  def testConv2DDeterministicRepeatability(self):
    """Run same Conv2D twice and compare CPU vs MUSA for deterministic consistency."""
    dtype = tf.float32
    rtol, atol = self._tolerance_for_dtype(dtype)

    x_tf, w_tf = self._make_input_and_filter(
        input_shape=[1, 10, 10, 3],
        filter_shape=[3, 3, 3, 8],
        dtype=dtype,
        seed=2026)

    def conv2d_wrapper(x, w):
      return tf.nn.conv2d(
          x, w, strides=[1, 1, 1, 1], padding="SAME", data_format="NHWC")

    # First run compare CPU/MUSA
    self._compare_cpu_musa_results(
        conv2d_wrapper, [x_tf, w_tf], dtype, rtol=rtol, atol=atol)

    # Second run compare CPU/MUSA again
    self._compare_cpu_musa_results(
        conv2d_wrapper, [x_tf, w_tf], dtype, rtol=rtol, atol=atol)


if __name__ == "__main__":
  tf.test.main()