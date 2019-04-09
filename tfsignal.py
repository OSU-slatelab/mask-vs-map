# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""mel conversion ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import fractions

from tensorflow.python.framework import dtypes, ops, tensor_util, tensor_shape
from tensorflow.python.ops import array_ops, math_ops


# mel spectrum constants.
_MEL_BREAK_FREQUENCY_HERTZ = 700.0
_MEL_HIGH_FREQUENCY_Q = 1127.0


def _mel_to_hertz(mel_values, name=None):
  """Converts frequencies in `mel_values` from the mel scale to linear scale.

  Args:
    mel_values: A `Tensor` of frequencies in the mel scale.
    name: An optional name for the operation.

  Returns:
    A `Tensor` of the same shape and type as `mel_values` containing linear
    scale frequencies in Hertz.
  """
  with ops.name_scope(name, 'mel_to_hertz', [mel_values]):
    mel_values = ops.convert_to_tensor(mel_values)
    return _MEL_BREAK_FREQUENCY_HERTZ * (
        math_ops.exp(mel_values / _MEL_HIGH_FREQUENCY_Q) - 1.0
    )


def _hertz_to_mel(frequencies_hertz, name=None):
  """Converts frequencies in `frequencies_hertz` in Hertz to the mel scale.

  Args:
    frequencies_hertz: A `Tensor` of frequencies in Hertz.
    name: An optional name for the operation.

  Returns:
    A `Tensor` of the same shape and type of `frequencies_hertz` containing
    frequencies in the mel scale.
  """
  with ops.name_scope(name, 'hertz_to_mel', [frequencies_hertz]):
    frequencies_hertz = ops.convert_to_tensor(frequencies_hertz)
    return _MEL_HIGH_FREQUENCY_Q * math_ops.log(
        1.0 + (frequencies_hertz / _MEL_BREAK_FREQUENCY_HERTZ))


def _validate_arguments(num_mel_bins, sample_rate,
                        lower_edge_hertz, upper_edge_hertz, dtype):
  """Checks the inputs to linear_to_mel_weight_matrix."""
  if num_mel_bins <= 0:
    raise ValueError('num_mel_bins must be positive. Got: %s' % num_mel_bins)
  if sample_rate <= 0.0:
    raise ValueError('sample_rate must be positive. Got: %s' % sample_rate)
  if lower_edge_hertz < 0.0:
    raise ValueError('lower_edge_hertz must be non-negative. Got: %s' %
                     lower_edge_hertz)
  if lower_edge_hertz >= upper_edge_hertz:
    raise ValueError('lower_edge_hertz %.1f >= upper_edge_hertz %.1f' %
                     (lower_edge_hertz, upper_edge_hertz))
  if upper_edge_hertz > sample_rate / 2:
    raise ValueError('upper_edge_hertz must not be larger than the Nyquist '
                     'frequency (sample_rate / 2). Got: %s for sample_rate: %s'
                     % (upper_edge_hertz, sample_rate))
  if not dtype.is_floating:
    raise ValueError('dtype must be a floating point type. Got: %s' % dtype)


def linear_to_mel_weight_matrix(num_mel_bins=20,
                                num_spectrogram_bins=129,
                                sample_rate=8000,
                                lower_edge_hertz=125.0,
                                upper_edge_hertz=3800.0,
                                dtype=dtypes.float32,
                                name=None):
  """Returns a matrix to warp linear scale spectrograms to the [mel scale][mel].

  Returns a weight matrix that can be used to re-weight a `Tensor` containing
  `num_spectrogram_bins` linearly sampled frequency information from
  `[0, sample_rate / 2]` into `num_mel_bins` frequency information from
  `[lower_edge_hertz, upper_edge_hertz]` on the [mel scale][mel].

  For example, the returned matrix `A` can be used to right-multiply a
  spectrogram `S` of shape `[frames, num_spectrogram_bins]` of linear
  scale spectrum values (e.g. STFT magnitudes) to generate a "mel spectrogram"
  `M` of shape `[frames, num_mel_bins]`.

      # `S` has shape [frames, num_spectrogram_bins]
      # `M` has shape [frames, num_mel_bins]
      M = tf.matmul(S, A)

  The matrix can be used with `tf.tensordot` to convert an arbitrary rank
  `Tensor` of linear-scale spectral bins into the mel scale.

      # S has shape [..., num_spectrogram_bins].
      # M has shape [..., num_mel_bins].
      M = tf.tensordot(S, A, 1)
      # tf.tensordot does not support shape inference for this case yet.
      M.set_shape(S.shape[:-1].concatenate(A.shape[-1:]))

  Args:
    num_mel_bins: Python int. How many bands in the resulting mel spectrum.
    num_spectrogram_bins: An integer `Tensor`. How many bins there are in the
      source spectrogram data, which is understood to be `fft_size // 2 + 1`,
      i.e. the spectrogram only contains the nonredundant FFT bins.
    sample_rate: Python float. Samples per second of the input signal used to
      create the spectrogram. We need this to figure out the actual frequencies
      for each spectrogram bin, which dictates how they are mapped into the mel
      scale.
    lower_edge_hertz: Python float. Lower bound on the frequencies to be
      included in the mel spectrum. This corresponds to the lower edge of the
      lowest triangular band.
    upper_edge_hertz: Python float. The desired top edge of the highest
      frequency band.
    dtype: The `DType` of the result matrix. Must be a floating point type.
    name: An optional name for the operation.

  Returns:
    A `Tensor` of shape `[num_spectrogram_bins, num_mel_bins]`.

  Raises:
    ValueError: If num_mel_bins/num_spectrogram_bins/sample_rate are not
      positive, lower_edge_hertz is negative, frequency edges are incorrectly
      ordered, or upper_edge_hertz is larger than the Nyquist frequency.

  [mel]: https://en.wikipedia.org/wiki/Mel_scale
  """
  with ops.name_scope(name, 'linear_to_mel_weight_matrix') as name:
    # Note: As num_spectrogram_bins is passed to `math_ops.linspace`
    # and the validation is already done in linspace (both in shape function
    # and in kernel), there is no need to validate num_spectrogram_bins here.
    _validate_arguments(num_mel_bins, sample_rate,
                        lower_edge_hertz, upper_edge_hertz, dtype)

    # This function can be constant folded by graph optimization since there are
    # no Tensor inputs.
    sample_rate = ops.convert_to_tensor(
        sample_rate, dtype, name='sample_rate')
    lower_edge_hertz = ops.convert_to_tensor(
        lower_edge_hertz, dtype, name='lower_edge_hertz')
    upper_edge_hertz = ops.convert_to_tensor(
        upper_edge_hertz, dtype, name='upper_edge_hertz')
    zero = ops.convert_to_tensor(0.0, dtype)

    # HTK excludes the spectrogram DC bin.
    bands_to_zero = 1
    nyquist_hertz = sample_rate / 2.0
    linear_frequencies = math_ops.linspace(
        zero, nyquist_hertz, num_spectrogram_bins)[bands_to_zero:]
    spectrogram_bins_mel = array_ops.expand_dims(
        _hertz_to_mel(linear_frequencies), 1)

    # Compute num_mel_bins triples of (lower_edge, center, upper_edge). The
    # center of each band is the lower and upper edge of the adjacent bands.
    # Accordingly, we divide [lower_edge_hertz, upper_edge_hertz] into
    # num_mel_bins + 2 pieces.
    band_edges_mel = frame(
        math_ops.linspace(_hertz_to_mel(lower_edge_hertz),
                          _hertz_to_mel(upper_edge_hertz),
                          num_mel_bins + 2), frame_length=3, frame_step=1)

    # Split the triples up and reshape them into [1, num_mel_bins] tensors.
    lower_edge_mel, center_mel, upper_edge_mel = tuple(array_ops.reshape(
        t, [1, num_mel_bins]) for t in array_ops.split(
            band_edges_mel, 3, axis=1))

    # Calculate lower and upper slopes for every spectrogram bin.
    # Line segments are linear in the mel domain, not Hertz.
    lower_slopes = (spectrogram_bins_mel - lower_edge_mel) / (
        center_mel - lower_edge_mel)
    upper_slopes = (upper_edge_mel - spectrogram_bins_mel) / (
        upper_edge_mel - center_mel)

    # Intersect the line segments with each other and zero.
    mel_weights_matrix = math_ops.maximum(
        zero, math_ops.minimum(lower_slopes, upper_slopes))

    # Re-add the zeroed lower bins we sliced out above.
    return array_ops.pad(
        mel_weights_matrix, [[bands_to_zero, 0], [0, 0]], name=name)


def _infer_frame_shape(signal, frame_length, frame_step, pad_end, axis):
  """Infers the shape of the return value of `frame`."""
  frame_length = tensor_util.constant_value(frame_length)
  frame_step = tensor_util.constant_value(frame_step)
  axis = tensor_util.constant_value(axis)
  if signal.shape.ndims is None:
    return None
  if axis is None:
    return [None] * (signal.shape.ndims + 1)

  signal_shape = signal.shape.as_list()
  num_frames = None
  frame_axis = signal_shape[axis]
  outer_dimensions = signal_shape[:axis]
  inner_dimensions = signal_shape[axis:][1:]
  if signal_shape and frame_axis is not None:
    if frame_step is not None and pad_end:
      # Double negative is so that we round up.
      num_frames = max(0, -(-frame_axis // frame_step))
    elif frame_step is not None and frame_length is not None:
      assert not pad_end
      num_frames = max(
          0, (frame_axis - frame_length + frame_step) // frame_step)
  return outer_dimensions + [num_frames, frame_length] + inner_dimensions


def frame(signal, frame_length, frame_step, pad_end=False, pad_value=0, axis=-1,
          name=None):
  """Expands `signal`'s `axis` dimension into frames of `frame_length`.

  Slides a window of size `frame_length` over `signal`'s `axis` dimension
  with a stride of `frame_step`, replacing the `axis` dimension with
  `[frames, frame_length]` frames.

  If `pad_end` is True, window positions that are past the end of the `axis`
  dimension are padded with `pad_value` until the window moves fully past the
  end of the dimension. Otherwise, only window positions that fully overlap the
  `axis` dimension are produced.

  For example:

  ```python
  pcm = tf.placeholder(tf.float32, [None, 9152])
  frames = tf.signal.frame(pcm, 512, 180)
  magspec = tf.abs(tf.signal.rfft(frames, [512]))
  image = tf.expand_dims(magspec, 3)
  ```

  Args:
    signal: A `[..., samples, ...]` `Tensor`. The rank and dimensions
      may be unknown. Rank must be at least 1.
    frame_length: The frame length in samples. An integer or scalar `Tensor`.
    frame_step: The frame hop size in samples. An integer or scalar `Tensor`.
    pad_end: Whether to pad the end of `signal` with `pad_value`.
    pad_value: An optional scalar `Tensor` to use where the input signal
      does not exist when `pad_end` is True.
    axis: A scalar integer `Tensor` indicating the axis to frame. Defaults to
      the last axis. Supports negative values for indexing from the end.
    name: An optional name for the operation.

  Returns:
    A `Tensor` of frames with shape `[..., frames, frame_length, ...]`.

  Raises:
    ValueError: If `frame_length`, `frame_step`, `pad_value`, or `axis` are not
      scalar.
  """
  with ops.name_scope(name, "frame", [signal, frame_length, frame_step,
                                      pad_value]):
    signal = ops.convert_to_tensor(signal, name="signal")
    frame_length = ops.convert_to_tensor(frame_length, name="frame_length")
    frame_step = ops.convert_to_tensor(frame_step, name="frame_step")
    axis = ops.convert_to_tensor(axis, name="axis")

    signal.shape.with_rank_at_least(1)
    frame_length.shape.assert_has_rank(0)
    frame_step.shape.assert_has_rank(0)
    axis.shape.assert_has_rank(0)

    result_shape = _infer_frame_shape(signal, frame_length, frame_step, pad_end,
                                      axis)

    # Axis can be negative. Convert it to positive.
    signal_rank = array_ops.rank(signal)
    axis = math_ops.range(signal_rank)[axis]

    signal_shape = array_ops.shape(signal)
    outer_dimensions, length_samples, inner_dimensions = array_ops.split(
        signal_shape, [axis, 1, signal_rank - 1 - axis])
    length_samples = array_ops.reshape(length_samples, [])
    num_outer_dimensions = array_ops.size(outer_dimensions)
    num_inner_dimensions = array_ops.size(inner_dimensions)

    # If padding is requested, pad the input signal tensor with pad_value.
    if pad_end:
      pad_value = ops.convert_to_tensor(pad_value, signal.dtype)
      pad_value.shape.assert_has_rank(0)

      # Calculate number of frames, using double negatives to round up.
      num_frames = -(-length_samples // frame_step)

      # Pad the signal by up to frame_length samples based on how many samples
      # are remaining starting from last_frame_position.
      pad_samples = math_ops.maximum(
          0, frame_length + frame_step * (num_frames - 1) - length_samples)

      # Pad the inner dimension of signal by pad_samples.
      paddings = array_ops.concat(
          [array_ops.zeros([num_outer_dimensions, 2], dtype=pad_samples.dtype),
           [[0, pad_samples]],
           array_ops.zeros([num_inner_dimensions, 2], dtype=pad_samples.dtype)],
          0)
      signal = array_ops.pad(signal, paddings, constant_values=pad_value)

      signal_shape = array_ops.shape(signal)
      length_samples = signal_shape[axis]
    else:
      num_frames = math_ops.maximum(
          0, 1 + (length_samples - frame_length) // frame_step)

    subframe_length = gcd(frame_length, frame_step)
    subframes_per_frame = frame_length // subframe_length
    subframes_per_hop = frame_step // subframe_length
    num_subframes = length_samples // subframe_length

    slice_shape = array_ops.concat([outer_dimensions,
                                    [num_subframes * subframe_length],
                                    inner_dimensions], 0)
    subframe_shape = array_ops.concat([outer_dimensions,
                                       [num_subframes, subframe_length],
                                       inner_dimensions], 0)
    subframes = array_ops.reshape(array_ops.strided_slice(
        signal, array_ops.zeros_like(signal_shape),
        slice_shape), subframe_shape)

    # frame_selector is a [num_frames, subframes_per_frame] tensor
    # that indexes into the appropriate frame in subframes. For example:
    # [[0, 0, 0, 0], [2, 2, 2, 2], [4, 4, 4, 4]]
    frame_selector = array_ops.reshape(
        math_ops.range(num_frames) * subframes_per_hop, [num_frames, 1])

    # subframe_selector is a [num_frames, subframes_per_frame] tensor
    # that indexes into the appropriate subframe within a frame. For example:
    # [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]
    subframe_selector = array_ops.reshape(
        math_ops.range(subframes_per_frame), [1, subframes_per_frame])

    # Adding the 2 selector tensors together produces a [num_frames,
    # subframes_per_frame] tensor of indices to use with tf.gather to select
    # subframes from subframes. We then reshape the inner-most
    # subframes_per_frame dimension to stitch the subframes together into
    # frames. For example: [[0, 1, 2, 3], [2, 3, 4, 5], [4, 5, 6, 7]].
    selector = frame_selector + subframe_selector

    frames = array_ops.reshape(
        array_ops.gather(subframes, selector),#, axis=axis),
        array_ops.concat([outer_dimensions, [num_frames, frame_length],
                          inner_dimensions], 0))

    if result_shape:
      frames.set_shape(result_shape)
    return frames


def gcd(a, b, name=None):
  """Returns the greatest common divisor via Euclid's algorithm.
  Args:
    a: The dividend. A scalar integer `Tensor`.
    b: The divisor. A scalar integer `Tensor`.
    name: An optional name for the operation.
  Returns:
    A scalar `Tensor` representing the greatest common divisor between `a` and
    `b`.
  Raises:
    ValueError: If `a` or `b` are not scalar integers.
  """
  with ops.name_scope(name, 'gcd', [a, b]):
    a = ops.convert_to_tensor(a)
    b = ops.convert_to_tensor(b)

    a.shape.assert_has_rank(0)
    b.shape.assert_has_rank(0)

    if not a.dtype.is_integer:
      raise ValueError('a must be an integer type. Got: %s' % a.dtype)
    if not b.dtype.is_integer:
      raise ValueError('b must be an integer type. Got: %s' % b.dtype)

    # TPU requires static shape inference. GCD is used for subframe size
    # computation, so we should prefer static computation where possible.
    const_a = tensor_util.constant_value(a)
    const_b = tensor_util.constant_value(b)
    if const_a is not None and const_b is not None:
      return ops.convert_to_tensor(fractions.gcd(const_a, const_b))

    cond = lambda _, b: math_ops.greater(b, array_ops.zeros_like(b))
    body = lambda a, b: [b, math_ops.mod(a, b)]
    a, b = control_flow_ops.while_loop(cond, body, [a, b], back_prop=False)
    return a

def mfccs_from_log_mel_spectrograms(log_mel_spectrograms, name=None):
  """Computes [MFCCs][mfcc] of `log_mel_spectrograms`.

  Implemented with GPU-compatible ops and supports gradients.

  [Mel-Frequency Cepstral Coefficient (MFCC)][mfcc] calculation consists of
  taking the DCT-II of a log-magnitude mel-scale spectrogram. [HTK][htk]'s MFCCs
  use a particular scaling of the DCT-II which is almost orthogonal
  normalization. We follow this convention.

  All `num_mel_bins` MFCCs are returned and it is up to the caller to select
  a subset of the MFCCs based on their application. For example, it is typical
  to only use the first few for speech recognition, as this results in
  an approximately pitch-invariant representation of the signal.

  For example:

  ```python
  sample_rate = 16000.0
  # A Tensor of [batch_size, num_samples] mono PCM samples in the range [-1, 1].
  pcm = tf.placeholder(tf.float32, [None, None])

  # A 1024-point STFT with frames of 64 ms and 75% overlap.
  stfts = tf.signal.stft(pcm, frame_length=1024, frame_step=256,
                         fft_length=1024)
  spectrograms = tf.abs(stfts)

  # Warp the linear scale spectrograms into the mel-scale.
  num_spectrogram_bins = stfts.shape[-1].value
  lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, 80
  linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
    num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz,
    upper_edge_hertz)
  mel_spectrograms = tf.tensordot(
    spectrograms, linear_to_mel_weight_matrix, 1)
  mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(
    linear_to_mel_weight_matrix.shape[-1:]))

  # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
  log_mel_spectrograms = tf.log(mel_spectrograms + 1e-6)

  # Compute MFCCs from log_mel_spectrograms and take the first 13.
  mfccs = tf.signal.mfccs_from_log_mel_spectrograms(
    log_mel_spectrograms)[..., :13]
  ```

  Args:
    log_mel_spectrograms: A `[..., num_mel_bins]` `float32` `Tensor` of
      log-magnitude mel-scale spectrograms.
    name: An optional name for the operation.
  Returns:
    A `[..., num_mel_bins]` `float32` `Tensor` of the MFCCs of
    `log_mel_spectrograms`.

  Raises:
    ValueError: If `num_mel_bins` is not positive.

  [mfcc]: https://en.wikipedia.org/wiki/Mel-frequency_cepstrum
  [htk]: https://en.wikipedia.org/wiki/HTK_(software)
  """
  with ops.name_scope(name, 'mfccs_from_log_mel_spectrograms',
                      [log_mel_spectrograms]):
    # Compute the DCT-II of the resulting log-magnitude mel-scale spectrogram.
    # The DCT used in HTK scales every basis vector by sqrt(2/N), which is the
    # scaling required for an "orthogonal" DCT-II *except* in the 0th bin, where
    # the true orthogonal DCT (as implemented by scipy) scales by sqrt(1/N). For
    # this reason, we don't apply orthogonal normalization and scale the DCT by
    # `0.5 * sqrt(2/N)` manually.
    log_mel_spectrograms = ops.convert_to_tensor(log_mel_spectrograms,
                                                 dtype=dtypes.float32)
    if (log_mel_spectrograms.shape.ndims and
        log_mel_spectrograms.shape.dims[-1].value is not None):
      num_mel_bins = log_mel_spectrograms.shape.dims[-1].value
      if num_mel_bins == 0:
        raise ValueError('num_mel_bins must be positive. Got: %s' %
                         log_mel_spectrograms)
    else:
      num_mel_bins = array_ops.shape(log_mel_spectrograms)[-1]

    dct2 = dct(log_mel_spectrograms, type=2)
    return dct2 * math_ops.rsqrt(math_ops.to_float(num_mel_bins) * 2.0)


def _validate_dct_arguments(input_tensor, dct_type, n, axis, norm):
  """Checks that DCT/IDCT arguments are compatible and well formed."""
  if n is not None:
    raise NotImplementedError("The DCT length argument is not implemented.")
  if axis != -1:
    raise NotImplementedError("axis must be -1. Got: %s" % axis)
  if dct_type not in (1, 2, 3):
    raise ValueError("Only Types I, II and III (I)DCT are supported.")
  if dct_type == 1:
    if norm == "ortho":
      raise ValueError("Normalization is not supported for the Type-I DCT.")
    if input_tensor.shape[-1] is not None and input_tensor.shape[-1] < 2:
      raise ValueError(
          "Type-I DCT requires the dimension to be greater than one.")

  if norm not in (None, "ortho"):
    raise ValueError(
        "Unknown normalization. Expected None or 'ortho', got: %s" % norm)


# TODO(rjryan): Implement `n` and `axis` parameters.
def dct(input, type=2, n=None, axis=-1, norm=None, name=None):  # pylint: disable=redefined-builtin
  """Computes the 1D [Discrete Cosine Transform (DCT)][dct] of `input`.

  Currently only Types I, II and III are supported.
  Type I is implemented using a length `2N` padded `tf.spectral.rfft`.
  Type II is implemented using a length `2N` padded `tf.spectral.rfft`, as
  described here:
  https://dsp.stackexchange.com/a/10606.
  Type III is a fairly straightforward inverse of Type II
  (i.e. using a length `2N` padded `tf.spectral.irfft`).

  @compatibility(scipy)
  Equivalent to scipy.fftpack.dct for Type-I, Type-II and Type-III DCT.
  https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
  @end_compatibility

  Args:
    input: A `[..., samples]` `float32` `Tensor` containing the signals to
      take the DCT of.
    type: The DCT type to perform. Must be 1, 2 or 3.
    n: For future expansion. The length of the transform. Must be `None`.
    axis: For future expansion. The axis to compute the DCT along. Must be `-1`.
    norm: The normalization to apply. `None` for no normalization or `'ortho'`
      for orthonormal normalization.
    name: An optional name for the operation.

  Returns:
    A `[..., samples]` `float32` `Tensor` containing the DCT of `input`.

  Raises:
    ValueError: If `type` is not `1`, `2` or `3`, `n` is not `None, `axis` is
      not `-1`, or `norm` is not `None` or `'ortho'`.
    ValueError: If `type` is `1` and `norm` is `ortho`.

  [dct]: https://en.wikipedia.org/wiki/Discrete_cosine_transform
  """
  _validate_dct_arguments(input, type, n, axis, norm)
  with ops.name_scope(name, "dct", [input]):
    # We use the RFFT to compute the DCT and TensorFlow only supports float32
    # for FFTs at the moment.
    input = ops.convert_to_tensor(input, dtype=dtypes.float32)

    axis_dim = (tensor_shape.dimension_value(input.shape[-1])
                or array_ops.shape(input)[-1])
    axis_dim_float = math_ops.to_float(axis_dim)

    if type == 1:
      dct1_input = array_ops.concat([input, input[..., -2:0:-1]], axis=-1)
      dct1 = math_ops.real(fft_ops.rfft(dct1_input))
      return dct1

    if type == 2:
      scale = 2.0 * math_ops.exp(
          math_ops.complex(
              0.0, -math_ops.range(axis_dim_float) * _math.pi * 0.5 /
              axis_dim_float))

      # TODO(rjryan): Benchmark performance and memory usage of the various
      # approaches to computing a DCT via the RFFT.
      dct2 = math_ops.real(
          fft_ops.rfft(
              input, fft_length=[2 * axis_dim])[..., :axis_dim] * scale)

      if norm == "ortho":
        n1 = 0.5 * math_ops.rsqrt(axis_dim_float)
        n2 = n1 * math_ops.sqrt(2.0)
        # Use tf.pad to make a vector of [n1, n2, n2, n2, ...].
        weights = array_ops.pad(
            array_ops.expand_dims(n1, 0), [[0, axis_dim - 1]],
            constant_values=n2)
        dct2 *= weights

      return dct2

    elif type == 3:
      if norm == "ortho":
        n1 = _math_ops.sqrt(axis_dim_float)
        n2 = n1 * math_ops.sqrt(0.5)
        # Use tf.pad to make a vector of [n1, n2, n2, n2, ...].
        weights = array_ops.pad(
            array_ops.expand_dims(n1, 0), [[0, axis_dim - 1]],
            constant_values=n2)
        input *= weights
      else:
        input *= axis_dim_float
      scale = 2.0 * math_ops.exp(
          math_ops.complex(
              0.0,
              math_ops.range(axis_dim_float) * _math.pi * 0.5 /
              axis_dim_float))
      dct3 = _math_ops.real(
          fft_ops.irfft(
              scale * math_ops.complex(input, 0.0),
              fft_length=[2 * axis_dim]))[..., :axis_dim]

      return dct3

