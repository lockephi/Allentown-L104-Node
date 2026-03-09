"""
Apple Metal GPU Acceleration — Additive synthesis and envelope multiply.

Falls back gracefully to NumPy on non-Metal hardware.

INVARIANT: 527.5184818492612 | PILOT: LONDEL
"""

from __future__ import annotations

import ctypes
import platform
import numpy as np

# ── Metal GPU Initialization ────────────────────────────────────────────────
METAL_AVAILABLE = False
_metal_device = None
_metal_queue = None
_metal_lib = None
_MetalFramework = None

try:
    if platform.system() == "Darwin":
        import objc
        from Foundation import NSBundle
        Metal = NSBundle.bundleWithPath_("/System/Library/Frameworks/Metal.framework")
        if Metal and Metal.load():
            import Metal as _MetalFramework_module
            _MetalFramework = _MetalFramework_module
            _metal_device = _MetalFramework.MTLCreateSystemDefaultDevice()
            if _metal_device is not None:
                _metal_queue = _metal_device.newCommandQueue()
                _METAL_KERNEL_SRC = """
#include <metal_stdlib>
using namespace metal;

kernel void additive_synthesis(
    device const float *freqs        [[buffer(0)]],
    device const float *phases       [[buffer(1)]],
    device const float *weights      [[buffer(2)]],
    device float       *output       [[buffer(3)]],
    device const float *params       [[buffer(4)]],
    uint               gid           [[thread_position_in_grid]])
{
    float sr      = params[0];
    int n_part    = int(params[1]);
    float t_sec   = float(gid) / sr;
    float sample  = 0.0;
    float TWO_PI  = 6.2831853071795864;
    for (int i = 0; i < n_part; i++) {
        sample += weights[i] * sin(TWO_PI * freqs[i] * t_sec + phases[i]);
    }
    output[gid] = sample;
}

kernel void envelope_multiply(
    device float       *signal       [[buffer(0)]],
    device const float *envelope     [[buffer(1)]],
    uint               gid           [[thread_position_in_grid]])
{
    signal[gid] *= envelope[gid];
}
"""
                _err = objc.nil
                _metal_lib = _metal_device.newLibraryWithSource_options_error_(
                    _METAL_KERNEL_SRC, objc.nil, None
                )
                if isinstance(_metal_lib, tuple):
                    _metal_lib = _metal_lib[0]
                if _metal_lib is not None:
                    METAL_AVAILABLE = True
except Exception:
    METAL_AVAILABLE = False


def metal_additive_synthesis(
    freqs: np.ndarray,
    phases: np.ndarray,
    weights: np.ndarray,
    n_samples: int,
    sample_rate: int,
) -> np.ndarray:
    """GPU-accelerated additive sinusoidal synthesis via Apple Metal.

    Falls back to NumPy CPU synthesis if Metal is unavailable.

    Parameters
    ----------
    freqs : ndarray — partial frequencies
    phases : ndarray — base phase offsets
    weights : ndarray — amplitude weights
    n_samples : int — total output samples
    sample_rate : int — audio sample rate

    Returns
    -------
    signal : ndarray (float64)
    """
    if not METAL_AVAILABLE:
        # CPU fallback: vectorized NumPy
        t = np.arange(n_samples, dtype=np.float64) / sample_rate
        signal = np.zeros(n_samples, dtype=np.float64)
        for i in range(len(freqs)):
            signal += weights[i] * np.sin(2.0 * np.pi * freqs[i] * t + phases[i])
        return signal

    try:
        f32_freqs = freqs.astype(np.float32)
        f32_phases = phases.astype(np.float32)
        f32_weights = weights.astype(np.float32)
        f32_params = np.array([float(sample_rate), float(len(freqs))], dtype=np.float32)
        f32_output = np.zeros(n_samples, dtype=np.float32)

        buf_freqs = _metal_device.newBufferWithBytes_length_options_(
            f32_freqs.tobytes(), f32_freqs.nbytes, 0)
        buf_phases = _metal_device.newBufferWithBytes_length_options_(
            f32_phases.tobytes(), f32_phases.nbytes, 0)
        buf_weights = _metal_device.newBufferWithBytes_length_options_(
            f32_weights.tobytes(), f32_weights.nbytes, 0)
        buf_output = _metal_device.newBufferWithLength_options_(
            f32_output.nbytes, 0)
        buf_params = _metal_device.newBufferWithBytes_length_options_(
            f32_params.tobytes(), f32_params.nbytes, 0)

        fn = _metal_lib.newFunctionWithName_("additive_synthesis")
        pipeline = _metal_device.newComputePipelineStateWithFunction_error_(fn, None)
        if isinstance(pipeline, tuple):
            pipeline = pipeline[0]

        cmd_buf = _metal_queue.commandBuffer()
        encoder = cmd_buf.computeCommandEncoder()
        encoder.setComputePipelineState_(pipeline)
        encoder.setBuffer_offset_atIndex_(buf_freqs, 0, 0)
        encoder.setBuffer_offset_atIndex_(buf_phases, 0, 1)
        encoder.setBuffer_offset_atIndex_(buf_weights, 0, 2)
        encoder.setBuffer_offset_atIndex_(buf_output, 0, 3)
        encoder.setBuffer_offset_atIndex_(buf_params, 0, 4)

        max_threads = pipeline.maxTotalThreadsPerThreadgroup()
        grid_size = _MetalFramework.MTLSizeMake(n_samples, 1, 1)
        group_size = _MetalFramework.MTLSizeMake(min(max_threads, 256), 1, 1)
        encoder.dispatchThreads_threadsPerThreadgroup_(grid_size, group_size)
        encoder.endEncoding()
        cmd_buf.commit()
        cmd_buf.waitUntilCompleted()

        ptr = buf_output.contents()
        result = np.frombuffer(
            (ctypes.c_char * f32_output.nbytes).from_address(int(ptr)),
            dtype=np.float32,
        ).copy().astype(np.float64)
        return result
    except Exception:
        # Fallback on any Metal error
        t = np.arange(n_samples, dtype=np.float64) / sample_rate
        signal = np.zeros(n_samples, dtype=np.float64)
        for i in range(len(freqs)):
            signal += weights[i] * np.sin(2.0 * np.pi * freqs[i] * t + phases[i])
        return signal


def metal_envelope_multiply(signal: np.ndarray, envelope: np.ndarray) -> np.ndarray:
    """GPU-accelerated element-wise envelope multiplication via Apple Metal.

    Falls back to NumPy CPU multiply if Metal is unavailable.
    """
    if not METAL_AVAILABLE or len(signal) != len(envelope):
        return signal * envelope

    try:
        f32_sig = signal.astype(np.float32)
        f32_env = envelope.astype(np.float32)
        buf_sig = _metal_device.newBufferWithBytes_length_options_(
            f32_sig.tobytes(), f32_sig.nbytes, 0)
        buf_env = _metal_device.newBufferWithBytes_length_options_(
            f32_env.tobytes(), f32_env.nbytes, 0)
        fn = _metal_lib.newFunctionWithName_("envelope_multiply")
        pipeline = _metal_device.newComputePipelineStateWithFunction_error_(fn, None)
        if isinstance(pipeline, tuple):
            pipeline = pipeline[0]
        cmd_buf = _metal_queue.commandBuffer()
        encoder = cmd_buf.computeCommandEncoder()
        encoder.setComputePipelineState_(pipeline)
        encoder.setBuffer_offset_atIndex_(buf_sig, 0, 0)
        encoder.setBuffer_offset_atIndex_(buf_env, 0, 1)
        max_threads = pipeline.maxTotalThreadsPerThreadgroup()
        grid_size = _MetalFramework.MTLSizeMake(len(signal), 1, 1)
        group_size = _MetalFramework.MTLSizeMake(min(max_threads, 256), 1, 1)
        encoder.dispatchThreads_threadsPerThreadgroup_(grid_size, group_size)
        encoder.endEncoding()
        cmd_buf.commit()
        cmd_buf.waitUntilCompleted()
        ptr = buf_sig.contents()
        result = np.frombuffer(
            (ctypes.c_char * f32_sig.nbytes).from_address(int(ptr)),
            dtype=np.float32,
        ).copy().astype(np.float64)
        return result
    except Exception:
        return signal * envelope
