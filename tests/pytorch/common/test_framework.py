"""
PyTorch test framework for FlashCK ops (correctness & performance)
"""
import torch
import time
import numpy as np


class TestTypeTraits:
    traits = {
        torch.float32: dict(rtol=1e-3, atol=1e-4, max_errors=10),
        torch.float16: dict(rtol=2e-2, atol=2e-3, max_errors=10),
        torch.float64: dict(rtol=1e-8, atol=1e-10, max_errors=10),
    }

    @staticmethod
    def get(dtype):
        return TestTypeTraits.traits.get(dtype, TestTypeTraits.traits[torch.float32])


class PerformanceResult:
    def __init__(self, latency, tflops, bandwidth, config_name, operation_type):
        self.latency = latency
        self.tflops = tflops
        self.bandwidth = bandwidth
        self.config_name = config_name
        self.operation_type = operation_type

    def print(self):
        print(f"{self.operation_type} - {self.config_name}:")
        print(f"  Latency: {self.latency:.3f} ms")
        print(f"  TFLOPs: {self.tflops:.3f} TFlops")
        print(f"  Bandwidth: {self.bandwidth:.3f} GB/s\n")


class UnifiedTestSuite:
    def __init__(self, dtype=torch.float32, device='cuda'):
        self.dtype = dtype
        self.device = device
        self.rng = torch.Generator(device=device)

    def _run_correctness(self, configs, reference_impl, flashck_impl, verbose=False):
        print("\n=== CORRECTNESS TESTING ===")
        traits = TestTypeTraits.get(self.dtype)
        rtol = traits['rtol']
        atol = traits['atol']
        for config in configs:
            print(f"Testing: {config['name']} ({config['operation_type']})")
            op_args = config.get('op_args', {})
            ref_out = reference_impl(**op_args)
            flashck_out = flashck_impl(**op_args)
            np.testing.assert_allclose(flashck_out.cpu().numpy(
            ), ref_out.cpu().numpy(), rtol=rtol, atol=atol)
            if verbose:
                print("✓ Passed\n")
        print(f"Correctness testing completed: {len(configs)} configurations")

    def _run_performance(self, configs, flashck_impl, num_runs=10, warmup_runs=3):
        print("\n=== PERFORMANCE BENCHMARKING ===")
        results = []
        for config in configs:
            print(f"Benchmarking: {config['name']}...")
            op_args = config.get('op_args', {})
            # Warmup
            for _ in range(warmup_runs):
                y = flashck_impl(**op_args)
                torch.cuda.synchronize()
            # Timing with CUDA events
            x = op_args.get('x', None)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_event.record()
            for _ in range(num_runs):
                y = flashck_impl(**op_args)
            end_event.record()
            torch.cuda.synchronize()
            elapsed_ms = start_event.elapsed_time(end_event)
            avg_time = elapsed_ms / num_runs if num_runs > 0 else 0
            total_bytes = x.numel() * x.element_size() if x is not None else 0
            m = x.shape[0] if x is not None else 0
            n = x.shape[-1] if x is not None else 0
            bandwidth = (total_bytes / 1e9) / (avg_time /
                                               1000.0) if total_bytes > 0 else 0
            tflops = (7.0 * m * n / 1e12) / (avg_time /
                                             1000.0) if m > 0 and n > 0 else 0
            result = PerformanceResult(
                avg_time, tflops, bandwidth, config['name'], config['operation_type'])
            result.print()
            results.append(result)
        return results

    def run_test(self, configs, reference_impl, flashck_impl, benchmark=False, num_runs=10, warmup_runs=3, verbose=False):
        self._run_correctness(configs, reference_impl,
                              flashck_impl, verbose=verbose)
        if benchmark:
            return self._run_performance(configs, flashck_impl, num_runs=num_runs, warmup_runs=warmup_runs)
        return None
