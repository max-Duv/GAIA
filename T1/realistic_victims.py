"""
Realistic AI workload victims for Project GAIA experiments.

Provides iterative training and inference workloads alongside
the existing synthetic matmul microbenchmark, so T1 can measure
attack effects on workloads the proposal specifically discusses.

Each victim is a callable that runs one iteration and returns a
dict with timing fields matching the assessor interface.
"""

import time
import os
import sys

_TORCH_AVAILABLE = False
_TORCH_CHECKED = False


def _ensure_torch():
    global _TORCH_AVAILABLE, _TORCH_CHECKED
    if _TORCH_CHECKED:
        return _TORCH_AVAILABLE
    _TORCH_CHECKED = True
    try:
        import torch
        if torch.cuda.is_available():
            _TORCH_AVAILABLE = True
        else:
            print("[victims] PyTorch available but CUDA not accessible")
    except ImportError:
        print("[victims] PyTorch not installed")
    return _TORCH_AVAILABLE


class SyntheticMatmulVictim:
    """Existing matmul microbenchmark, wrapped for the unified interface."""

    def __init__(self, device_idx=0, matrix_size=2048):
        import torch
        self.device = torch.device(f"cuda:{device_idx}")
        self.matrix_size = matrix_size
        self.A = torch.randn(matrix_size, matrix_size, device=self.device)
        self.B = torch.randn(matrix_size, matrix_size, device=self.device)
        self._start_event = torch.cuda.Event(enable_timing=True)
        self._end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(self.device)

    def __call__(self):
        import torch
        t0 = time.perf_counter()
        self._start_event.record()
        C = torch.matmul(self.A, self.B)
        self._end_event.record()
        torch.cuda.synchronize(self.device)
        t1 = time.perf_counter()
        gpu_ms = self._start_event.elapsed_time(self._end_event)
        total_ms = (t1 - t0) * 1000.0
        return {
            "total_latency_ms": total_ms,
            "gpu_compute_ms": gpu_ms,
            "launch_delay_ms": max(total_ms - gpu_ms, 0.0),
            "victim_backend": "synthetic_matmul",
        }


class TrainingVictim:
    """Small CNN training loop on random data (one batch per call).

    Uses a simple ConvNet rather than a full ResNet to keep batch times
    short enough for 15-second experiment phases while still exercising
    forward + backward + optimizer step — the workload pattern the
    proposal discusses.
    """

    def __init__(self, device_idx=0, batch_size=32, num_classes=10,
                 input_channels=3, input_size=32):
        import torch
        import torch.nn as nn

        self.device = torch.device(f"cuda:{device_idx}")
        self.batch_size = batch_size

        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        ).to(self.device)

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = nn.CrossEntropyLoss()
        self.input_size = input_size
        self.input_channels = input_channels
        self.num_classes = num_classes

        self._start_event = torch.cuda.Event(enable_timing=True)
        self._end_event = torch.cuda.Event(enable_timing=True)

        self.model.train()
        self._warmup()

    def _warmup(self):
        import torch
        x = torch.randn(self.batch_size, self.input_channels,
                         self.input_size, self.input_size, device=self.device)
        y = torch.randint(0, self.num_classes, (self.batch_size,),
                          device=self.device)
        out = self.model(x)
        loss = self.criterion(out, y)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        import torch.cuda
        torch.cuda.synchronize(self.device)

    def __call__(self):
        import torch
        x = torch.randn(self.batch_size, self.input_channels,
                         self.input_size, self.input_size, device=self.device)
        y = torch.randint(0, self.num_classes, (self.batch_size,),
                          device=self.device)

        t0 = time.perf_counter()
        self._start_event.record()

        self.optimizer.zero_grad()
        out = self.model(x)
        loss = self.criterion(out, y)
        loss.backward()
        self.optimizer.step()

        self._end_event.record()
        torch.cuda.synchronize(self.device)
        t1 = time.perf_counter()

        gpu_ms = self._start_event.elapsed_time(self._end_event)
        total_ms = (t1 - t0) * 1000.0

        return {
            "total_latency_ms": total_ms,
            "gpu_compute_ms": gpu_ms,
            "launch_delay_ms": max(total_ms - gpu_ms, 0.0),
            "victim_backend": "training_cnn",
        }


class InferenceVictim:
    """Model inference loop (one batch per call).

    Uses the same model architecture as TrainingVictim for consistency,
    but runs in eval mode with torch.no_grad().
    """

    def __init__(self, device_idx=0, batch_size=32, num_classes=10,
                 input_channels=3, input_size=32):
        import torch
        import torch.nn as nn

        self.device = torch.device(f"cuda:{device_idx}")
        self.batch_size = batch_size

        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        ).to(self.device)

        self.input_size = input_size
        self.input_channels = input_channels

        self._start_event = torch.cuda.Event(enable_timing=True)
        self._end_event = torch.cuda.Event(enable_timing=True)

        self.model.eval()
        self._warmup()

    def _warmup(self):
        import torch
        with torch.no_grad():
            x = torch.randn(self.batch_size, self.input_channels,
                             self.input_size, self.input_size,
                             device=self.device)
            self.model(x)
        torch.cuda.synchronize(self.device)

    def __call__(self):
        import torch
        x = torch.randn(self.batch_size, self.input_channels,
                         self.input_size, self.input_size, device=self.device)

        t0 = time.perf_counter()
        self._start_event.record()
        with torch.no_grad():
            out = self.model(x)
        self._end_event.record()
        torch.cuda.synchronize(self.device)
        t1 = time.perf_counter()

        gpu_ms = self._start_event.elapsed_time(self._end_event)
        total_ms = (t1 - t0) * 1000.0

        return {
            "total_latency_ms": total_ms,
            "gpu_compute_ms": gpu_ms,
            "launch_delay_ms": max(total_ms - gpu_ms, 0.0),
            "victim_backend": "inference_cnn",
        }


class LegitimateHighPowerWorkload:
    """A legitimate GPU-heavy workload that draws high power.

    Runs large matrix multiplications (simulating dense-layer computation
    in a real model). This is the hard-negative case for the power-level
    detector — it draws high power legitimately.
    """

    def __init__(self, device_idx=0, matrix_size=4096):
        import torch
        self.device = torch.device(f"cuda:{device_idx}")
        self.matrix_size = matrix_size
        self.A = torch.randn(matrix_size, matrix_size, device=self.device)
        self.B = torch.randn(matrix_size, matrix_size, device=self.device)
        self._start_event = torch.cuda.Event(enable_timing=True)
        self._end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(self.device)

    def __call__(self):
        import torch
        t0 = time.perf_counter()
        self._start_event.record()
        C = torch.matmul(self.A, self.B)
        self._end_event.record()
        torch.cuda.synchronize(self.device)
        t1 = time.perf_counter()
        gpu_ms = self._start_event.elapsed_time(self._end_event)
        total_ms = (t1 - t0) * 1000.0
        return {
            "total_latency_ms": total_ms,
            "gpu_compute_ms": gpu_ms,
            "launch_delay_ms": max(total_ms - gpu_ms, 0.0),
            "victim_backend": "legitimate_highpower",
        }


VICTIM_TYPES = {
    "synthetic": SyntheticMatmulVictim,
    "training": TrainingVictim,
    "inference": InferenceVictim,
}


def create_victim(victim_type, device_idx=0, **kwargs):
    if not _ensure_torch():
        raise RuntimeError(
            "PyTorch with CUDA required for realistic victims. "
            "On Jetson, install from NVIDIA JetPack wheels."
        )
    cls = VICTIM_TYPES.get(victim_type)
    if cls is None:
        raise ValueError(
            f"Unknown victim type '{victim_type}'. "
            f"Available: {list(VICTIM_TYPES.keys())}"
        )
    return cls(device_idx=device_idx, **kwargs)


def create_legitimate_workload(device_idx=0, **kwargs):
    if not _ensure_torch():
        raise RuntimeError("PyTorch with CUDA required.")
    return LegitimateHighPowerWorkload(device_idx=device_idx, **kwargs)
