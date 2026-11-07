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
    """Matmul microbenchmark. Uses PyTorch if available, else compiled CUDA kernel."""

    def __init__(self, device_idx=0, matrix_size=2048):
        self.matrix_size = matrix_size
        self._use_torch = _ensure_torch()
        if self._use_torch:
            import torch
            self.device = torch.device(f"cuda:{device_idx}")
            self.A = torch.randn(matrix_size, matrix_size, device=self.device)
            self.B = torch.randn(matrix_size, matrix_size, device=self.device)
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(self.device)
        else:
            import subprocess, re
            self._subprocess = subprocess
            self._re = re
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self._victim_binary = os.path.join(script_dir, "gpu_victim_kernel")
            self._compiled = os.path.exists(self._victim_binary)
            if not self._compiled:
                self._compile(script_dir)
            print(f"[victims] Using compiled CUDA victim (no PyTorch)")

    def _compile(self, script_dir):
        src = os.path.join(script_dir, "gpu_victim_kernel.cu")
        if not os.path.exists(src):
            raise FileNotFoundError(f"Victim kernel source not found: {src}")
        arch = self._detect_arch()
        print(f"[victims] Compiling victim kernel ({arch})...")
        cmd = ["nvcc", "-O3", f"-arch={arch}", src, "-o", self._victim_binary, "-lcuda"]
        res = self._subprocess.run(cmd, stdout=self._subprocess.PIPE,
                                   stderr=self._subprocess.PIPE, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"Victim kernel compilation failed:\n{res.stderr}")
        self._compiled = True

    @staticmethod
    def _detect_arch():
        try:
            out = __import__("subprocess").check_output(
                ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                text=True).strip().split("\n")[0].replace(".", "")
            return f"sm_{out}"
        except Exception:
            return "sm_87"

    def __call__(self):
        if self._use_torch:
            return self._call_torch()
        return self._call_cuda()

    def _call_torch(self):
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

    def _call_cuda(self):
        t0 = time.perf_counter()
        cmd = [self._victim_binary, "--reps", "1", "--elements", "16777216",
               "--inner-iters", "64"]
        proc = self._subprocess.Popen(cmd, stdout=self._subprocess.PIPE,
                                      stderr=self._subprocess.PIPE, text=True)
        try:
            stdout, stderr = proc.communicate(timeout=15)
        except self._subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2)
            return {"total_latency_ms": 15000.0, "gpu_compute_ms": 15000.0,
                    "launch_delay_ms": 0.0, "victim_backend": "synthetic_cuda_timeout"}
        t1 = time.perf_counter()
        match = self._re.search(r"VICTIM_LATENCY_MS:\s*([\d.]+)", stdout)
        gpu_ms = float(match.group(1)) if match else (t1 - t0) * 1000.0
        return {
            "total_latency_ms": (t1 - t0) * 1000.0,
            "gpu_compute_ms": gpu_ms,
            "launch_delay_ms": 0.0,
            "victim_backend": "synthetic_cuda_compiled",
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
    cls = VICTIM_TYPES.get(victim_type)
    if cls is None:
        raise ValueError(
            f"Unknown victim type '{victim_type}'. "
            f"Available: {list(VICTIM_TYPES.keys())}"
        )
    if victim_type != "synthetic" and not _ensure_torch():
        raise RuntimeError(
            f"PyTorch with CUDA required for '{victim_type}' victim. "
            "On Jetson, install from NVIDIA JetPack wheels. "
            "Use --victim-type synthetic for CUDA-only fallback."
        )
    return cls(device_idx=device_idx, **kwargs)


def create_legitimate_workload(device_idx=0, **kwargs):
    if not _ensure_torch():
        raise RuntimeError("PyTorch with CUDA required.")
    return LegitimateHighPowerWorkload(device_idx=device_idx, **kwargs)
