"""
Project GAIA — Filler Kernel Wrapper
Launches and manages the filler kernel defense subprocess.
"""

import os
import sys
import signal
import subprocess
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_EXE = ".exe" if sys.platform == "win32" else ""
FILLER_BINARY = os.path.join(SCRIPT_DIR, "filler_kernel" + _EXE)


def _cuda_subprocess_env():
    env = os.environ.copy()
    if sys.platform != "win32":
        ld = env.get("LD_LIBRARY_PATH", "")
        if ld:
            cleaned = [p for p in ld.split(":")
                       if "conda" not in p and "miniconda" not in p]
            env["LD_LIBRARY_PATH"] = ":".join(cleaned)
    return env


class FillerKernel:
    """Manages a filler kernel subprocess."""

    def __init__(self, intensity_pct=25, duty=0.5, binary_path=None):
        self.intensity_pct = intensity_pct
        self.duty = duty
        self.binary = binary_path or FILLER_BINARY
        self._proc = None

    def start(self, duration_s=0):
        if self._proc is not None and self._proc.poll() is None:
            return
        if not os.path.exists(self.binary):
            raise FileNotFoundError(
                f"Filler kernel binary not found: {self.binary}\n"
                f"Compile it first: cd {SCRIPT_DIR} && make"
            )
        cmd = [
            self.binary,
            "--intensity", str(self.intensity_pct),
            "--duty", str(self.duty),
            "--duration", str(duration_s),
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=_cuda_subprocess_env(),
        )
        time.sleep(0.3)
        if self._proc.poll() is not None:
            _, stderr = self._proc.communicate(timeout=2)
            raise RuntimeError(f"Filler kernel exited immediately: {stderr}")

    def stop(self, timeout=5):
        if self._proc is None or self._proc.poll() is not None:
            self._proc = None
            return
        self._proc.send_signal(signal.SIGTERM)
        try:
            self._proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
        self._proc = None

    def is_running(self):
        return self._proc is not None and self._proc.poll() is None

    @property
    def pid(self):
        return self._proc.pid if self._proc else None

    def __del__(self):
        self.stop(timeout=2)


def compile_filler_kernel(arch="sm_110"):
    src = os.path.join(SCRIPT_DIR, "filler_kernel.cu")
    out = FILLER_BINARY
    cmd = ["nvcc", "-O3", f"-arch={arch}", src, "-o", out, "-lcuda"]
    print(f"[filler] Compiling: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Compilation failed:\n{res.stderr}")
    print(f"[filler] Compiled: {out}")
    return out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Filler Kernel Wrapper")
    parser.add_argument("--intensity", type=int, default=25)
    parser.add_argument("--duty", type=float, default=0.5)
    parser.add_argument("--duration", type=float, default=30)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--arch", default="sm_110")
    args = parser.parse_args()

    if args.compile:
        compile_filler_kernel(args.arch)
        sys.exit(0)

    fk = FillerKernel(intensity_pct=args.intensity, duty=args.duty)
    fk.start(duration_s=args.duration)
    print(f"[filler] Running (pid={fk.pid}, intensity={args.intensity}%, "
          f"duty={args.duty})")
    try:
        while fk.is_running():
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[filler] Stopping...")
        fk.stop()
    print("[filler] Done.")
