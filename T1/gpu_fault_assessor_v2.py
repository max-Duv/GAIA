import os
import re
import sys
import time
import glob
import subprocess
import threading

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_EXE = ".exe" if sys.platform == "win32" else ""


def _cuda_subprocess_env():
    """Return an env dict for running compiled CUDA binaries.
    On Linux, strips conda/pip CUDA libraries from LD_LIBRARY_PATH so the
    system's JetPack CUDA runtime is used instead of the desktop one."""
    env = os.environ.copy()
    if sys.platform != "win32":
        ld = env.get("LD_LIBRARY_PATH", "")
        if ld:
            cleaned = [p for p in ld.split(":")
                       if "conda" not in p and "miniconda" not in p]
            env["LD_LIBRARY_PATH"] = ":".join(cleaned)
    return env

try:
    import pynvml
    _PYNVML_AVAILABLE = True
except ImportError:
    _PYNVML_AVAILABLE = False

try:
    from jtop import jtop as JTop
    _JTOP_AVAILABLE = True
except ImportError:
    _JTOP_AVAILABLE = False


class GPUFaultAssessorV2:

    def __init__(self, target_gpu_index=0, arch_override=None):
        self.gpu_idx = target_gpu_index
        self._arch_override = arch_override
        print("[assessor] Discovering INA3221 sensor nodes...")
        self.ina3221_nodes = self._discover_jetson_ina3221_nodes()
        print(f"[assessor] Found {len(self.ina3221_nodes)} INA3221 nodes")

        self._nvml_handle = None
        self._has_pynvml = False
        if _PYNVML_AVAILABLE:
            try:
                print("[assessor] Initializing pynvml...")
                result = [None]
                def _try_nvml():
                    try:
                        pynvml.nvmlInit()
                        result[0] = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_idx)
                    except Exception:
                        pass
                t = threading.Thread(target=_try_nvml, daemon=True)
                t.start()
                t.join(timeout=5)
                if result[0] is not None:
                    self._nvml_handle = result[0]
                    self._has_pynvml = True
                    print("[assessor] pynvml ready")
                else:
                    print("[assessor] pynvml timed out or failed (expected on Jetson)")
            except Exception:
                self._nvml_handle = None
                self._has_pynvml = False

        self._jtop = None
        self._jtop_ok = False
        if _JTOP_AVAILABLE:
            try:
                print("[assessor] Starting jtop...")
                self._jtop = JTop()
                self._jtop.start()
                self._jtop_ok = True
                print("[assessor] jtop ready")
            except Exception:
                self._jtop = None
                print("[assessor] jtop failed")

        self._victim_binary = os.path.join(SCRIPT_DIR, "gpu_victim_kernel" + _EXE)
        self._victim_compiled = os.path.exists(self._victim_binary)

    def _discover_jetson_ina3221_nodes(self):
        nodes = {}
        hwmon_paths = glob.glob("/sys/class/hwmon/hwmon*")

        for h_path in hwmon_paths:
            name_file = os.path.join(h_path, "name")
            drv_name = ""
            if os.path.exists(name_file):
                try:
                    with open(name_file, "r") as f:
                        drv_name = f.read().strip()
                except Exception:
                    pass

            if "ina3221" in drv_name.lower() or "jetson" in drv_name.lower() or "i2c" in h_path:
                for i in range(1, 4):
                    in_file = os.path.join(h_path, f"in{i}_input")
                    curr_file = os.path.join(h_path, f"curr{i}_input")
                    power_file = os.path.join(h_path, f"power{i}_input")

                    if os.path.exists(in_file):
                        nodes[f"rail_{i}_voltage_sysfs"] = in_file
                    if os.path.exists(curr_file):
                        nodes[f"rail_{i}_current_sysfs"] = curr_file
                    if os.path.exists(power_file):
                        nodes[f"rail_{i}_power_sysfs"] = power_file

        return nodes

    def query_jtop_telemetry(self):
        if not self._jtop_ok or self._jtop is None:
            return None
        try:
            power = self._jtop.power
            if power is None:
                return None
            rails = power.get('rail', {})

            gpu_rail = None
            gpu_rail_name = None
            for name in ('VDD_GPU_SOC', 'VDD_CPU_GPU_CV'):
                if name in rails:
                    gpu_rail = rails[name]
                    gpu_rail_name = name
                    break

            if gpu_rail is None:
                return None

            gpu_power_mw = gpu_rail.get('power', 0)
            total_power_mw = power.get('tot', {}).get('power', gpu_power_mw)
            voltage_mv = gpu_rail.get('voltage', 0)
            current_ma = gpu_rail.get('current', 0)

            return {
                'power_w': total_power_mw / 1000.0,
                'gpu_power_w': gpu_power_mw / 1000.0,
                'gpu_rail': gpu_rail_name,
                'voltage_v': voltage_mv / 1000.0 if voltage_mv else 0.0,
                'current_a': current_ma / 1000.0 if current_ma else 0.0,
                'provenance': 'Jetson_jtop',
                'voltage_measurement_type': 'measured',
            }
        except Exception:
            return None

    def query_jetson_ina3221_telemetry(self):
        if not self.ina3221_nodes:
            return None

        telemetry = {}
        total_power_w = 0.0
        bus_voltage_v = 0.0
        total_current_a = 0.0

        try:
            for key, path in self.ina3221_nodes.items():
                with open(path, "r") as f:
                    val = float(f.read().strip())
                    if "voltage" in key:
                        telemetry[key] = val / 1000.0
                        if bus_voltage_v == 0.0:
                            bus_voltage_v = val / 1000.0
                    elif "current" in key:
                        telemetry[key] = val / 1000.0
                        total_current_a += val / 1000.0
                    elif "power" in key:
                        telemetry[key] = val / 1e6
                        total_power_w += telemetry[key]

            telemetry["power_w"] = total_power_w if total_power_w > 0 else (bus_voltage_v * total_current_a)
            telemetry["voltage_v"] = bus_voltage_v if bus_voltage_v > 0 else 5.0
            telemetry["current_a"] = total_current_a
            telemetry["provenance"] = "Jetson_INA3221_Hardware"
            telemetry["voltage_measurement_type"] = "measured"
            return telemetry
        except Exception:
            return None

    def query_nvml_telemetry(self):
        if not self._has_pynvml or self._nvml_handle is None:
            return None

        handle = self._nvml_handle
        result = {
            "power_w": 0.0,
            "sm_clock_mhz": 0,
            "gr_clock_mhz": 0,
            "mem_clock_mhz": 0,
            "pstate": "P0",
            "sw_power_cap": 0,
            "hw_slowdown": 0,
            "throttle_bitmask": 0,
            "throttle_bitmask_raw": 0,
            "provenance": "NVML_Hardware",
            "voltage_measurement_type": "derived_parasitic_model"
        }

        try:
            result["power_w"] = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
        except pynvml.NVMLError:
            pass

        try:
            result["sm_clock_mhz"] = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
            result["gr_clock_mhz"] = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
            result["mem_clock_mhz"] = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
        except pynvml.NVMLError:
            pass

        try:
            result["pstate"] = f"P{pynvml.nvmlDeviceGetPerformanceState(handle)}"
        except pynvml.NVMLError:
            pass

        try:
            raw = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle)
            sw_cap = int(bool(raw & 0x0000000000000004))
            hw_slow = int(bool(raw & 0x0000000000000008))
            result["sw_power_cap"] = sw_cap
            result["hw_slowdown"] = hw_slow
            result["throttle_bitmask"] = (sw_cap << 0) | (hw_slow << 1)
            result["throttle_bitmask_raw"] = raw
        except pynvml.NVMLError:
            pass

        return result

    def get_hardware_telemetry(self):
        """
        Unified telemetry getter: jtop (named rails) -> INA3221 sysfs -> NVML.
        Raises RuntimeError if no hardware telemetry source is available.

        Returns dict with 'voltage_measurement_type' field:
          - "measured" when voltage comes from jtop or INA3221 hardware sensor
          - "derived_parasitic_model" when voltage must be computed from power via PDN model
        """
        jtop_data = self.query_jtop_telemetry()
        ina = self.query_jetson_ina3221_telemetry()
        nvml = self.query_nvml_telemetry()

        if jtop_data and jtop_data.get("power_w", 0) > 0:
            res = jtop_data
            if nvml:
                res["sm_clock_mhz"] = nvml["sm_clock_mhz"]
                res["sw_power_cap"] = nvml["sw_power_cap"]
                res["hw_slowdown"] = nvml["hw_slowdown"]
                res["throttle_bitmask"] = nvml["throttle_bitmask"]
                res["throttle_bitmask_raw"] = nvml["throttle_bitmask_raw"]
            return res

        if ina and ina.get("power_w", 0) > 0:
            res = ina
            if nvml:
                res["sm_clock_mhz"] = nvml["sm_clock_mhz"]
                res["sw_power_cap"] = nvml["sw_power_cap"]
                res["hw_slowdown"] = nvml["hw_slowdown"]
                res["throttle_bitmask"] = nvml["throttle_bitmask"]
                res["throttle_bitmask_raw"] = nvml["throttle_bitmask_raw"]
            else:
                res["sm_clock_mhz"] = 0
                res["sw_power_cap"] = 0
                res["hw_slowdown"] = 0
                res["throttle_bitmask"] = 0
                res["throttle_bitmask_raw"] = 0
            return res

        if nvml and nvml.get("power_w", 0) > 0:
            return nvml

        raise RuntimeError(
            "No hardware telemetry source available. "
            "Requires Jetson jtop (jetson-stats), INA3221 sysfs nodes, or NVML (pynvml). "
            "Verify GPU is present and drivers are loaded."
        )

    def _detect_gpu_arch(self):
        """Returns the sm_XX flag for the current GPU's compute capability."""
        if self._arch_override:
            return self._arch_override

        from controlled_3phase_runner_v2 import detect_gpu_arch
        return detect_gpu_arch()

    def _compile_victim_kernel(self):
        """Compiles gpu_victim_kernel.cu with auto-detected architecture."""
        src = os.path.join(SCRIPT_DIR, "gpu_victim_kernel.cu")
        if not os.path.exists(src):
            raise FileNotFoundError(f"Victim kernel source not found: {src}")

        arch = self._detect_gpu_arch()
        print(f"[victim] Compiling victim kernel ({arch})...")
        cmd = ["nvcc", "-O3", f"-arch={arch}", src, "-o", self._victim_binary, "-lcuda"]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"Victim kernel compilation failed:\n{res.stderr}")

        print(f"[victim] Compiled: {self._victim_binary}")
        self._victim_compiled = True

    def _run_compiled_victim(self, elements=16777216, inner_iters=64):
        if not self._victim_compiled:
            self._compile_victim_kernel()

        cmd = [
            self._victim_binary,
            "--reps", "1",
            "--elements", str(elements),
            "--inner-iters", str(inner_iters)
        ]
        env = _cuda_subprocess_env()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        try:
            stdout, stderr = proc.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
            raise RuntimeError("Victim kernel timed out (15s) — CUDA runtime may be hanging")
        if proc.returncode != 0:
            raise RuntimeError(f"Victim kernel execution failed:\n{stderr}")

        match = re.search(r"VICTIM_LATENCY_MS:\s*([\d.]+)", stdout)
        if not match:
            raise RuntimeError(f"Could not parse victim kernel output:\n{stdout}")

        gpu_ms = float(match.group(1))
        return {
            "total_latency_ms": gpu_ms,
            "gpu_compute_ms": gpu_ms,
            "launch_delay_ms": 0.0,
            "checksum": 0.0,
            "victim_backend": "compiled_cuda_kernel"
        }

    def _run_pytorch_victim(self, matrix_size=2048):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch CUDA is not available.")

        device = torch.device(f"cuda:{self.gpu_idx}")
        A = torch.randn(matrix_size, matrix_size, device=device)
        B = torch.randn(matrix_size, matrix_size, device=device)

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        t_enqueue = time.perf_counter()
        start_event.record()
        C = torch.matmul(A, B)
        end_event.record()

        torch.cuda.synchronize()
        t_finish = time.perf_counter()

        gpu_compute_ms = start_event.elapsed_time(end_event)
        total_latency_ms = (t_finish - t_enqueue) * 1000.0
        launch_delay_ms = max(total_latency_ms - gpu_compute_ms, 0.0)

        return {
            "total_latency_ms": total_latency_ms,
            "launch_delay_ms": launch_delay_ms,
            "gpu_compute_ms": gpu_compute_ms,
            "checksum": float(C[0, 0].cpu()),
            "victim_backend": "pytorch_matmul"
        }

    def run_victim_workload_iteration(self, matrix_size=2048):
        # Try compiled CUDA victim kernel first
        try:
            return self._run_compiled_victim()
        except (FileNotFoundError, RuntimeError) as e:
            if not hasattr(self, '_victim_error_printed'):
                print(f"[victim] Compiled victim failed: {e}")
                print("[victim] Falling back to PyTorch (with 10s timeout)...")
                self._victim_error_printed = True

        # PyTorch fallback with timeout (pip torch hangs on Jetson)
        result = [None]
        error = [None]
        def _try_torch():
            try:
                result[0] = self._run_pytorch_victim(matrix_size)
            except Exception as e:
                error[0] = e
        t = threading.Thread(target=_try_torch, daemon=True)
        t.start()
        t.join(timeout=10)
        if result[0] is not None:
            return result[0]

        if t.is_alive():
            msg = "PyTorch CUDA timed out (10s) — pip-installed torch is incompatible with Jetson."
        elif error[0]:
            msg = f"PyTorch failed: {error[0]}"
        else:
            msg = "PyTorch returned no result"

        raise RuntimeError(
            f"No victim workload backend available. {msg}\n"
            "On Jetson, install torch from NVIDIA's JetPack wheels, not pip."
        )

    def close(self):
        if self._jtop is not None:
            try:
                self._jtop.close()
            except Exception:
                pass
            self._jtop = None
            self._jtop_ok = False


if __name__ == "__main__":
    assessor = GPUFaultAssessorV2()
    telem = assessor.get_hardware_telemetry()
    print("Hardware Telemetry Sample:", telem)
    victim_iter = assessor.run_victim_workload_iteration()
    print("Victim Iteration Decomposed Sample:", victim_iter)
