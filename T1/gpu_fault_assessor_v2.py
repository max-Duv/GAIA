import os
import re
import time
import glob
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

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

    def __init__(self, target_gpu_index=0):
        self.gpu_idx = target_gpu_index
        self.ina3221_nodes = self._discover_jetson_ina3221_nodes()

        self._nvml_handle = None
        self._has_pynvml = False
        if _PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_idx)
                self._has_pynvml = True
            except Exception:
                self._nvml_handle = None
                self._has_pynvml = False

        self._jtop = None
        self._jtop_ok = False
        if _JTOP_AVAILABLE:
            try:
                self._jtop = JTop()
                self._jtop.start()
                self._jtop_ok = True
            except Exception:
                self._jtop = None

        self._victim_binary = os.path.join(SCRIPT_DIR, "gpu_victim_kernel")
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

        try:
            handle = self._nvml_handle

            power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
            p_w = power_mw / 1000.0

            sm_clk = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
            gr_clk = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
            mem_clk = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)

            pstate = pynvml.nvmlDeviceGetPerformanceState(handle)

            throttle_bitmask_raw = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle)
            sw_cap = int(bool(throttle_bitmask_raw & 0x0000000000000004))  # SwPowerCap
            hw_slow = int(bool(throttle_bitmask_raw & 0x0000000000000008))  # HwSlowdown

            bitmask = (sw_cap << 0) | (hw_slow << 1)

            return {
                "power_w": p_w,
                "sm_clock_mhz": sm_clk,
                "gr_clock_mhz": gr_clk,
                "mem_clock_mhz": mem_clk,
                "pstate": f"P{pstate}",
                "sw_power_cap": sw_cap,
                "hw_slowdown": hw_slow,
                "throttle_bitmask": bitmask,
                "throttle_bitmask_raw": throttle_bitmask_raw,
                "provenance": "NVML_Hardware",
                "voltage_measurement_type": "derived_parasitic_model"
            }
        except pynvml.NVMLError as e:
            raise RuntimeError(f"NVML query failed: {e}")

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
                raise RuntimeError(
                    "INA3221 power rails available but NVML is unreachable — "
                    "cannot read clock/throttle state. Ensure pynvml is installed and the GPU is visible."
                )
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
        try:
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader,nounits"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            if res.returncode == 0:
                cap = res.stdout.strip().split("\n")[0].strip()
                return f"sm_{cap.replace('.', '')}"
        except FileNotFoundError:
            pass
        raise RuntimeError("Cannot detect GPU compute capability for victim kernel compilation.")

    def _compile_victim_kernel(self):
        """Compiles gpu_victim_kernel.cu with auto-detected architecture."""
        src = os.path.join(SCRIPT_DIR, "gpu_victim_kernel.cu")
        if not os.path.exists(src):
            raise FileNotFoundError(f"Victim kernel source not found: {src}")

        arch = self._detect_gpu_arch()
        cmd = ["nvcc", "-O3", f"-arch={arch}", src, "-o", self._victim_binary]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"Victim kernel compilation failed:\n{res.stderr}")

        self._victim_compiled = True

    def _run_compiled_victim(self, elements=16777216, inner_iters=64):
        """
        Runs the compiled gpu_victim_kernel binary and parses CUDA-event latency.
        Returns pure GPU-side timing with no Python dispatch overhead.
        """
        if not self._victim_compiled:
            self._compile_victim_kernel()

        cmd = [
            self._victim_binary,
            "--reps", "1",
            "--elements", str(elements),
            "--inner-iters", str(inner_iters)
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
        if res.returncode != 0:
            raise RuntimeError(f"Victim kernel execution failed:\n{res.stderr}")

        match = re.search(r"VICTIM_LATENCY_MS:\s*([\d.]+)", res.stdout)
        if not match:
            raise RuntimeError(f"Could not parse victim kernel output:\n{res.stdout}")

        gpu_ms = float(match.group(1))
        return {
            "total_latency_ms": gpu_ms,
            "gpu_compute_ms": gpu_ms,
            "launch_delay_ms": 0.0,
            "checksum": 0.0,
            "victim_backend": "compiled_cuda_kernel"
        }

    def run_victim_workload_iteration(self, matrix_size=2048):
        """
        Executes a single victim workload iteration on GPU.
        Prefers compiled CUDA victim kernel for clean GPU-side timing.
        Falls back to PyTorch matmul if compilation is unavailable.
        Raises RuntimeError if neither path works.
        """
        try:
            result = self._run_compiled_victim()
            return result
        except (FileNotFoundError, RuntimeError) as e:
            pass

        try:
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
        except Exception:
            pass

        raise RuntimeError(
            "No victim workload backend available. "
            "Requires either compiled gpu_victim_kernel (nvcc + CUDA) or PyTorch with CUDA support."
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
