#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#ifdef USE_NVTX
#include <nvtx3/nvToolsExt.h>
#endif

// ============================================================================
// PROJECT GAIA
// Max Duverneuil
// CYSE 698
// SPIRE LAB 2026
// ============================================================================

#define CUDA_CHECK(call) do { \
    cudaError_t _err = (call); \
    if (_err != cudaSuccess) { \
        std::cerr << "[-] CUDA error at " << __FILE__ << ":" << __LINE__ \
                  << " — " << cudaGetErrorString(_err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// --- KERNEL 1: PTX Cache-Eviction Kernel with Explicit Cache Hints ---
__global__ void modulated_cache_enemy_ptx(
    uint32_t *buf, 
    size_t size, 
    unsigned long long active_cycles, 
    unsigned long long idle_cycles, 
    unsigned long long total_pulses) 
{
    size_t t_id = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = 32;
    size_t buf_elements = size / sizeof(uint32_t);

    for (unsigned long long pulse = 0; pulse < total_pulses; ++pulse) {
        for (unsigned long long k = 0; k < active_cycles; ++k) {
            size_t idx = (t_id * stride + k * 1024) % buf_elements;
            uint32_t *ptr = &buf[idx];
            uint32_t val = (uint32_t)(idx + k);
            asm volatile(
                "st.global.wt.u32 [%0], %1;"
                :
                : "l"(ptr), "r"(val)
                : "memory"
            );
            uint32_t read_back;
            asm volatile(
                "ld.global.cg.u32 %0, [%1];"
                : "=r"(read_back)
                : "l"(ptr)
                : "memory"
            );
        }

        // --- Idle Phase (Microsecond Clock Spin-Stall for sharp di/dt drop) ---
        long long int start_time = clock64();
        while ((clock64() - start_time) < (long long int)idle_cycles) {
            asm volatile("" : : : "memory");
        }
    }
}

// --- KERNEL 2: INT32 Pipeline Saturation Kernel ---
__global__ void modulated_compute_enemy(
    unsigned long long active_cycles, 
    unsigned long long idle_cycles, 
    unsigned long long total_pulses) 
{
    uint32_t val1 = threadIdx.x;
    uint32_t val2 = blockIdx.x;

    for (unsigned long long pulse = 0; pulse < total_pulses; ++pulse) {
        for (unsigned long long k = 0; k < active_cycles; ++k) {
            val1 = (val1 * 1103515245 + 12345) + val2;
            val2 = (val2 * 1103515245 + 12345) + val1;
            asm volatile("" : "+r"(val1), "+r"(val2) :: "memory");
        }

        // --- Idle Phase (Force rapid di/dt transient drop) ---
        long long int start_time = clock64();
        while ((clock64() - start_time) < (long long int)idle_cycles) {
            asm volatile("" : : : "memory");
        }
    }
}

// --- KERNEL 3: L2 Scan Attack — Floods L2 with streaming data (evict_first) ---
__global__ void l2_scan_enemy(
    uint32_t *buf,
    size_t size,
    unsigned long long active_cycles,
    unsigned long long idle_cycles,
    unsigned long long total_pulses)
{
    size_t t_id = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = 32;
    size_t buf_elements = size / sizeof(uint32_t);

    for (unsigned long long pulse = 0; pulse < total_pulses; ++pulse) {
        for (unsigned long long k = 0; k < active_cycles; ++k) {
            size_t idx = (t_id * stride + k * 1024) % buf_elements;
            uint32_t *ptr = &buf[idx];
            uint32_t val = (uint32_t)(idx ^ k);
            asm volatile(
                "st.global.L2::evict_first.b32 [%0], %1;"
                :
                : "l"(ptr), "r"(val)
                : "memory"
            );
            uint32_t read_back;
            asm volatile(
                "ld.global.L2::evict_first.b32 %0, [%1];"
                : "=r"(read_back)
                : "l"(ptr)
                : "memory"
            );
        }

        long long int start_time = clock64();
        while ((clock64() - start_time) < (long long int)idle_cycles) {
            asm volatile("" : : : "memory");
        }
    }
}

// --- KERNEL 4: L2 Pin Attack — Locks cache lines with evict_last ---
__global__ void l2_pin_enemy(
    uint32_t *buf,
    size_t size,
    unsigned long long active_cycles,
    unsigned long long idle_cycles,
    unsigned long long total_pulses)
{
    size_t t_id = blockIdx.x * blockDim.x + threadIdx.x;
    size_t buf_elements = size / sizeof(uint32_t);
    size_t pin_stride = 32;

    for (unsigned long long pulse = 0; pulse < total_pulses; ++pulse) {
        for (unsigned long long k = 0; k < active_cycles; ++k) {
            size_t idx = (t_id * pin_stride + k * pin_stride) % buf_elements;
            uint32_t *ptr = &buf[idx];
            uint32_t val = (uint32_t)(t_id + k);
            asm volatile(
                "st.global.L2::evict_last.b32 [%0], %1;"
                :
                : "l"(ptr), "r"(val)
                : "memory"
            );
        }

        long long int start_time = clock64();
        while ((clock64() - start_time) < (long long int)idle_cycles) {
            asm volatile("" : : : "memory");
        }
    }
}

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  --mode <cache|compute|oversubscribe|all|math_only|memory_only|pin|scan> Attack mode (default: compute)\n"
              << "  --freq <float>       Modulation frequency in kHz (default: 10.0 kHz)\n"
              << "  --duty <float>       Duty cycle of active phase (0.1 to 0.9, default: 0.5)\n"
              << "  --duration <float>   Total test duration in seconds (default: 5.0s)\n"
              << "  --streams <int>      Number of concurrent CUDA streams (default: 8)\n"
              << "  --bufsize <int>      Cache buffer size in MB (default: 12MB)\n";
}

int main(int argc, char* argv[]) {
    std::string mode = "compute";
    float freq_khz = 10.0f;
    float duty_cycle = 0.5f;
    float duration_s = 5.0f;
    int num_streams = 8;
    int buf_size_mb = 12;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--mode" && i + 1 < argc) mode = argv[++i];
        else if (arg == "--freq" && i + 1 < argc) freq_khz = std::stof(argv[++i]);
        else if (arg == "--duty" && i + 1 < argc) duty_cycle = std::stof(argv[++i]);
        else if (arg == "--duration" && i + 1 < argc) duration_s = std::stof(argv[++i]);
        else if (arg == "--streams" && i + 1 < argc) num_streams = std::stoi(argv[++i]);
        else if (arg == "--bufsize" && i + 1 < argc) buf_size_mb = std::stoi(argv[++i]);
        else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
    }


    cudaDeviceProp prop;
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::cout << "Target Hardware GPU: " << prop.name << "\n";
    std::cout << "SM Count: " << prop.multiProcessorCount << "\n";
    int clock_rate_khz = 0;
    cudaDeviceGetAttribute(&clock_rate_khz, cudaDevAttrClockRate, device);
    std::cout << "Peak Clock Rate: " << clock_rate_khz / 1000.0f << " MHz\n\n";

    if ((mode == "pin" || mode == "scan") && prop.major < 8) {
        std::cerr << "[-] Pin/scan modes require sm_80+ (Ampere or later). "
                  << "Detected sm_" << prop.major << prop.minor << "." << std::endl;
        return EXIT_FAILURE;
    }

    double clock_hz = clock_rate_khz * 1e3;
    double period_s = 1.0 / (freq_khz * 1e3);
    unsigned long long active_cycles = (unsigned long long)(period_s * duty_cycle * clock_hz);
    unsigned long long idle_cycles = (unsigned long long)(period_s * (1.0f - duty_cycle) * clock_hz);
    unsigned long long total_pulses = (unsigned long long)(duration_s / period_s);

    if (active_cycles == 0) active_cycles = 1;
    if (idle_cycles == 0) idle_cycles = 1;

    std::cout << "Modulation Configurations:\n"
              << "  - Frequency: " << freq_khz << " kHz (Period: " << period_s * 1e6 << " us)\n"
              << "  - Duty Cycle: " << duty_cycle * 100.0f << "%\n"
              << "  - Active Cycles: " << active_cycles << "\n"
              << "  - Idle Cycles: " << idle_cycles << "\n"
              << "  - Total Pulsing Phases: " << total_pulses << "\n"
              << "  - Attack Mode: " << mode << "\n"
              << "  - Streams: " << num_streams << "\n\n";

    size_t total_buf_bytes = (size_t)buf_size_mb * 1024 * 1024;
    uint32_t *d_buf = nullptr;
    if (mode == "cache" || mode == "oversubscribe" || mode == "all" || mode == "memory_only" || mode == "pin" || mode == "scan") {
        CUDA_CHECK(cudaMalloc(&d_buf, total_buf_bytes));
        std::cout << "Allocated " << buf_size_mb << " MB VRAM buffer for PTX L2 cache-eviction.\n";
    }

    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; ++i) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }

#ifdef USE_NVTX
    nvtxRangePushA("GAIA_Stress_Execution");
#endif

    dim3 threads(256);
    dim3 blocks(prop.multiProcessorCount);

    std::cout << "Executing attack kernels...\n";

    if (mode == "oversubscribe" || mode == "all") {
        for (int i = 0; i < num_streams; ++i) {
            if (i % 2 == 0 && d_buf != nullptr) {
                modulated_cache_enemy_ptx<<<blocks, threads, 0, streams[i]>>>(
                    d_buf, total_buf_bytes, active_cycles, idle_cycles, total_pulses);
            } else {
                modulated_compute_enemy<<<blocks, threads, 0, streams[i]>>>(
                    active_cycles, idle_cycles, total_pulses);
            }
            CUDA_CHECK(cudaGetLastError());
        }
    } else if (mode == "cache" || mode == "memory_only") {
        for (int i = 0; i < num_streams; ++i) {
            modulated_cache_enemy_ptx<<<blocks, threads, 0, streams[i]>>>(
                d_buf, total_buf_bytes, active_cycles, idle_cycles, total_pulses);
            CUDA_CHECK(cudaGetLastError());
        }
    } else if (mode == "scan") {
        for (int i = 0; i < num_streams; ++i) {
            l2_scan_enemy<<<blocks, threads, 0, streams[i]>>>(
                d_buf, total_buf_bytes, active_cycles, idle_cycles, total_pulses);
            CUDA_CHECK(cudaGetLastError());
        }
    } else if (mode == "pin") {
        l2_pin_enemy<<<blocks, threads, 0, streams[0]>>>(
            d_buf, total_buf_bytes, active_cycles, idle_cycles, total_pulses);
        CUDA_CHECK(cudaGetLastError());
    } else { // compute / math_only
        for (int i = 0; i < num_streams; ++i) {
            modulated_compute_enemy<<<blocks, threads, 0, streams[i]>>>(
                active_cycles, idle_cycles, total_pulses);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    for (int i = 0; i < num_streams; ++i) {
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
    }

#ifdef USE_NVTX
    nvtxRangePop();
#endif

    std::cout << "Stress execution finished successfully.\n";

    if (d_buf != nullptr) {
        CUDA_CHECK(cudaFree(d_buf));
    }
    for (int i = 0; i < num_streams; ++i) {
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }

    return 0;
}
