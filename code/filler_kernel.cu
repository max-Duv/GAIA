#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <string>
#include <chrono>
#include <thread>
#include <csignal>
#include <cstdlib>
#include <cstring>

// ============================================================================
// PROJECT GAIA — Filler Kernel Defense
// A lightweight GPU kernel that occupies idle SM capacity to dampen
// adversarial power spikes by reducing the attacker's available resources.
// ============================================================================

#define CUDA_CHECK(call) do { \
    cudaError_t _err = (call); \
    if (_err != cudaSuccess) { \
        std::cerr << "[-] CUDA error at " << __FILE__ << ":" << __LINE__ \
                  << " — " << cudaGetErrorString(_err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while (0)

static volatile int g_running = 1;

void signal_handler(int sig) {
    (void)sig;
    g_running = 0;
}

// Filler kernel: lightweight integer arithmetic with controllable duty cycle.
// Each thread does minimal work per active phase, then spin-idles.
// By occupying SM warp slots, it reduces the scheduling capacity available
// to an attacker kernel, dampening sudden power injection.
__global__ void filler_kernel(
    unsigned long long active_cycles,
    unsigned long long idle_cycles,
    volatile int *d_running)
{
    uint32_t val = threadIdx.x + blockIdx.x;

    while (*d_running) {
        // Active phase: lightweight integer ops (low power, occupies warp slots)
        long long int active_start = clock64();
        while ((clock64() - active_start) < (long long int)active_cycles) {
            val = val * 1103515245 + 12345;
            asm volatile("" : "+r"(val) :: "memory");
        }

        // Idle phase: spin-stall (holds warp slot but draws minimal power)
        long long int idle_start = clock64();
        while ((clock64() - idle_start) < (long long int)idle_cycles) {
            asm volatile("" : : : "memory");
        }
    }
}

void print_usage(const char *prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "  --intensity <int>   SM occupancy target 1-100% (default: 25)\n"
              << "  --duration <float>  Run time in seconds, 0=forever (default: 0)\n"
              << "  --duty <float>      Active fraction per cycle 0.01-1.0 (default: 0.5)\n"
              << "  --help              Show this message\n";
}

int main(int argc, char *argv[]) {
    int intensity_pct = 25;
    float duration_s = 0.0f;   // 0 = run until signal
    float duty = 0.5f;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--intensity" && i + 1 < argc) intensity_pct = std::stoi(argv[++i]);
        else if (arg == "--duration" && i + 1 < argc) duration_s = std::stof(argv[++i]);
        else if (arg == "--duty" && i + 1 < argc) duty = std::stof(argv[++i]);
        else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
    }

    if (intensity_pct < 1) intensity_pct = 1;
    if (intensity_pct > 100) intensity_pct = 100;
    if (duty < 0.01f) duty = 0.01f;
    if (duty > 1.0f) duty = 1.0f;

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // Query device properties to scale block count to intensity
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    int max_blocks_per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm, filler_kernel, 128, 0));

    int total_max_blocks = max_blocks_per_sm * prop.multiProcessorCount;
    int target_blocks = std::max(1, (total_max_blocks * intensity_pct) / 100);
    int threads_per_block = 128;  // small footprint per block

    std::cout << "FILLER_START"
              << " intensity=" << intensity_pct << "%"
              << " blocks=" << target_blocks
              << " threads=" << threads_per_block
              << " SMs=" << prop.multiProcessorCount
              << " duty=" << duty
              << " duration=" << (duration_s > 0 ? std::to_string(duration_s) + "s" : "infinite")
              << std::endl;

    // Compute active/idle cycles from duty cycle.
    // Base cycle period ~0.1ms at ~1.5GHz -> 150000 clocks
    unsigned long long cycle_clocks = 150000ULL;
    unsigned long long active_cycles = (unsigned long long)(cycle_clocks * duty);
    unsigned long long idle_cycles = cycle_clocks - active_cycles;

    // Allocate device-side running flag
    int h_running = 1;
    int *d_running;
    CUDA_CHECK(cudaMalloc(&d_running, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_running, &h_running, sizeof(int), cudaMemcpyHostToDevice));

    filler_kernel<<<target_blocks, threads_per_block>>>(
        active_cycles, idle_cycles, d_running);

    auto t_start = std::chrono::steady_clock::now();

    while (g_running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        if (duration_s > 0) {
            auto elapsed = std::chrono::steady_clock::now() - t_start;
            double secs = std::chrono::duration<double>(elapsed).count();
            if (secs >= duration_s) break;
        }
    }

    // Signal kernel to stop
    h_running = 0;
    CUDA_CHECK(cudaMemcpy(d_running, &h_running, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_running));

    auto elapsed = std::chrono::steady_clock::now() - t_start;
    double total_secs = std::chrono::duration<double>(elapsed).count();
    std::cout << "FILLER_DONE duration_s=" << total_secs << std::endl;

    return 0;
}
