// Minimal "victim" workload for Project GAIA's fault assessor.
// Runs a fixed amount of deterministic GPU work (FMA-bound + global memory
// read) and prints its GPU-side latency (via CUDA events) per repetition.
// Used to measure REAL contention slowdown when co-located with the
// gpu_power_poc_v2 attack, rather than inferring it indirectly.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

__global__ void victim_kernel(const float *a, const float *b, float *c, size_t n, int inner_iters) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float av = a[idx];
    float bv = b[idx];
    float acc = 0.0f;
    for (int i = 0; i < inner_iters; ++i) {
        acc = fmaf(av, bv, acc);
        av = av * 1.0000001f + 1e-7f;
    }
    c[idx] = acc;
}

int main(int argc, char **argv) {
    size_t n = 16ull * 1024 * 1024; // 16M elements (~64MB per buffer)
    int inner_iters = 64;
    int reps = 1;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--reps" && i + 1 < argc) reps = atoi(argv[++i]);
        else if (arg == "--elements" && i + 1 < argc) n = (size_t)atoll(argv[++i]);
        else if (arg == "--inner-iters" && i + 1 < argc) inner_iters = atoi(argv[++i]);
    }

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_c, n * sizeof(float));
    cudaMemset(d_a, 0x3f, n * sizeof(float)); // arbitrary nonzero float pattern
    cudaMemset(d_b, 0x3f, n * sizeof(float));

    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm-up launch (excluded from timing) to absorb one-time JIT/context costs.
    victim_kernel<<<blocks, threads>>>(d_a, d_b, d_c, n, inner_iters);
    cudaDeviceSynchronize();

    for (int r = 0; r < reps; ++r) {
        cudaEventRecord(start);
        victim_kernel<<<blocks, threads>>>(d_a, d_b, d_c, n, inner_iters);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        printf("VICTIM_LATENCY_MS: %.4f\n", ms);
        fflush(stdout);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}
