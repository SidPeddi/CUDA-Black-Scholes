#include <cmath>
#include <vector>
#include <chrono>
#include <fstream>
#include <cuda_runtime.h>
#include "utils.h"

__device__ float cnd(float x) {
    return 0.5f * erfcf(-x * M_SQRT1_2);
}

__global__ void blackScholesKernel(float* d_call, float* d_put, float* d_stock, float* d_strike, float* d_time, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        float S = d_stock[i];
        float K = d_strike[i];
        float T = d_time[i];
        float r = 0.05f;
        float sigma = 0.2f;

        float d1 = (logf(S/K) + (r + 0.5f*sigma*sigma)*T) / (sigma*sqrtf(T));
        float d2 = d1 - sigma*sqrtf(T);
        d_call[i] = S * cnd(d1) - K * expf(-r*T) * cnd(d2);
        d_put[i] = K * expf(-r*T) * cnd(-d2) - S * cnd(-d1);
    }
}

void runBlackScholesCUDA() {
    int N = 1 << 20;
    std::vector<float> stock(N, 100.0f), strike(N, 100.0f), time(N, 1.0f);
    std::vector<float> call(N), put(N);

    float *d_stock, *d_strike, *d_time, *d_call, *d_put;
    cudaMalloc(&d_stock, N*sizeof(float));
    cudaMalloc(&d_strike, N*sizeof(float));
    cudaMalloc(&d_time, N*sizeof(float));
    cudaMalloc(&d_call, N*sizeof(float));
    cudaMalloc(&d_put, N*sizeof(float));

    cudaMemcpy(d_stock, stock.data(), N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_strike, strike.data(), N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_time, time.data(), N*sizeof(float), cudaMemcpyHostToDevice);

    auto start = std::chrono::high_resolution_clock::now();
    blackScholesKernel<<<(N + 255)/256, 256>>>(d_call, d_put, d_stock, d_strike, d_time, N);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    float duration = std::chrono::duration<float, std::milli>(end - start).count();

    cudaMemcpy(call.data(), d_call, N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(put.data(), d_put, N*sizeof(float), cudaMemcpyDeviceToHost);

    std::ofstream out("data/benchmark_results.csv", std::ios::app);
    out << duration << "\n";
    out.close();

    cudaFree(d_stock); cudaFree(d_strike); cudaFree(d_time);
    cudaFree(d_call); cudaFree(d_put);
}
