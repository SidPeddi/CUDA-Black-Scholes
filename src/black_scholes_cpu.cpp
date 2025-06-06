#include <cmath>
#include <vector>
#include <chrono>
#include <fstream>
#include "utils.h"

float cnd(float x) {
    return 0.5f * erfcf(-x * M_SQRT1_2);
}

void runBlackScholesCPU() {
    int N = 1 << 20;
    std::vector<float> stock(N, 100.0f), strike(N, 100.0f), time(N, 1.0f);
    std::vector<float> call(N), put(N);
    float r = 0.05f, sigma = 0.2f;

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        float S = stock[i], K = strike[i], T = time[i];
        float d1 = (logf(S/K) + (r + 0.5f*sigma*sigma)*T) / (sigma*sqrtf(T));
        float d2 = d1 - sigma*sqrtf(T);
        call[i] = S * cnd(d1) - K * expf(-r*T) * cnd(d2);
        put[i] = K * expf(-r*T) * cnd(-d2) - S * cnd(-d1);
    }
    auto end = std::chrono::high_resolution_clock::now();
    float duration = std::chrono::duration<float, std::milli>(end - start).count();

    std::ofstream out("data/benchmark_results.csv");
    out << "N,CPU,CUDA\n";
    out << N << "," << duration << ",";
    out.close();
}
