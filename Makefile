all:
	nvcc -o black_scholes src/black_scholes_cuda.cu src/black_scholes_cpu.cpp src/main.cpp -std=c++14
