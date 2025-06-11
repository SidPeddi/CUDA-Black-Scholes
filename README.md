# CUDA Black-Scholes Model

A high-performance CUDA-based implementation of the Black-Scholes model for pricing European call and put options. This project demonstrates GPU acceleration over traditional CPU methods and includes benchmarking and visualization of results.

### Build
```bash
make
```

### Run
```bash
./black_scholes
```

### Plot Benchmark
```bash
python plots/performance_plot.py
```

## Output
The program will generate a `benchmark_results.csv` file and a `performance.png` plot showing the speed comparison between CPU and GPU.


