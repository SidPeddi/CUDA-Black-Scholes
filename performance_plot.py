import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/benchmark_results.csv')
plt.plot(df['N'], df['CPU'], label='CPU')
plt.plot(df['N'], df['CUDA'], label='CUDA')
plt.xlabel('Number of Options')
plt.ylabel('Execution Time (ms)')
plt.title('Black-Scholes CPU vs CUDA')
plt.legend()
plt.grid(True)
plt.savefig('plots/performance.png')
