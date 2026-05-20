#!/usr/bin/env python3
import os
import numpy as np
from scipy.optimize import minimize
import cec2017.functions as functions
import matplotlib.pyplot as plt
import time
import pandas as pd

DIMENSION = 10 # размерность аргумента
N_RUNS = 10 # кол-во запусков
DOMAIN = (-100, 100) # границы
METHOD = "L-BFGS-B" # метод
MAX_ITER = 1000 # ограничение для оптимизатора

GLOBAL_SEED = 42
FIXED_SEEDS = np.random.RandomState(GLOBAL_SEED).randint(0, 10000, size=N_RUNS).tolist()

def main():
    results = {f"f{i}": [] for i in range(1, 29)}
    timings = {f"f{i}_time": [] for i in range(1, 29)}

    for i in range(1, 29):
        if i == 2:
            # f2 исключён из спецификации CEC2017
            results["f2"] = [np.nan] * N_RUNS
            timings["f2_time"] = [np.nan] * N_RUNS
            continue

        func = getattr(functions, f"f{i}")
        # scipy подаёт 1D-вектор, CEC ожидает (N, D)
        objective = lambda x, f=func: float(f(x.reshape(1, -1))[0])
        bounds = [DOMAIN] * DIMENSION

        print(f"оптимизация f{i}")
        for run in range(N_RUNS):
            np.random.seed(FIXED_SEEDS[run])
            x0 = np.random.uniform(DOMAIN[0], DOMAIN[1], size=DIMENSION)

            start = time.perf_counter()
            res = minimize(
                objective,
                x0,
                method=METHOD,
                bounds=bounds,
                options={"maxiter": MAX_ITER}
            )
            elapsed = time.perf_counter() - start 
            results[f"f{i}"].append(res.fun)
            timings[f"f{i}_time"].append(elapsed)
    
    pd.DataFrame(results, index=[f"run_{r+1}" for r in range(N_RUNS)]).to_csv("results/optimization_l_bfgs_b.csv", float_format="%.6e")
    pd.DataFrame(timings, index=[f"run_{r+1}" for r in range(N_RUNS)]).to_csv("results/optimization_l_bfgs_b_time.csv", float_format="%.3f")
    
    df = pd.DataFrame(results, index=range(N_RUNS)).drop(columns=["f2"], errors="ignore")
    plt.figure(figsize=(20, 6))
    df.boxplot(patch_artist=True, widths=0.6)
    plt.yscale("log")
    plt.title("f1–f28: f_min (L-BFGS-B, 10 runs, D=10)")
    plt.xlabel("Функция")
    plt.ylabel("f_min")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig("results/optimization_l_bfgs_b.png", dpi=200, bbox_inches="tight")
    plt.close()
    
    df = pd.DataFrame(timings, index=range(N_RUNS)).drop(columns=["f2_time"], errors="ignore")
    plt.figure(figsize=(20, 6))
    df.boxplot(patch_artist=True, widths=0.6)
    plt.title("L-BFGS-B execution time (seconds)")
    plt.xlabel("Функция")
    plt.ylabel("Time (s)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("results/optimization_l_bfgs_b_timings.png", dpi=200, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    main()