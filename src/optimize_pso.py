#!/usr/bin/env python3
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyswarms as ps
import cec2017.functions as functions

DIMENSION = 10
N_RUNS = 10
N_PARTICLES = 10
MAX_ITERS = 1000
DOMAIN = (-100, 100)
W = 0.9
OUTPUT_CSV = "results/optimization_pso.csv"
OUTPUT_TIME = "results/optimization_pso_time.csv"
OUTPUT_PLOT = "results/optimization_pso.png"

GLOBAL_SEED = 42
FIXED_SEEDS = np.random.RandomState(GLOBAL_SEED).randint(0, 10000, size=N_RUNS).tolist()

def main():
    os.makedirs("results", exist_ok=True)
    
    results = {f"f{i}": [] for i in range(1, 29)}
    timings = {f"f{i}_time": [] for i in range(1, 29)}
    
    for i in range(1, 29):
        if i == 2:
            results["f2"] = [np.nan] * N_RUNS
            timings["f2_time"] = [np.nan] * N_RUNS
            continue

        func = getattr(functions, f"f{i}")
        objective = lambda X, f=func: f(X)
        bounds = (np.array([DOMAIN[0]] * DIMENSION), np.array([DOMAIN[1]] * DIMENSION))

        print(f"optimization PSO f{i}")
        for run in range(N_RUNS):
            np.random.seed(FIXED_SEEDS[run])
            c1 = np.round(0.5 * np.random.random() + 0.25, 2)
            c2 = np.round(0.3 * np.random.random() + 0.1, 2)
            options = {'c1': c1, 'c2': c2, 'w': W}
            
            start = time.perf_counter()
            optimizer = ps.single.GlobalBestPSO(
                n_particles=N_PARTICLES,
                dimensions=DIMENSION,
                options=options,
                bounds=bounds
            )
            cost, _ = optimizer.optimize(objective, iters=MAX_ITERS, verbose=False)
            elapsed = time.perf_counter() - start
            
            results[f"f{i}"].append(cost)
            timings[f"f{i}_time"].append(elapsed)
            print(f"  [{run+1}/{N_RUNS}] f_min={cost:.4e}, t={elapsed:.2f}s")

    pd.DataFrame(results, index=[f"run_{r+1}" for r in range(N_RUNS)]).to_csv(OUTPUT_CSV, float_format="%.6e")
    pd.DataFrame(timings, index=[f"run_{r+1}" for r in range(N_RUNS)]).to_csv(OUTPUT_TIME, float_format="%.3f")
    
    # Боксплоты
    df = pd.DataFrame(results, index=range(N_RUNS)).drop(columns=["f2"], errors="ignore")
    plt.figure(figsize=(20, 6))
    df.boxplot(patch_artist=True, widths=0.6)
    plt.yscale("log")
    plt.title("f1–f28: f_min (PSO, 10 runs, D=10)")
    plt.xlabel("Функция"); plt.ylabel("f_min"); plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=200, bbox_inches="tight")
    plt.close()
    
    df = pd.DataFrame(timings, index=range(N_RUNS)).drop(columns=["f2"], errors="ignore")
    plt.figure(figsize=(20, 6))
    df.boxplot(patch_artist=True, widths=0.6)
    plt.title("PSO execution time (seconds)")
    plt.xlabel("Функция"); plt.ylabel("Time (s)"); plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT.replace(".png", "_timings.png"), dpi=200, bbox_inches="tight")
    plt.close()
    

if __name__ == "__main__":
    main()