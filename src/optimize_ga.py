#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pygad
import time

import cec2017.functions as functions

DIMENSION = 10
N_RUNS = 10
N_GENERATIONS = 1000
N_PARENTS = 5
POPULATION_SIZE = 20
DOMAIN = (-100, 100)
MUTATION_PERCENT_GENES = 10
OUTPUT_CSV = "results/optimization_ga.csv"
OUTPUT_TIME = "results/optimization_ga_time.csv"
OUTPUT_PLOT = "results/optimization_ga.png"
GLOBAL_SEED = 42
FIXED_SEEDS = np.random.RandomState(GLOBAL_SEED).randint(0, 10000, size=N_RUNS).tolist()

def make_fitness(cec_func):
    """Возвращает fitness-функцию с 3 параметрами"""
    def fitness(ga, solution, solution_idx):
        # solution: (dim,), CEC ожидает (N, dim)
        return -float(cec_func(solution.reshape(1, -1))[0])
    return fitness

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
        fitness_func = make_fitness(func)
        print(f"optimization GA f{i}")
        
        for run in range(N_RUNS):
            start = time.perf_counter()
            
            ga_instance = pygad.GA(
                num_generations=N_GENERATIONS,
                num_parents_mating=N_PARENTS,
                fitness_func=fitness_func,
                sol_per_pop=POPULATION_SIZE,
                num_genes=DIMENSION,
                gene_type=float,
                gene_space=[DOMAIN[0], DOMAIN[1]],
                mutation_percent_genes=MUTATION_PERCENT_GENES,
                parent_selection_type="sss",
                crossover_type="single_point",
                mutation_type="random",
                keep_elitism=1,
                random_seed=FIXED_SEEDS[run]
            )
            ga_instance.run()
            
            elapsed = time.perf_counter() - start
            _, best_fitness, _ = ga_instance.best_solution()
            f_min = -best_fitness
            
            results[f"f{i}"].append(f_min)
            timings[f"f{i}_time"].append(elapsed)
            print(f"  [{run+1}/{N_RUNS}] f_min={f_min:.4e}, t={elapsed:.2f}s")

    pd.DataFrame(results, index=[f"run_{r+1}" for r in range(N_RUNS)]).to_csv(OUTPUT_CSV, float_format="%.6e")
    pd.DataFrame(timings, index=[f"run_{r+1}" for r in range(N_RUNS)]).to_csv(OUTPUT_TIME, float_format="%.3f")
    
    df = pd.DataFrame(results, index=range(N_RUNS)).drop(columns=["f2"], errors="ignore")
    plt.figure(figsize=(20, 6))
    df.boxplot(patch_artist=True, widths=0.6)
    plt.yscale("log")
    plt.title("f1–f28: f_min (GA, 10 runs, D=10)")
    plt.xlabel("Функция")
    plt.ylabel("f_min")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=200, bbox_inches="tight")
    plt.close()
    
    df = pd.DataFrame(timings, index=range(N_RUNS)).drop(columns=["f2_time"], errors="ignore")
    
    plt.figure(figsize=(20, 6))
    df.boxplot(patch_artist=True, widths=0.6)
    plt.title("GA execution time (seconds)")
    plt.xlabel("Функция")
    plt.ylabel("Time (s)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT.replace(".png", "_timings.png"), dpi=200, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    main()