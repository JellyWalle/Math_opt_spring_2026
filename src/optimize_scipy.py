#!/usr/bin/env python3
import os
import numpy as np
from scipy.optimize import minimize
import cec2017.functions as functions
import pandas as pd

DIMENSION = 10 # размерность аргумента
N_RUNS = 10 # кол-во запусков
DOMAIN = (-100, 100) # границы
METHOD = "L-BFGS-B" # метод
MAX_ITER = 1000 # ограничение для оптимизатора


def main():
    results = {f"f{i}": [] for i in range(1, 29)}

    for i in range(1, 29):
        if i == 2:
            # f2 исключён из спецификации CEC2017
            results["f2"] = [np.nan] * N_RUNS
            continue

        func = getattr(functions, f"f{i}")
        # scipy подаёт 1D-вектор, CEC ожидает (N, D)
        objective = lambda x, f=func: float(f(x.reshape(1, -1))[0])
        bounds = [DOMAIN] * DIMENSION

        print(f"оптимизация f{i}")
        for run in range(N_RUNS):
            x0 = np.random.uniform(DOMAIN[0], DOMAIN[1], size=DIMENSION)
            res = minimize(
                objective,
                x0,
                method=METHOD,
                bounds=bounds,
                options={"maxiter": MAX_ITER}
            )
            results[f"f{i}"].append(res.fun)

    df = pd.DataFrame(results, index=[f"run_{r+1}" for r in range(N_RUNS)])

    os.makedirs("results", exist_ok=True)
    output_path = "results/optimization_l_bfgs_b.csv"
    df.to_csv(output_path, float_format="%.6e")

if __name__ == "__main__":
    main()