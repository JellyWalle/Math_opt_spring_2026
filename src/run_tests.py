import os
import pandas as pd
import numpy as np
from scipy import stats
import scikit_posthocs as sp

def load_and_aggregate(file_path):
    df = pd.read_csv(file_path, index_col=0)
    # Убираем f2 
    if 'f2' in df.columns:
        df = df.drop(columns=['f2'])
    return df.median(axis=0)

def main():
    files = {
        'L-BFGS-B': 'results/optimization_l_bfgs_b.csv',
        'PSO':      'results/optimization_pso.csv',
        'GA':       'results/optimization_ga.csv'
    }

    aggregated = {method: load_and_aggregate(fname) for method, fname in files.items()}
    # Сортируем функции
    common_funcs = sorted(list(set.intersection(*[set(df.index) for df in aggregated.values()])))

    # Формируем матрицу: строки = функции, столбцы = алгоритмы
    # Размер: (27, 3)
    data_matrix = np.column_stack([aggregated[m][common_funcs].values for m in files.keys()])

    #  Тест Фридмана
    groups = [data_matrix[:, i] for i in range(data_matrix.shape[1])]
    stat, p_friedman = stats.friedmanchisquare(*groups)
    
    print(f"статистика Фридмана: \chi^2 = {stat:.4f}") 
    print(f"p-value: {p_friedman:.6f}")


    if p_friedman < 0.05:
        # Пост-хок тест Неменьи (попарные сравнения всех со всеми)
        # нужен массив размером (n_samples, n_groups) (27, 3)
        p_nemenyi = sp.posthoc_nemenyi_friedman(data_matrix)

        methods = list(files.keys())
        df_nemenyi = p_nemenyi.copy().set_axis(methods, axis=0).set_axis(methods, axis=1)
        
        print(df_nemenyi.round(4))
        
        # Выведем только значимые различия
        found = False
        for i, m1 in enumerate(methods):
            for j, m2 in enumerate(methods):
                if i < j and df_nemenyi.iloc[i, j] < 0.05:
                    print(f"- {m1} vs {m2}: p = {df_nemenyi.iloc[i, j]:.4f}")
                    found = True
        if not found:
            print("Нет значимых попарных различий")
            
        df_nemenyi.to_csv("nemenyi_pvalues.csv", float_format="%.4f")
    else:
        print(" Различия между методами не статистически значимы (p >= 0.05)")

if __name__ == "__main__":
    main()