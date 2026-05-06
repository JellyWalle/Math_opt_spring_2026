#!/usr/bin/env python3
import sys
import os
import matplotlib.pyplot as plt
import cec2017.functions as functions
import cec2017.utils as utils

def run(output_folder="plots", points=80, domain=(-100, 100)):
    os.makedirs(output_folder, exist_ok=True)

    for i in range(1, 11):
        func = getattr(functions, f'f{i}')

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        utils.surface_plot(func, domain=domain, points=points, dimension=2, ax=ax)

        file_path = os.path.join(output_folder, f"f{i}.png")
        fig.savefig(file_path, dpi=200, bbox_inches='tight')
        
        plt.close(fig) 


if __name__ == "__main__":
    run(points=80, domain=(-100, 100))