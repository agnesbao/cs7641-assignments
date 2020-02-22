# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 22:16:27 2020

@author: Xiaojun
"""

import os
import mlrose
import pandas as pd
import matplotlib.pyplot as plt


ALGO_DICT = {
    "RHC": mlrose.random_hill_climb,
    "SA": mlrose.simulated_annealing,
    "GA": mlrose.genetic_alg,
    "MIMIC": mlrose.mimic,
}
MAX_ITER = 100


class SimpleProblem:
    def __init__(self, problem):
        self.problem = problem
        self.fitness_name = problem.fitness_fn.__class__.__name__

    def run_fitness_curve(self):
        datadir = os.path.join("data", f"{self.fitness_name.lower()}_curve.csv")

        res_list = []
        for key, algo in ALGO_DICT.items():
            _, _, curve = algo(
                self.problem, max_iters=MAX_ITER, curve=True, random_state=42
            )
            res_list.append(pd.DataFrame(data=curve, columns=[key]))
        res_df = pd.concat(res_list, axis=1)
        res_df.to_csv(datadir)

        res_df.plot()
        plt.xlabel("Iteration")
        plt.ylabel("Fitness")
        plt.title(f"{self.fitness_name}: fitness curve for different algorithms")
        plt.savefig(f"{self.fitness_name}_algo.png")
        plt.clf()
