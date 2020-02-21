# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 22:43:43 2020

@author: Xiaojun
"""

import mlrose
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LENGTH = 100
MAX_ITER = 100

fitness = mlrose.OneMax()
problem = mlrose.DiscreteOpt(LENGTH, fitness)

def int_to_vec(x):
    bitstring = "{0:b}".format(x)
    return [int(s) for s in bitstring]

def plot_problem(length):
    y = []
    for x in range(2**length):
        y.append(fitness.evaluate(int_to_vec(x)))
    plt.plot(y)
    plt.xlabel("Integer representation of bitstring")
    plt.ylabel("Count of Ones")
    plt.title("OneMax Problem")
    plt.savefig("onemax_demo.png")
    plt.clf()
    
plot_problem(7)

_, _, rhc = mlrose.random_hill_climb(problem, max_iters=MAX_ITER, curve=True, random_state=42)
_, _, sa = mlrose.simulated_annealing(problem, max_iters=MAX_ITER, curve=True, random_state=42)
_, _, ga = mlrose.genetic_alg(problem, max_iters=MAX_ITER, curve=True, random_state=42)
_, _, mimic = mlrose.mimic(problem, max_iters=MAX_ITER, curve=True, random_state=42)

res_df = pd.DataFrame(data=np.array([rhc, sa, ga, mimic]).T, 
                      columns=["RHC", "SA", "GA", "MIMIC"])
res_df.plot()