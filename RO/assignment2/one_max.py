# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 22:43:43 2020

@author: Xiaojun
"""

from compare_algo import *
import mlrose
import matplotlib.pyplot as plt
import pandas as pd

fitness = mlrose.OneMax()


def int_to_vec(x):
    bitstring = "{0:b}".format(x)
    return [int(s) for s in bitstring]


def plot_problem(length):
    y = []
    for x in range(2 ** length):
        y.append(fitness.evaluate(int_to_vec(x)))
    plt.plot(y)
    plt.xlabel("Integer representation of bitstring")
    plt.ylabel("Count of Ones")
    plt.title("OneMax Problem")
    plt.savefig("onemax_demo.png")
    plt.clf()


plot_problem(7)

problem = mlrose.DiscreteOpt(100, fitness)
onemax = SimpleProblem(problem)
onemax.run_fitness_curve()

#%% problem complexity
datadir = os.path.join("data", "onemax_complexity.csv")

df_list = []
for key, algo in ALGO_DICT.items():
    res_list = []
    length = range(10, 101, 10)
    for l in length:
        problem = mlrose.DiscreteOpt(l, fitness)
        _, best_fitness = algo(problem, random_state=42)
        res_list.append(best_fitness)
    df = pd.DataFrame(data=res_list, columns=[key])
    df_list.append(df)
res_df = pd.concat(df_list, axis=1)
res_df.index = length
res_df.to_csv(datadir)

res_df.plot(style=".-")
plt.xlabel("Length of bitstring")
plt.ylabel("Best fitness")
plt.title("OneMax Problem Complexity")
plt.savefig("onemax_length.png")
plt.clf()
