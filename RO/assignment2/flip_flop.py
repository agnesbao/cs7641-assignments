# -*- coding: utf-8 -*-

import mlrose_hiive as mlrose
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from time import process_time

print("Running FlipFlop...")

fitness = mlrose.FlipFlop()
problem = mlrose.FlipFlopOpt(100, fitness)

RANDOM_SEED = 42
MAX_ATTEMPTS = 100

#%% tuning for SA
curve_list = []
decays = [0.999, 0.99]
for d in decays:
    schedule = mlrose.GeomDecay(decay=d)
    _, _, curve = mlrose.simulated_annealing(
        problem,
        schedule=schedule,
        max_attempts=MAX_ATTEMPTS,
        max_iters=3000,
        curve=True,
        random_state=RANDOM_SEED,
    )
    curve_list.append(curve)

df = pd.DataFrame(curve_list).transpose()
df.columns = decays
df.plot()
plt.xlabel("Iteration")
plt.ylabel("Fitness")
plt.title("FlipFlop: Fitness curve vs decay rate in SA")
plt.savefig("output/flipflop_sa_decay.png")
plt.close()

print(df.max())

#%% tuning for GA
curve_list = []
pop_sizes = [50, 100, 200, 400]
for p in pop_sizes:
    _, _, curve = mlrose.genetic_alg(
        problem,
        max_attempts=MAX_ATTEMPTS,
        max_iters=500,
        pop_size=p,
        elite_dreg_ratio=1,
        curve=True,
        random_state=RANDOM_SEED,
    )
    curve_list.append(curve)

df = pd.DataFrame(curve_list).transpose()
df.columns = pop_sizes
df.plot()
plt.xlabel("Iteration")
plt.ylabel("Fitness")
plt.title("FlipFlop: Fitness curve vs population size in GA")
plt.savefig("output/flipflop_ga_pop.png")
plt.close()

print(df.max())

#%% tuning for MIMIC

# curve_list = []
# nth_pct = [0.1, 0.2, 0.4]
# for p in nth_pct:
#    _, _, curve = mlrose.mimic(
#        problem,
#        max_attempts=MAX_ATTEMPTS,
#        max_iters=50,
#        keep_pct=p,
#        curve=True,
#        random_state=RANDOM_SEED,
#    )
#    curve_list.append(curve)
#
# df = pd.DataFrame(curve_list).transpose()
# df.columns = nth_pct
# df.plot()
# plt.xlabel("Iteration")
# plt.ylabel("Fitness")
# plt.title("FlipFlop: Fitness curve vs nth percentile in MIMIC")
# plt.savefig("output/flipflop_mimic_nth.png")
# plt.close()
#
# print(df.max())

#%% Putting together

curve_list = []
time_list = []
n_eval = []
algo_list = ["RHC", "SA", "GA", "MIMIC"]

# RHC
t1 = process_time()
_, _, curve = mlrose.random_hill_climb(
    problem, max_attempts=MAX_ATTEMPTS, curve=True, random_state=RANDOM_SEED,
)
t2 = process_time()
time_list.append((t2 - t1) / len(curve))
curve_list.append(curve)
n_eval.append(np.argmax(curve) + 1)

# SA
t1 = process_time()
_, _, curve = mlrose.simulated_annealing(
    problem, max_attempts=MAX_ATTEMPTS, curve=True, random_state=RANDOM_SEED,
)
t2 = process_time()
time_list.append((t2 - t1) / len(curve))
curve_list.append(curve)
n_eval.append(np.argmax(curve) + 1)

# GA
t1 = process_time()
_, _, curve = mlrose.genetic_alg(
    problem,
    max_attempts=MAX_ATTEMPTS,
    elite_dreg_ratio=1,
    curve=True,
    random_state=RANDOM_SEED,
)
t2 = process_time()
time_list.append((t2 - t1) / len(curve))
curve_list.append(curve)
n_eval.append((np.argmax(curve) + 1) * 200)

# MIMIC
t1 = process_time()
_, _, curve = mlrose.mimic(
    problem, max_attempts=MAX_ATTEMPTS, curve=True, random_state=RANDOM_SEED,
)
t2 = process_time()
time_list.append((t2 - t1) / len(curve))
curve_list.append(curve)
n_eval.append((np.argmax(curve) + 1) * 200)

df = pd.DataFrame(curve_list).transpose()
df.columns = algo_list
df.plot()
plt.xlabel("Iteration")
plt.ylabel("Fitness")
plt.title("FlipFlop: Fitness curve vs algorithms")
plt.savefig("output/flipflop_algo.png")
plt.close()

print("time per iteration:")
print(time_list)
print("number of func eval reaching maxima:")
print(n_eval)
print("maxima reached:")
print(df.max())
