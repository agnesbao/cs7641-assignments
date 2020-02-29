import mlrose_hiive as mlrose
import matplotlib.pyplot as plt
import pandas as pd
from time import process_time

fitness = mlrose.OneMax()
problem = mlrose.DiscreteOpt(100, fitness)

RANDOM_SEED = 42

#%% tuning for SA
curve_list = []
decays = [0.999, 0.99, 0.9]
for d in decays:
    schedule = mlrose.GeomDecay(decay=d)
    _, _, curve = mlrose.simulated_annealing(
        problem,
        schedule=schedule,
        max_attempts=100,
        max_iters=500,
        curve=True,
        random_state=RANDOM_SEED,
    )
    curve_list.append(curve)

df = pd.DataFrame(curve_list).transpose()
df.columns = decays
df.plot()
plt.xlabel("Iteration")
plt.ylabel("Fitness")
plt.title("OneMax: Fitness curve vs decay rate in SA")
plt.savefig("onemax_sa_decay.png")
plt.close()

#%% tuning for GA
curve_list = []
pop_sizes = [10, 20, 30, 40, 100, 200]
for p in pop_sizes:
    _, _, curve = mlrose.genetic_alg(
        problem,
        max_attempts=100,
        max_iters=50,
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
plt.title("OneMax: Fitness curve vs population size in GA")
plt.savefig("onemax_ga_pop.png")
plt.close()

#%% tuning for MIMIC

curve_list = []
nth_pct = [0.1, 0.2, 0.4]
for p in nth_pct:
    _, _, curve = mlrose.mimic(
        problem, keep_pct=p, curve=True, random_state=RANDOM_SEED,
    )
    curve_list.append(curve)

df = pd.DataFrame(curve_list).transpose()
df.columns = nth_pct * 10
df.plot()
plt.xlabel("Iteration")
plt.ylabel("Fitness")
plt.title("OneMax: Fitness curve vs nth percentile in MIMIC")
plt.savefig("onemax_ga_pop.png")
plt.close()

#%% Putting together
RANDOM_SEED = 21

curve_list = []
time_list = []
algo_list = ["RHC", "SA", "GA", "MIMIC"]

# RHC
t1 = process_time()
_, _, curve = mlrose.random_hill_climb(
    problem, max_attempts=100, curve=True, random_state=RANDOM_SEED
)
t2 = process_time()
time_list.append(t2 - t1)
curve_list.append(curve)

# SA
schedule = mlrose.GeomDecay(decay=0.9)
t1 = process_time()
_, _, curve = mlrose.simulated_annealing(
    problem, schedule=schedule, max_attempts=100, curve=True, random_state=RANDOM_SEED
)
t2 = process_time()
time_list.append(t2 - t1)
curve_list.append(curve)

# GA
t1 = process_time()
_, _, curve = mlrose.genetic_alg(
    problem, elite_dreg_ratio=1, curve=True, random_state=RANDOM_SEED
)
t2 = process_time()
time_list.append(t2 - t1)
curve_list.append(curve)

# MIMIC
t1 = process_time()
_, _, curve = mlrose.mimic(problem, curve=True, random_state=RANDOM_SEED)
t2 = process_time()
time_list.append(t2 - t1)
curve_list.append(curve)

df = pd.DataFrame(curve_list).transpose()
df.columns = algo_list
df.plot()
