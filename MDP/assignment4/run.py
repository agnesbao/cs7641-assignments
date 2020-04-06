import random
import pandas as pd
from hiive.mdptoolbox.mdp import ValueIteration
from hiive.mdptoolbox.mdp import PolicyIteration
from hiive.mdptoolbox.mdp import QLearning
import matplotlib.pyplot as plt

from make_mdp import PROBS

to_solve = ["frozen_lake", "forest"]

for prob_key in PROBS:
    if prob_key not in to_solve:
        continue
    print(f"Running {prob_key}...")
    P, R = PROBS[prob_key]
    vi = ValueIteration(P, R, gamma=0.99, epsilon=0.001, max_iter=1000)
    vi.run()
    vi_df = pd.DataFrame(vi.run_stats).set_index("Iteration")
    vi_df.columns = pd.MultiIndex.from_product([vi_df.columns, ["value_iter"]])
    print(f"Runtime per value iter: {vi_df.Time.iloc[-1]/len(vi_df)} sec")

    pi = PolicyIteration(P, R, gamma=0.99, eval_type=1, max_iter=1000)
    pi.run()
    pi_df = pd.DataFrame(pi.run_stats).set_index("Iteration")
    pi_df.columns = pd.MultiIndex.from_product([pi_df.columns, ["policy_iter"]])
    print(f"Runtime per policy iter: {pi_df.Time.iloc[-1]/len(pi_df)} sec")

    ql = QLearning(
        P,
        R,
        gamma=0.99,
        alpha=0.5,
        alpha_min=0.01,
        alpha_decay=0.9999,
        epsilon=1,
        epsilon_min=0.1,
        epsilon_decay=0.99995,
        n_iter=100000,
    )
    random.seed(0)
    ql.run()
    ql_df = pd.DataFrame(ql.run_stats).set_index("Iteration")
    ql_df.to_csv(f"{prob_key}_qlearn.csv")
    ql_df[["Reward", "Error", "Time", "Alpha", "Epsilon", "Max V", "Mean V"]].plot(
        subplots=True, title="Q-Learning training"
    )
    plt.xlabel("Step")
    plt.savefig(f"output/{prob_key}_q.png")
    plt.close()

    res_df = vi_df.join(pi_df, how="outer")
    res_df.to_csv(f"{prob_key}_vpiter.csv")
    for var in ["Reward", "Error", "Time", "Max V", "Mean V"]:
        res_df.xs(var, level=0, axis=1).plot(title=var, figsize=(8, 4))
        plt.savefig(f"output/{prob_key}_{var.replace(' ','')}.png")
        plt.close()

    print(f"Policy diff vi vs. pi: {sum([abs(v-p).sum() for v,p in zip(vi.P, pi.P)])}")
    print(f"Policy diff vi vs. ql: {sum([abs(v-p).sum() for v,p in zip(vi.P, ql.P)])}")
    print(f"Policy diff pi vs. ql: {sum([abs(v-p).sum() for v,p in zip(ql.P, ql.P)])}")

    print(f"Value diff vi vs. pi: {sum([abs(v-p) for v,p in zip(vi.V, pi.V)])}")
    print(f"Value diff vi vs. ql: {sum([abs(v-p) for v,p in zip(vi.V, ql.V)])}")
    print(f"Value diff pi vs. ql: {sum([abs(v-p) for v,p in zip(ql.V, pi.V)])}")
