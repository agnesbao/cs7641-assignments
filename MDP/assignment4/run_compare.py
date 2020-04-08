import random
import numpy as np
import pandas as pd
from hiive.mdptoolbox.mdp import ValueIteration
from hiive.mdptoolbox.mdp import PolicyIteration
import matplotlib.pyplot as plt

from make_mdp import PROBS
from q_learn import decay_schedule
from q_learn import QLearning
from q_learn import test_policy


def print_frozen_lake_policy(policy):
    actions = "<v>^"
    nE = np.sqrt(len(policy)).astype(int)
    map_action = [actions[p] for p in policy]
    return np.array(map_action).reshape((nE, nE))


to_solve = ["frozen_lake"]

for prob_key in PROBS:
    if prob_key not in to_solve:
        continue
    print(f"Running {prob_key}...")
    P, R = PROBS[prob_key]

    print("..Running value iteration...")
    vi = ValueIteration(P, R, gamma=0.99, epsilon=0.001, max_iter=1000)
    vi.run()
    vi_df = pd.DataFrame(vi.run_stats).set_index("Iteration")
    vi_df.columns = pd.MultiIndex.from_product([vi_df.columns, ["value_iter"]])
    print(f"Runtime per value iter: {vi.time/vi.iter} sec")
    print(f"VI performance: {test_policy(P,R,vi.policy)}")

    print("..Running policy iteration...")
    pi = PolicyIteration(P, R, gamma=0.99, eval_type=1, max_iter=1000)
    pi.run()
    pi_df = pd.DataFrame(pi.run_stats).set_index("Iteration")
    pi_df.columns = pd.MultiIndex.from_product([pi_df.columns, ["policy_iter"]])
    print(f"Runtime per policy iter: {pi.time/pi.iter} sec")
    print(f"PI performance: {test_policy(P,R,pi.policy)}")

    print("..Running q-learning...")
    n_episode = 10000
    alpha_schedule = decay_schedule(0.1, 0.001, 0.999, n_episode)
    epsilon_schedule = decay_schedule(1, 0.01, 0.999, n_episode)
    ql = QLearning(
        P,
        R,
        gamma=0.99,
        alpha_schedule=alpha_schedule,
        epsilon_schedule=epsilon_schedule,
        n_episode=n_episode,
    )
    random.seed(0)
    ql.run()
    ql_df = pd.DataFrame(ql.run_stats)
    ql_df["Alpha"] = alpha_schedule
    ql_df["Epsilon"] = epsilon_schedule
    ql_df.to_csv(f"{prob_key}_qlearn.csv")
    ql_df[["Reward_rolling_mean", "Error_rolling_mean"]] = (
        ql_df[["Reward", "Error"]].rolling(1000).mean()
    )
    ql_df[
        [
            "Reward_rolling_mean",
            "Error_rolling_mean",
            "Time",
            "Alpha",
            "Epsilon",
            "Max V",
            "Mean V",
        ]
    ].plot(subplots=True, title=f"Q-Learning training on {prob_key}")
    plt.xlabel("Episode")
    plt.savefig(f"output/{prob_key}_q.png")
    plt.close()
    print(f"Runtime per q-learning episode: {ql.time/ql.n_episode} sec")
    print(f"Q-learning performance: {test_policy(P,R,ql.policy)}")

    res_df = vi_df.join(pi_df, how="outer")
    res_df.to_csv(f"{prob_key}_vpiter.csv")
    for var in ["Reward", "Error", "Time", "Max V", "Mean V"]:
        res_df.xs(var, level=0, axis=1).plot(
            title=f"{var} on {prob_key}", figsize=(8, 4)
        )
        plt.savefig(f"output/{prob_key}_{var.replace(' ','')}.png")
        plt.close()

    print(
        f"Policy diff vi vs. pi: {sum([p1!=p2 for p1,p2 in zip(vi.policy, pi.policy)])}"
    )
    print(
        f"Policy diff vi vs. ql: {sum([p1!=p2 for p1,p2 in zip(vi.policy, ql.policy)])}"
    )
    print(
        f"Policy diff pi vs. ql: {sum([p1!=p2 for p1,p2 in zip(pi.policy, ql.policy)])}"
    )

    print(f"Value diff vi vs. pi: {sum([abs(v1-v2) for v1,v2 in zip(vi.V, pi.V)])}")
    print(f"Value diff vi vs. ql: {sum([abs(v1-v2) for v1,v2 in zip(vi.V, ql.V)])}")
    print(f"Value diff pi vs. ql: {sum([abs(v1-v2) for v1,v2 in zip(pi.V, ql.V)])}")
