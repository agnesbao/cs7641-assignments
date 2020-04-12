import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

from make_mdp import PROBS
from q_fc import QLearning
from q_fc import test_policy

logging.basicConfig(
    format="[%(levelname)s] %(asctime)s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

to_solve = ["frozen_lake", "forest"]

for prob_key in PROBS:
    if prob_key not in to_solve:
        continue
    logging.info(f"Running {prob_key}...")
    P, R = PROBS[prob_key]

    if prob_key == "frozen_lake":
        gamma = 0.99
        N_EPISODES = [1000, 5000, 10000, 50000, 100000]

    elif prob_key == "forest":
        gamma = 0.99
        N_EPISODES = [10000, 50000, 100000, 500000, 1000000]

    logging.info("..Running Q-Learning...")

    res_dict = {
        "Mean Reward": [],
        "Mean dQ": [],
        "Max V": [],
        "Mean V": [],
        "Optimal policy reward": [],
    }
    V_all = []
    policy_all = []
    r_std_all = []

    # EXP 1: n_episode
    for n_epi in N_EPISODES:
        logging.info(f"..n_episode = {n_epi}...")
        # constant learning rate
        alpha_schedule = [0.01] * n_epi
        # all exploration
        epsilon_schedule = [0.5] * n_epi
        ql = QLearning(
            P,
            R,
            gamma=gamma,
            n_episode=n_epi,
            alpha_schedule=alpha_schedule,
            epsilon_schedule=epsilon_schedule,
        )
        random.seed(0)
        ql.run()
        res_dict["Mean Reward"].append(np.mean(ql.run_stats["Reward"][-1000:]))
        res_dict["Mean dQ"].append(np.mean(ql.run_stats["Error"][-1000:]))
        res_dict["Max V"].append(ql.run_stats["Max V"][-1])
        res_dict["Mean V"].append(ql.run_stats["Mean V"][-1])
        logging.info("...testing...")
        test_r_mean, test_r_std = test_policy(P, R, ql.policy)
        res_dict["Optimal policy reward"].append(test_r_mean)
        r_std_all.append(test_r_std)
        V_all.append(ql.V)
        policy_all.append(ql.policy)
    res_df = pd.DataFrame(res_dict)
    res_df.index = np.array(N_EPISODES).astype(str)
    res_df.to_csv(f"data/{prob_key}_ql_n_epi.csv")
    res_df.plot(
        subplots=True,
        title=f"Q-Learning vs. training episodes on {prob_key}",
        style=".-",
        figsize=(7, 7),
    )
    plt.xlabel("n_episode")
    plt.savefig(f"output/{prob_key}_ql_n_epi.png")
    plt.close()

    V_df = pd.DataFrame(V_all, index=res_df.index).T
    V_df.to_csv(f"data/{prob_key}_ql_n_epi_V.csv")

    policy_df = pd.DataFrame(policy_all, index=res_df.index).T
    policy_df.to_csv(f"data/{prob_key}_ql_n_epi_policy.csv")
