import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

from make_mdp import PROBS
from q_fc import QLearning
from q_fc import make_schedules
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
        n_epi = 10000
        epsilon_schedule = make_schedules(n_epi)

    elif prob_key == "forest":
        gamma = 0.99
        n_epi = 100000
        epsilon_schedules = make_schedules(n_epi)

    logging.info("..Running Q-Learning...")

    res_dict = {
        "Mean Reward": [],
        "Mean dQ": [],
        "Max V": [],
        "Mean V": [],
        "Optimal policy reward": [],
        "eps schedule key": [],
    }
    V_all = []
    policy_all = []
    r_std_all = []

    # EXP 2: epsilon
    for key in epsilon_schedules:
        logging.info(f"..epsilon schedule {key}...")
        alpha_schedule = [0.01] * n_epi
        eps_schedule = epsilon_schedules[key]
        ql = QLearning(
            P,
            R,
            gamma=gamma,
            n_episode=n_epi,
            alpha_schedule=alpha_schedule,
            epsilon_schedule=eps_schedule,
        )
        random.seed(0)
        ql.run()
        res_dict["Mean Reward"].append(np.mean(ql.run_stats["Reward"][-1000:]))
        res_dict["Mean dQ"].append(np.mean(ql.run_stats["Error"][-1000:]))
        res_dict["Max V"].append(ql.run_stats["Max V"][-1])
        res_dict["Mean V"].append(ql.run_stats["Mean V"][-1])
        res_dict["eps schedule key"].append(key)
        logging.info("...testing...")
        test_r_mean, test_r_std = test_policy(P, R, ql.policy)
        res_dict["Optimal policy reward"].append(test_r_mean)
        r_std_all.append(test_r_std)
        V_all.append(ql.V)
        policy_all.append(ql.policy)
    res_df = pd.DataFrame(res_dict).set_index("eps schedule key")
    res_df.to_csv(f"data/{prob_key}_ql_eps.csv")
    res_df.plot(
        subplots=True,
        title=f"Q-Learning vs. exploration schedule on {prob_key}",
        style=".-",
        figsize=(7, 7),
    )
    plt.xlabel("n_episode")
    plt.savefig(f"output/{prob_key}_ql_eps.png")
    plt.close()

    V_df = pd.DataFrame(V_all, index=res_df.index).T
    V_df.to_csv(f"data/{prob_key}_ql_eps_V.csv")

    policy_df = pd.DataFrame(policy_all, index=res_df.index).T
    policy_df.to_csv(f"data/{prob_key}_ql_eps_policy.csv")

    pd.DataFrame(epsilon_schedules).plot(
        title=f"Q-Learning epsilon schedule on {prob_key}"
    )
    plt.xlabel("n_episode")
    plt.ylabel("epsilon")
    plt.savefig(f"output/{prob_key}_ql_eps_schedule.png")
    plt.close()
