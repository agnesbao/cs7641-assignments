import numpy as np
import pandas as pd
from hiive.mdptoolbox.mdp import ValueIteration
import matplotlib.pyplot as plt

from make_mdp import PROBS
from q_fc import test_policy

to_solve = ["frozen_lake", "forest"]

DISCOUNT_RATES = [0.3, 0.5, 0.8, 0.9, 0.99, 0.999, 0.9999]

for prob_key in PROBS:
    if prob_key not in to_solve:
        continue
    print(f"Running {prob_key}...")
    P, R = PROBS[prob_key]

    print("..Running value iteration...")

    res_dict = {
        "Iteration to converge": [],
        "Reward": [],
        "Max V": [],
        "Mean V": [],
        "Optimal policy reward": [],
    }
    V_all = []
    policy_all = []
    r_std_all = []
    for g in DISCOUNT_RATES:
        print(f"..discount rate = {g}...")
        vi = ValueIteration(P, R, gamma=g, epsilon=0.001)
        vi.run()
        res_dict["Iteration to converge"].append(vi.iter)
        res_dict["Reward"].append(vi.run_stats[-1]["Reward"])
        res_dict["Max V"].append(vi.run_stats[-1]["Max V"])
        res_dict["Mean V"].append(vi.run_stats[-1]["Mean V"])
        print("...testing...")
        test_r_mean, test_r_std = test_policy(P, R, vi.policy)
        res_dict["Optimal policy reward"].append(test_r_mean)
        r_std_all.append(test_r_std)
        V_all.append(vi.V)
        policy_all.append(vi.policy)
    res_df = pd.DataFrame(res_dict)
    res_df.index = np.array(DISCOUNT_RATES).astype(str)
    res_df.plot(
        subplots=True,
        title=f"Value iteration vs. discount rate on {prob_key}",
        style=".-",
        figsize=(7, 7),
    )
    plt.xlabel("discount rate")
    plt.savefig(f"output/{prob_key}_vi_gamma.png")
    plt.close()
    res_df.to_csv(f"data/{prob_key}_vi_gamma.csv")

    V_df = pd.DataFrame(V_all, index=res_df.index).T
    V_df.to_csv(f"data/{prob_key}_vi_V.csv")

    policy_df = pd.DataFrame(policy_all, index=res_df.index).T
    policy_df.to_csv(f"data/{prob_key}_vi_policy.csv")
