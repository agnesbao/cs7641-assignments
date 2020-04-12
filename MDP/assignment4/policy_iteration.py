import numpy as np
import pandas as pd
from hiive.mdptoolbox.mdp import PolicyIteration
import matplotlib.pyplot as plt

from make_mdp import PROBS
from q_fc import test_policy

to_solve = ["frozen_lake", "forest"]

DISCOUNT_RATES = [0.3, 0.5, 0.8, 0.9, 0.99, 0.999]

for prob_key in PROBS:
    if prob_key not in to_solve:
        continue
    print(f"Running {prob_key}...")
    P, R = PROBS[prob_key]

    print("..Running policy iteration...")

    res_dict = {
        "Iteration to converge": [],
        "Max V": [],
        "Mean V": [],
        "Optimal policy reward": [],
    }
    V_all = []
    policy_all = []
    r_std_all = []
    for g in DISCOUNT_RATES:
        print(f"..discount rate = {g}...")
        pi = PolicyIteration(P, R, gamma=g, eval_type=1, max_iter=1000)
        pi.run()
        res_dict["Iteration to converge"].append(pi.iter)
        res_dict["Max V"].append(pi.run_stats[-1]["Max V"])
        res_dict["Mean V"].append(pi.run_stats[-1]["Mean V"])
        print("...testing...")
        test_r_mean, test_r_std = test_policy(P, R, pi.policy)
        res_dict["Optimal policy reward"].append(test_r_mean)
        r_std_all.append(test_r_std)
        V_all.append(pi.V)
        policy_all.append(pi.policy)
    res_df = pd.DataFrame(res_dict)
    res_df.index = np.array(DISCOUNT_RATES).astype(str)
    res_df.plot(
        subplots=True,
        title=f"Policy iteration vs. discount rate on {prob_key}",
        style=".-",
        figsize=(7, 7),
    )
    plt.xlabel("discount rate")
    plt.savefig(f"output/{prob_key}_pi_gamma.png")
    plt.close()
    res_df.to_csv(f"data/{prob_key}_pi_gamma.csv")

    V_df = pd.DataFrame(V_all, index=res_df.index).T
    V_df.to_csv(f"data/{prob_key}_pi_V.csv")

    policy_df = pd.DataFrame(policy_all, index=res_df.index).T
    policy_df.to_csv(f"data/{prob_key}_pi_policy.csv")
