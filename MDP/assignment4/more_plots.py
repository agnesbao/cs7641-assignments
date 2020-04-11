import pandas as pd
import matplotlib.pyplot as plt

for prob_key in ["frozen_lake", "forest"]:
    vi_V = pd.read_csv(f"data/{prob_key}_vi_V.csv", index_col=0)
    pi_V = pd.read_csv(f"data/{prob_key}_pi_V.csv", index_col=0)
    _, axes = plt.subplots(
        vi_V.shape[1], 1, sharex=True, figsize=(8, 8), gridspec_kw={"hspace": 0.7}
    )
    i = 0
    for col in vi_V.columns:
        ax = axes[i]
        ax.plot(vi_V[col])
        ax.plot(pi_V[col], ":")
        ax.set_title(f"gamma={col}")
        i += 1
    plt.legend(["value iteration", "policy iteration"])
    plt.xlabel("state")
    plt.suptitle("Value function from value iteration and policy iteration")
    plt.savefig(f"output/{prob_key}_V_vi_pi.png")
    plt.close()

    V_diff = vi_V - pi_V
    axes = V_diff.plot(
        subplots=True,
        title="Value function difference between value iteration and policy iteration",
        figsize=(7, 7),
    )
    for ax in axes:
        ax.ticklabel_format(useOffset=False)
    plt.xlabel("state")
    plt.savefig(f"output/{prob_key}_V_diff.png")
    plt.close()

    vi_policy = pd.read_csv(f"data/{prob_key}_vi_policy.csv", index_col=0)
    pi_policy = pd.read_csv(f"data/{prob_key}_pi_policy.csv", index_col=0)
    _, axes = plt.subplots(
        vi_policy.shape[1], 1, sharex=True, figsize=(7, 7), gridspec_kw={"hspace": 0.7}
    )
    i = 0
    for col in vi_V.columns:
        ax = axes[i]
        ax.plot(vi_policy[col], "-")
        ax.plot(pi_policy[col], ":")
        ax.set_title(f"gamma={col}")
        i += 1
    plt.legend(["value iteration", "policy iteration"])
    plt.xlabel("state")
    plt.suptitle("Policy from value iteration and policy iteration")
    plt.savefig(f"output/{prob_key}_policy_vi_pi.png")
    plt.close()
