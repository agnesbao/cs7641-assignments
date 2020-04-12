import pandas as pd
from hiive.mdptoolbox.example import forest
from hiive.mdptoolbox.mdp import ValueIteration
from hiive.mdptoolbox.mdp import PolicyIteration

N_STATES = range(100, 1001, 100)
V_all = {}
policy_all = {}
vi_res = {
    "Iteration to converge": [],
    "Time to converge": [],
    "Max V": [],
    "Mean V": [],
}
pi_res = {
    "Iteration to converge": [],
    "Time to converge": [],
    "Max V": [],
    "Mean V": [],
}
for s in N_STATES:
    print(f"Running nS {s}...")
    P, R = forest(S=s, p=0.001, r1=100, r2=10)
    vi = ValueIteration(P, R, gamma=0.99, epsilon=0.001)
    vi.run()
    vi_res["Iteration to converge"].append(vi.iter)
    vi_res["Time to converge"].append(vi.time)
    vi_res["Max V"].append(vi.run_stats[-1]["Max V"])
    vi_res["Mean V"].append(vi.run_stats[-1]["Mean V"])
    V_all[("vi", s)] = vi.V
    policy_all[("vi", s)] = vi.policy
    pi = PolicyIteration(P, R, gamma=0.99, eval_type=1, max_iter=1000)
    pi.run()
    pi_res["Iteration to converge"].append(pi.iter)
    pi_res["Time to converge"].append(pi.time)
    pi_res["Max V"].append(pi.run_stats[-1]["Max V"])
    pi_res["Mean V"].append(pi.run_stats[-1]["Mean V"])
    V_all[("pi", s)] = pi.V
    policy_all[("pi", s)] = pi.policy
vi_df = pd.DataFrame(vi_res, index=N_STATES)
pi_df = pd.DataFrame(pi_res, index=N_STATES)

vi_df.to_csv("data/vi_nS.csv")
pi_df.to_csv("data/pi_nS.csv")
pd.DataFrame.from_dict(V_all, orient="index").T.to_csv("data/nS_V.csv")
pd.DataFrame.from_dict(policy_all, orient="index").T.to_csv("data/nS_policy.csv")
