from hiive.mdptoolbox.mdp import ValueIteration
from make_mdp import PROBS

to_solve = ["forest"]

for prob_key in PROBS:
    if prob_key not in to_solve:
        continue
    P, R = PROBS[prob_key]
    vi = ValueIteration(P, R, gamma=0.99, skip_check=True)
    # vi.run()
