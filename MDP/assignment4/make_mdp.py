import numpy as np
import gym
from hiive.mdptoolbox.example import forest


PROBS = {}

env = gym.make("FrozenLake8x8-v0")

nA, nS = env.nA, env.nS
P = np.zeros([nA, nS, nS])
R = np.zeros([nS, nA])
for s in range(nS):
    for a in range(nA):
        transitions = env.P[s][a]
        for p_trans, next_s, reward, _ in transitions:
            P[a, s, next_s] += p_trans
            R[s, a] += reward * p_trans
        P[a, s, :] /= np.sum(P[a, s, :])
PROBS["frozen_lake"] = (P, R)

P, R = forest(S=1000000, is_sparse=True)
PROBS["forest"] = (P, R)
