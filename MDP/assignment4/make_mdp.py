import numpy as np
import gym
from hiive.mdptoolbox.example import forest


PROBS = {}

PROBS["forest"] = forest(S=10000)

env = gym.make("FrozenLake8x8-v0")

nA, nS = env.nA, env.nS
P = np.zeros([nA, nS, nS])
R = np.zeros([nS, nA])
for s in range(nS):
    for a in range(nA):
        transitions = env.P[s][a]
        for p_trans, next_s, reward, _ in transitions:
            P[a, s, next_s] += p_trans
            R[s, a] = reward
        P[a, s, :] /= np.sum(P[a, s, :])
PROBS["frozen_lake"] = (P, R)
