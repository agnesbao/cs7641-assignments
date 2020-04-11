import numpy as np
import gym
from hiive.mdptoolbox.example import forest
import matplotlib.pyplot as plt


PROBS = {}

PROBS["forest"] = forest(S=1000, p=0.001, r1=100, r2=10)

env = gym.make("FrozenLake8x8-v0")

nA, nS = env.nA, env.nS
P = np.zeros([nA, nS, nS])
R = np.zeros([nA, nS, nS])
DONE = np.zeros(nS)
for s in range(nS):
    for a in range(nA):
        transitions = env.P[s][a]
        for p_trans, next_s, reward, done in transitions:
            P[a, s, next_s] += p_trans
            R[a, s, next_s] += reward
            DONE[next_s] = done
        P[a, s, :] /= np.sum(P[a, s, :])
PROBS["frozen_lake"] = (P, R)

nA, nS = env.nA, env.nS
P = np.zeros([nA, nS, nS])
R = np.zeros([nA, nS, nS])
DONE = np.zeros(nS)
for s in range(nS):
    for a in range(nA):
        transitions = env.P[s][a]
        for p_trans, next_s, reward, done in transitions:
            P[a, s, next_s] += p_trans
            # penalize hole
            if done and reward == 0 and s != next_s:
                reward = -0.1
            R[a, s, next_s] += reward
            DONE[next_s] = done
        P[a, s, :] /= np.sum(P[a, s, :])
PROBS["frozen_lake_modrew"] = (P, R)

_, ax = plt.subplots(figsize=(5, 5))
for i in range(8):
    for j in range(8):
        if env.desc[i, j] == b"S":
            c = "y"
        elif env.desc[i, j] == b"H":
            c = "r"
        elif env.desc[i, j] == b"G":
            c = "g"
        elif env.desc[i, j] == b"F":
            c = "b"
        x = (i + 0.5) / 8
        y = (7 - j + 0.5) / 8
        p = plt.Rectangle((i / 8, (7 - j) / 8), 1 / 8, 1 / 8, color=c)
        ax.add_patch(p)
        plt.text(
            x, y, env.desc[i, j].decode(), size=10, c="w", ha="center", va="center"
        )
plt.xticks([])
plt.yticks([])
plt.title("frozen_lake env map")
plt.savefig("output/frozen_lake_map.png")
