import random
import time
import numpy as np
from attr import attrs, attrib


@attrs
class QLearning:
    P = attrib()
    R = attrib()
    gamma = attrib(default=0.99)
    alpha_schedule = attrib(default=[0.01] * 1000)
    epsilon_schedule = attrib(default=[0.1] * 1000)
    n_episode = attrib(default=1000)
    max_step = attrib(default=1000)

    def __attrs_post_init__(self):
        if len(self.R.shape) == 3:
            self.nA, self.nS = self.R.shape[:2]
        else:
            self.nS, self.nA = self.R.shape
        self.Q = np.zeros((self.nS, self.nA))

    def _step(self, a, s):
        p = 0
        p_next_s = np.random.random()
        next_s = -1
        while (p < p_next_s) and (next_s < self.nS - 1):
            next_s += 1
            p = p + self.P[a, s, next_s]

        try:
            r = self.R[a, s, next_s]
        except IndexError:
            try:
                r = self.R[s, a]
            except IndexError:
                r = self.R[s]

        if len(self.R.shape) == 3:
            done = self.P[a, next_s, next_s] == 1
        else:
            done = next_s == 0

        return next_s, r, done

    def run(self):
        self.run_stats = {
            "Reward": [],
            "Error": [],
            "Time": [],
            "Max V": [],
            "Mean V": [],
        }
        start_time = time.time()
        for episode in range(self.n_episode):
            eps = self.epsilon_schedule[episode]
            alpha = self.alpha_schedule[episode]
            s = random.randrange(self.nS)
            done = False
            step = 0
            self.Q_old = self.Q.copy()
            r_episode = 0
            while not done and step < self.max_step:
                if random.random() < eps:
                    a = random.randrange(self.nA)
                else:
                    a = np.argmax(self.Q[s, :])
                next_s, r, done = self._step(a, s)
                if not done:
                    self.Q[s, a] += alpha * (
                        r + self.gamma * max(self.Q[next_s, :]) - self.Q[s, a]
                    )
                else:
                    self.Q[s, a] = r
                r_episode += r
                step += 1

            self.time = time.time() - start_time
            self.run_stats["Time"].append(self.time)
            self.run_stats["Reward"].append(r_episode)
            error = abs((self.Q - self.Q_old)).sum()
            self.run_stats["Error"].append(error)

            # compute the value function and the policy
            self.V = self.Q.max(axis=1)
            self.policy = self.Q.argmax(axis=1)
            self.run_stats["Max V"].append(self.V.max())
            self.run_stats["Mean V"].append(self.V.mean())


def test_policy(P, R, policy):
    nS = P.shape[1]
    step_fc = QLearning(P, R)._step
    r_all = []
    for t in range(1000):
        if len(R.shape) == 3:
            # start frozen lake from 0
            s = 0
        else:
            # enter forest at random state
            s = random.randrange(nS)
        for i in range(1000):
            a = policy[s]
            next_s, r, done = step_fc(a, s)
            r = max(r, 0)
            if done:
                r_all.append(r)
                break
            else:
                s = next_s
        if not done:
            r_all.append(r)
    return np.mean(r_all), np.std(r_all)


def decay_schedule(start, end, decay, n):
    out = []
    val = start
    for i in range(n):
        out.append(val)
        val = max(val * decay, end)
    return out


def make_schedules(n_episode):
    return {
        "constant_1": [1] * n_episode,
        "constant_0.5": [0.5] * n_episode,
        "constant_0.1": [0.1] * n_episode,
        "constant_0.01": [0.01] * n_episode,
        "linear": np.linspace(1, 0.01, n_episode),
        "geom": np.geomspace(1, 0.01, n_episode),
        "exp_decay": decay_schedule(1, 0.01, 1 - 10 / n_episode, n_episode),
    }
