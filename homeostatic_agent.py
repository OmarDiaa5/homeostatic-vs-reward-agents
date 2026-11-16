"""Homeostatic vs reward-maximising PPO on a small grid world.
Reward (homeostatic): r = D(h_t) - D(h_{t+1}), D(h)=||h*-h||, h*=(0.8,0.8)  (Keramati & Gutkin 2014).
"""
import json, sys, time
import numpy as np, torch
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.set_num_threads(4)
N = 8
H_STAR = np.array([0.8, 0.8])
TRAIN_LAYOUT = dict(food=(0, 0), water=(7, 7), toxic=(3, 4))
SHIFT_LAYOUT = None  # random resource positions each episode (test only)
DECAY = 0.004
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 150_000
SEEDS = [int(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 else [0, 1, 2]
SANITY = len(sys.argv) > 3


class HomeoGrid(gym.Env):
    def __init__(self, mode, layout=TRAIN_LAYOUT, decay=DECAY):
        self.mode, self.layout, self.decay, self.fixed = mode, layout, decay, layout
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(0, 1, (10,), np.float32)
        self.last_event = None

    def _obs(self):
        l = self.layout
        pos = [self.pos[0], self.pos[1], *l["food"], *l["water"], *l["toxic"]]
        return np.concatenate([np.array(pos) / (N - 1), self.h]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pos = tuple(self.np_random.integers(0, N, 2))
        self.h = self.np_random.uniform(0.4, 1.0, 2)
        self.t = 0
        if self.fixed is None:
            cells = self.np_random.choice(N * N, 3, replace=False)
            self.layout = {k: (int(c // N), int(c % N)) for k, c in zip(["food", "water", "toxic"], cells)}
        return self._obs(), {}

    def step(self, a):
        dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][a]
        self.pos = (int(np.clip(self.pos[0] + dx, 0, N - 1)), int(np.clip(self.pos[1] + dy, 0, N - 1)))
        d_old = np.linalg.norm(H_STAR - self.h)
        self.h = self.h - self.decay
        self.last_event = None
        if self.pos == self.layout["food"]:
            self.h[0] += 0.3; self.last_event = "food"
        elif self.pos == self.layout["water"]:
            self.h[1] += 0.3; self.last_event = "water"
        elif self.pos == self.layout["toxic"]:
            self.h[0] -= 0.4
        self.h = np.minimum(self.h, 1.0)
        self.t += 1
        dead = bool(self.h.min() <= 0)
        if self.mode == "homeostatic":
            r = 10 * (d_old - np.linalg.norm(H_STAR - np.maximum(self.h, 0)))
        else:
            r = 1.0 if self.last_event else 0.0
        if dead:
            r -= 1.0
        return self._obs(), float(r), dead, self.t >= 500, {}


def rollout(model, env, n_ep, seed):
    surv, samples = [], []  # samples: (e-h, next consumption)
    for i in range(n_ep):
        obs, _ = env.reset(seed=seed * 1000 + i)
        diffs, done = [], False
        ep_samples, pending = [], []
        while not done:
            diffs.append(env.h[0] - env.h[1]); pending.append(diffs[-1])
            a, _ = model.predict(obs, deterministic=False)
            obs, _, term, trunc, _ = env.step(int(a))
            if env.last_event:
                ep_samples += [(d, env.last_event == "food") for d in pending]; pending = []
            done = term or trunc
        surv.append(env.t); samples += ep_samples
    return surv, samples


if __name__ == "__main__":
    t0 = time.time()
    res = {"steps": STEPS, "modes": {}}
    class Rand:
        def predict(self, o, deterministic=False): return np.random.randint(4), None
    res["random_survival_train"] = float(np.mean(rollout(Rand(), HomeoGrid("homeostatic"), 50, 99)[0]))
    res["random_survival_shift"] = float(np.mean(rollout(Rand(), HomeoGrid("homeostatic", SHIFT_LAYOUT, 2 * DECAY), 50, 99)[0]))
    print("random", res["random_survival_train"], res["random_survival_shift"], flush=True)
    bins = np.linspace(-0.6, 0.6, 7)
    fig1 = {}
    for mode in ["homeostatic", "reward_max"]:
        surv_train, surv_shift, all_samples = [], [], []
        for s in SEEDS:
            env = DummyVecEnv([lambda: HomeoGrid(mode) for _ in range(4)])
            model = PPO("MlpPolicy", env, seed=s, device="cpu", verbose=0, n_steps=256, batch_size=256, ent_coef=0.01)
            model.learn(STEPS)
            st, smp = rollout(model, HomeoGrid(mode), 30, s)
            ss, _ = rollout(model, HomeoGrid(mode, SHIFT_LAYOUT, 2 * DECAY), 30, s)
            surv_train.append(float(np.mean(st))); surv_shift.append(float(np.mean(ss)))
            all_samples += smp
            print(mode, s, surv_train[-1], surv_shift[-1], f"{time.time()-t0:.0f}s", flush=True)
        d = np.array([x[0] for x in all_samples]); f = np.array([x[1] for x in all_samples], float)
        idx = np.digitize(d, bins)
        p = [float(f[idx == k].mean()) if (idx == k).sum() >= 20 else None for k in range(1, len(bins))]
        fig1[mode] = p
        res["modes"][mode] = dict(
            survival_train_per_seed=surv_train, survival_shift_per_seed=surv_shift,
            survival_train_mean=float(np.mean(surv_train)), survival_train_sd=float(np.std(surv_train)),
            survival_shift_mean=float(np.mean(surv_shift)), survival_shift_sd=float(np.std(surv_shift)),
            p_food_by_bin=p)
    res["bin_edges_energy_minus_hydration"] = bins.tolist()
    res["train_minutes"] = (time.time() - t0) / 60
    if SANITY:
        sys.exit()
    json.dump(res, open("results.json", "w"), indent=1)

    cen = (bins[:-1] + bins[1:]) / 2
    plt.figure(figsize=(5, 3.5))
    for mode, c in [("homeostatic", "C0"), ("reward_max", "C1")]:
        pts = [(x, y) for x, y in zip(cen, fig1[mode]) if y is not None]
        plt.plot(*zip(*pts), "o-", color=c, label=mode)
    plt.axhline(0.5, color="gray", ls=":")
    plt.xlabel("energy - hydration"); plt.ylabel("P(next consumption = food)")
    plt.legend(); plt.tight_layout(); plt.savefig("fig_need_vs_choice.png", dpi=150)

    plt.figure(figsize=(5, 3.5))
    w = 0.35
    for j, mode in enumerate(["homeostatic", "reward_max"]):
        m = res["modes"][mode]
        plt.bar(np.arange(2) + (j - .5) * w, [m["survival_train_mean"], m["survival_shift_mean"]], w,
                yerr=[m["survival_train_sd"], m["survival_shift_sd"]], capsize=3, label=mode)
    plt.xticks([0, 1], ["training env", "shifted env"]); plt.ylabel("mean survival steps (max 500)")
    plt.legend(); plt.tight_layout(); plt.savefig("fig_robustness.png", dpi=150)
    print("done", f"{(time.time()-t0)/60:.1f} min")
