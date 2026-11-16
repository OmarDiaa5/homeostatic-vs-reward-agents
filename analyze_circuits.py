"""Circuit analysis of one homeostatic PPO agent: population PCA + single-unit ablation."""
import json, time
import numpy as np, torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from homeostatic_agent import HomeoGrid, STEPS
import os
SEED = int(os.environ.get("AGENT_SEED", 0))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.set_num_threads(4)
t0 = time.time()
env = DummyVecEnv([lambda: HomeoGrid("homeostatic") for _ in range(4)])
model = PPO("MlpPolicy", env, seed=SEED, device="cpu", verbose=0, n_steps=256, batch_size=256, ent_coef=0.01)
model.learn(STEPS)
model.save("model_homeostatic")
print("trained", f"{time.time()-t0:.0f}s", flush=True)

net = model.policy.mlp_extractor.policy_net  # Linear-Tanh-Linear-Tanh; output = last hidden layer
acts = []
hook = net.register_forward_hook(lambda m, i, o: acts.append(o.detach().numpy()[0].copy()))

def run(n_ep, seed0, record=False):
    e = HomeoGrid("homeostatic"); food, water, H = [], [], []
    for i in range(n_ep):
        obs, _ = e.reset(seed=seed0 + i); done = False; nf = nw = 0
        while not done:
            if record: H.append(e.h.copy())
            a, _ = model.predict(obs, deterministic=False)
            obs, _, term, trunc, _ = e.step(int(a))
            nf += e.last_event == "food"; nw += e.last_event == "water"
            done = term or trunc
        food.append(nf); water.append(nw)
    return np.mean(food), np.mean(water), np.array(H)

# population analysis
torch.manual_seed(0); np.random.seed(0)
acts.clear()
f0, w0, H = run(20, 5000, record=True)
A = np.array(acts); assert len(A) == len(H)
hook.remove()
Ac = A - A.mean(0)
U, S, Vt = np.linalg.svd(Ac, full_matrices=False)
pcs = Ac @ Vt[:2].T
var = (S ** 2) / (S ** 2).sum()
diff = H[:, 0] - H[:, 1]
corr = float(np.corrcoef(pcs[:, 0], diff)[0, 1])
print("var", var[:2], "corr PC1", corr, flush=True)

# ablation
n_units = A.shape[1]
def seeded_run(mask_unit):
    torch.manual_seed(1); np.random.seed(1)
    if mask_unit is None:
        return run(10, 9000)[:2]
    h = net.register_forward_hook(lambda m, i, o: torch.cat([o[:, :mask_unit], torch.zeros_like(o[:, mask_unit:mask_unit+1]), o[:, mask_unit+1:]], 1))
    try: return run(10, 9000)[:2]
    finally: h.remove()
bf, bw = seeded_run(None)
dF, dW = [], []
for u in range(n_units):
    f, w = seeded_run(u); dF.append(f - bf); dW.append(w - bw)
dF, dW = np.array(dF), np.array(dW)
print("baseline food/water per ep", bf, bw, f"{time.time()-t0:.0f}s", flush=True)
# drive-specific: loses >=25% of that drive's baseline while the other drive changes by <1/3 of that loss
specF = [u for u in range(n_units) if bf > 0 and -dF[u] >= 0.25 * bf and -dW[u] < (-dF[u]) / 3]
specW = [u for u in range(n_units) if bw > 0 and -dW[u] >= 0.25 * bw and -dF[u] < (-dW[u]) / 3]

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
sc = ax[0].scatter(pcs[:, 0], pcs[:, 1], c=diff, cmap="coolwarm", s=4)
plt.colorbar(sc, ax=ax[0], label="energy - hydration")
ax[0].set_xlabel(f"PC1 ({var[0]*100:.0f}%)"); ax[0].set_ylabel(f"PC2 ({var[1]*100:.0f}%)")
ax[0].set_title(f"Last hidden layer PCA (r(PC1, e-h) = {corr:.2f})")
ax[1].scatter(dF, dW, s=14, color="gray")
ax[1].scatter(dF[specF], dW[specF], s=30, color="C3", label="food-specific")
ax[1].scatter(dF[specW], dW[specW], s=30, color="C0", label="water-specific")
ax[1].axhline(0, c="k", lw=.5); ax[1].axvline(0, c="k", lw=.5)
ax[1].set_xlabel("change in food / episode"); ax[1].set_ylabel("change in water / episode")
ax[1].set_title("Single-unit ablation"); ax[1].legend()
plt.tight_layout(); plt.savefig("fig_circuits.png", dpi=150)

res = json.load(open("results.json"))
res["circuits"] = dict(agent_seed=SEED, pc1_var=float(var[0]), pc2_var=float(var[1]), pc1_corr_energy_minus_hydration=corr,
    baseline_food_per_ep=float(bf), baseline_water_per_ep=float(bw), n_units=n_units,
    food_specific_units=specF, water_specific_units=specW,
    delta_food=dF.tolist(), delta_water=dW.tolist(),
    criterion="drop >=25% of own baseline and other drive's drop < 1/3 of it; 10 episodes/unit, noisy",
    max_food_drop=float(-dF.min()), max_water_drop=float(-dW.min()))
json.dump(res, open("results.json", "w"), indent=1)
print(json.dumps({k: v for k, v in res["circuits"].items() if not k.startswith("delta")}), f"{(time.time()-t0)/60:.1f} min")
