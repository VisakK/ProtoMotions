import sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import numpy as np
from ps_hold import hold
from bench import CROW, HAND
res = {}
for name, pose in (("crow", CROW), ("handstand", HAND)):
    for N in (128, 256):
        for noise in (0.03, 0.05, 0.08):
            held = []
            tps = []
            for seed in (0, 1, 2):
                arr, tp = hold(pose, T=5.0, N=N, noise=noise, seed=seed)
                ok = (arr[:, 1] > arr[0, 1] - 0.10).all() and arr[:, 2].max() < 30 and arr[:, 3].min() > 0.06
                held.append(ok); tps.append(tp.mean())
            print(f"{name:9s} N={N} noise={noise}: held 5 s on {sum(held)}/3 seeds, replan {1e3*np.mean(tps):.0f} ms", flush=True)
