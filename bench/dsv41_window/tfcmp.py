"""Compare teacher-forced logits of trees against a reference label."""
import sys, numpy as np
d, ref, *others = sys.argv[1:]
R = np.load(f"{d}/tf/{ref}.npy").astype(np.float64)
def lsm(x): x = x - x.max(-1, keepdims=True); return x - np.log(np.exp(x).sum(-1, keepdims=True))
lr = lsm(R)
for o in others:
    X = np.load(f"{d}/tf/{o}.npy").astype(np.float64); lx = lsm(X)
    kl = (np.exp(lr) * (lr - lx)).sum(-1)
    top = (R.argmax(-1) == X.argmax(-1)).mean()
    print(f"{o:>14} vs {ref}: top1 agree {top:.3f}  KL mean {kl.mean():.2e} max {kl.max():.2e}  max|dlogit| {np.abs(R - X).max():.3g}")
