# Grokking on modular addition: (a + b) mod P with a small MLP.
# This is the one file the research agent may edit. Wall-clock budget is fixed by the harness.
# Last line printed: RESULT {...}. Metric: grok_step = first step with test accuracy >= 95% (lower is better; null = did not generalise within the budget).
import json, os, time
from functools import partial
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

MAX_STEPS = int(os.environ.get("MAX_STEPS", 8000))  # fixed step budget set by the harness
BUDGET_S = float(os.environ.get("BUDGET_S", 300))   # safety cap only
SEED = int(os.environ.get("SEED", 0))

# ---- experiment settings ----
P = 97
TRAIN_FRAC = 0.4
EMBED_DIM = 128
HIDDEN = 512
LR = 1e-3
WEIGHT_DECAY = 0.01
EVAL_EVERY = 200
# -----------------------------

mx.random.seed(SEED)
a = mx.repeat(mx.arange(P), P)
b = mx.tile(mx.arange(P), P)
X = mx.stack([a, b], axis=1)
Y = (a + b) % P
perm = mx.random.permutation(P * P)
n_train = int(TRAIN_FRAC * P * P)
Xtr, Ytr, Xte, Yte = X[perm[:n_train]], Y[perm[:n_train]], X[perm[n_train:]], Y[perm[n_train:]]
mx.eval(Xtr, Ytr, Xte, Yte)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(P, EMBED_DIM)
        self.mlp = nn.Sequential(nn.Linear(2 * EMBED_DIM, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, P))
    def __call__(self, x):
        return self.mlp(self.embed(x).reshape(x.shape[0], -1))

model = Model()
opt = optim.AdamW(learning_rate=LR, weight_decay=WEIGHT_DECAY)
mx.eval(model.parameters())
state = [model.state, opt.state]

def loss_fn(m, x, y):
    return nn.losses.cross_entropy(m(x), y).mean()

@partial(mx.compile, inputs=state, outputs=state)
def train_step(x, y):
    loss, grads = nn.value_and_grad(model, loss_fn)(model, x, y)
    opt.update(model, grads)
    return loss

def accuracy(x, y):
    return (mx.argmax(model(x), axis=-1) == y).mean().item()

start, step, grok_step, curve = time.time(), 0, None, []
while step < MAX_STEPS and time.time() - start < BUDGET_S:
    loss = train_step(Xtr, Ytr)
    mx.eval(state)
    step += 1
    if step % EVAL_EVERY == 0:
        point = {"step": step, "t": round(time.time() - start, 2), "train_acc": round(accuracy(Xtr, Ytr), 4), "test_acc": round(accuracy(Xte, Yte), 4), "loss": round(loss.item(), 5)}
        curve.append(point)
        print(json.dumps(point), flush=True)
        if grok_step is None and point["test_acc"] >= 0.95:
            grok_step = step

json.dump(curve, open(os.environ.get("CURVE_OUT", "curve.json"), "w"))
print("RESULT " + json.dumps({"test_acc": round(accuracy(Xte, Yte), 4), "train_acc": round(accuracy(Xtr, Ytr), 4), "grok_step": grok_step, "steps": step, "seconds": round(time.time() - start, 1)}))
