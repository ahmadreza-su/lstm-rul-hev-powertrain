"""Stage B v2: baselines on window summary statistics (principled + fast)."""
import os, json, time, warnings, sys
import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)

NP_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
X_train = np.load(f"{NP_DIR}/X_train.npy")   # (8988, 48, 23)
y_train = np.load(f"{NP_DIR}/y_train.npy")
X_test = np.load(f"{NP_DIR}/X_test.npy")
y_test = np.load(f"{NP_DIR}/y_test.npy")
with open(f"{NP_DIR}/meta.json") as f:
    meta = json.load(f)
y_min, y_max = meta["y_min"], meta["y_max"]
def inv(v): return v * (y_max - y_min) + y_min

def window_stats(X):
    # X: (n, T, F) -> (n, F*5)  [mean, std, min, max, last]
    mean = X.mean(axis=1)
    std = X.std(axis=1)
    mn = X.min(axis=1)
    mx = X.max(axis=1)
    last = X[:, -1, :]
    return np.concatenate([mean, std, mn, mx, last], axis=1)

Xtr_s = window_stats(X_train)
Xte_s = window_stats(X_test)
print(f"Window stats: train {Xtr_s.shape}, test {Xte_s.shape}", flush=True)
print(f"Feature stats dim = {Xtr_s.shape[1]} (5 x {X_train.shape[2]})", flush=True)

rng = np.random.RandomState(SEED)
sub = rng.choice(len(Xtr_s), size=min(5000, len(Xtr_s)), replace=False)
print(f"subsample: {len(sub)}", flush=True)

results_file = f"{NP_DIR}/baseline_results.json"
results = {}
yt_full = inv(y_test)

def report(name, y_pred_n):
    yp = inv(y_pred_n); yt = yt_full
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    mae = float(mean_absolute_error(yt, yp))
    r2 = float(r2_score(yt, yp))
    mape = float(np.mean(np.abs((yt - yp) / np.clip(yt, 1e-3, None))) * 100)
    results[name] = dict(rmse=rmse, mae=mae, r2=r2, mape=mape,
                         y_pred=yp.tolist(), y_true=yt.tolist())
    print(f"{name:18s}  RMSE={rmse:7.3f} MAE={mae:7.3f} "
          f"R2={r2:6.4f} MAPE={mape:6.2f}%", flush=True)
    with open(results_file, "w") as f:
        json.dump(results, f)

step = sys.argv[1] if len(sys.argv) > 1 else "all"
if step in ("svr", "all"):
    print("SVR (RBF) ...", flush=True)
    t0 = time.time()
    svr = SVR(kernel="rbf", C=10.0, gamma="scale", cache_size=1000)
    svr.fit(Xtr_s[sub], y_train[sub])
    print(f"  fit {time.time()-t0:.1f}s", flush=True)
    report("SVR (RBF)", svr.predict(Xte_s))

if step in ("rf", "all"):
    print("Random Forest ...", flush=True)
    t0 = time.time()
    rf = RandomForestRegressor(n_estimators=100, max_depth=22, n_jobs=-1,
                                random_state=SEED)
    rf.fit(Xtr_s[sub], y_train[sub])
    print(f"  fit {time.time()-t0:.1f}s", flush=True)
    report("Random Forest", rf.predict(Xte_s))

if step in ("mlp", "all"):
    print("MLP ...", flush=True)
    t0 = time.time()
    mlp = MLPRegressor(hidden_layer_sizes=(128, 64), activation="relu",
                       solver="adam", max_iter=80, random_state=SEED,
                       early_stopping=True)
    mlp.fit(Xtr_s[sub], y_train[sub])
    print(f"  fit {time.time()-t0:.1f}s", flush=True)
    report("MLP", mlp.predict(Xte_s))

print("DONE", flush=True)
