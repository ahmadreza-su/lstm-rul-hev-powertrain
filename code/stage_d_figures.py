"""Stage D: Generate result figures (pred vs actual, time series, residuals)."""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NP_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
with open(f"{NP_DIR}/all_results.json") as f:
    R = json.load(f)
with open(f"{NP_DIR}/meta.json") as f:
    meta = json.load(f)

# Reconstruct arrays
yp_lstm = np.array(R["Proposed LSTM"]["y_pred"])
yp_mlp  = np.array(R["MLP"]["y_pred"])
yp_svr  = np.array(R["SVR (RBF)"]["y_pred"])
yp_rf   = np.array(R["Random Forest"]["y_pred"])
yp_cl   = np.array(R["CNN-LSTM"]["y_pred"])
yt      = np.array(R["Proposed LSTM"]["y_true"])

print(f"Arrays ready. test N = {len(yt)}")

# ---- Fig 5: predicted vs actual scatter + residuals (LSTM) ----
fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
axes[0].scatter(yt, yp_lstm, s=4, alpha=0.30, color="#1f77b4")
mmax = max(yt.max(), yp_lstm.max()) * 1.05
axes[0].plot([0, mmax], [0, mmax], "r--", lw=1.2, label="ideal y=x")
axes[0].set_xlabel("Actual RUL (steps)")
axes[0].set_ylabel("Predicted RUL (steps)")
axes[0].set_title("(a) LSTM: predicted vs actual")
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.3)
res = yp_lstm - yt
axes[1].hist(res, bins=50, color="#2ca02c", edgecolor="white")
axes[1].axvline(0, color="red", ls="--", lw=1)
axes[1].set_xlabel("Residual (predicted - actual)")
axes[1].set_ylabel("Frequency")
axes[1].set_title(f"(b) LSTM residual (mean={res.mean():.2f}, "
                  f"std={res.std():.2f})")
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig5_pred_vs_actual.png")
plt.close()
print("Saved fig5_pred_vs_actual.png")

# ---- Fig 6: time-series comparison ----
seg = 600
x = np.arange(seg)
fig, ax = plt.subplots(figsize=(8.5, 3.4))
ax.plot(x, yt[:seg], label="Actual RUL", color="black", lw=1.3)
ax.plot(x, yp_lstm[:seg], label="Proposed LSTM", color="#1f77b4", lw=1, alpha=0.9)
ax.plot(x, yp_cl[:seg], label="CNN-LSTM", color="#ff7f0e", lw=1, alpha=0.7)
ax.plot(x, yp_mlp[:seg], label="MLP", color="#2ca02c", lw=1, alpha=0.6)
ax.plot(x, yp_svr[:seg], label="SVR", color="#9467bd", lw=1, alpha=0.5)
ax.set_xlabel("Test sample index (chronological)")
ax.set_ylabel("RUL (15-min steps)")
ax.set_title("RUL prediction on a chronological test segment")
ax.legend(fontsize=8, ncol=5, loc="upper center", framealpha=0.9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig6_timeseries_comparison.png")
plt.close()
print("Saved fig6_timeseries_comparison.png")

# ---- Fig 7: model comparison bar chart (RMSE, MAE, R2, MAPE) ----
models = ["SVR (RBF)", "Random Forest", "MLP", "CNN-LSTM", "Proposed LSTM"]
rmse = [R[m]["rmse"] for m in models]
mae  = [R[m]["mae"]  for m in models]
r2   = [R[m]["r2"]   for m in models]
mape = [R[m]["mape"] for m in models]
fig, axes = plt.subplots(1, 4, figsize=(11, 2.8))
colors = ["#9467bd", "#d62728", "#2ca02c", "#ff7f0e", "#1f77b4"]
short = ["SVR", "RF", "MLP", "CNN-LSTM", "LSTM"]
axes[0].bar(short, rmse, color=colors); axes[0].set_title("RMSE (lower better)")
axes[0].set_ylabel("RMSE (steps)")
axes[1].bar(short, mae, color=colors);  axes[1].set_title("MAE (lower better)")
axes[1].set_ylabel("MAE (steps)")
axes[2].bar(short, r2, color=colors);   axes[2].set_title("R² (higher better)")
axes[2].set_ylabel("R²")
axes[3].bar(short, mape, color=colors); axes[3].set_title("MAPE % (lower better)")
axes[3].set_ylabel("MAPE (%)")
for a in axes:
    a.tick_params(axis="x", labelsize=8, rotation=20)
    a.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig7_model_comparison.png")
plt.close()
print("Saved fig7_model_comparison.png")

# ---- Print summary table ----
print("\n=== FINAL RESULTS TABLE ===")
print(f"{'Model':<20}{'RMSE':>8}{'MAE':>8}{'R2':>8}{'MAPE%':>8}")
for m in models:
    print(f"{m:<20}{R[m]['rmse']:>8.3f}{R[m]['mae']:>8.3f}"
          f"{R[m]['r2']:>8.4f}{R[m]['mape']:>8.2f}")
