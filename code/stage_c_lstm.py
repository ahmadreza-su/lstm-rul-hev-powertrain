"""Stage C: Train proposed LSTM + CNN-LSTM on derived RUL, save results."""
import os, json, time, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Input, LSTM, Dense, Dropout,
                                     Conv1D, MaxPooling1D)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

NP_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
X_train = np.load(f"{NP_DIR}/X_train.npy")
y_train = np.load(f"{NP_DIR}/y_train.npy")
X_test = np.load(f"{NP_DIR}/X_test.npy")
y_test = np.load(f"{NP_DIR}/y_test.npy")
with open(f"{NP_DIR}/meta.json") as f:
    meta = json.load(f)
y_min, y_max = meta["y_min"], meta["y_max"]
def inv(v): return v * (y_max - y_min) + y_min

print(f"X_train {X_train.shape}  y_train {y_train.shape}")
print(f"X_test  {X_test.shape}  y_test  {y_test.shape}")

# load baseline results
with open(f"{NP_DIR}/baseline_results.json") as f:
    baseline = json.load(f)
results = dict(baseline)
yt = inv(y_test)

def report(name, y_pred_n):
    yp = inv(y_pred_n); 
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    mae = float(mean_absolute_error(yt, yp))
    r2 = float(r2_score(yt, yp))
    mape = float(np.mean(np.abs((yt - yp) / np.clip(yt, 1e-3, None))) * 100)
    results[name] = dict(rmse=rmse, mae=mae, r2=r2, mape=mape,
                         y_pred=yp.tolist(), y_true=yt.tolist())
    print(f"{name:20s}  RMSE={rmse:7.3f} MAE={mae:7.3f} "
          f"R2={r2:6.4f} MAPE={mape:6.2f}%", flush=True)

cb = [EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
      ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5)]

# ---- Proposed LSTM ----
print("\n=== Proposed LSTM ===", flush=True)
lstm = Sequential([
    Input(shape=X_train.shape[1:]),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dense(1, activation="linear"),
])
lstm.compile(optimizer=Adam(1e-3), loss="mse", metrics=["mae"])
lstm.summary()
t0 = time.time()
hist = lstm.fit(X_train, y_train, validation_split=0.15,
                epochs=30, batch_size=256, callbacks=cb, verbose=2)
print(f"LSTM training: {time.time()-t0:.1f}s", flush=True)
yhat = lstm.predict(X_test, verbose=0).ravel()
report("Proposed LSTM", yhat)
hist_lstm = hist.history

# ---- CNN-LSTM ----
print("\n=== CNN-LSTM ===", flush=True)
cnn_lstm = Sequential([
    Input(shape=X_train.shape[1:]),
    Conv1D(32, 3, activation="relu", padding="same"),
    MaxPooling1D(2),
    Conv1D(32, 3, activation="relu", padding="same"),
    MaxPooling1D(2),
    LSTM(48, return_sequences=False),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dense(1, activation="linear"),
])
cnn_lstm.compile(optimizer=Adam(1e-3), loss="mse", metrics=["mae"])
t0 = time.time()
hist2 = cnn_lstm.fit(X_train, y_train, validation_split=0.15,
                     epochs=25, batch_size=256, callbacks=cb, verbose=2)
print(f"CNN-LSTM training: {time.time()-t0:.1f}s", flush=True)
yhat2 = cnn_lstm.predict(X_test, verbose=0).ravel()
report("CNN-LSTM", yhat2)
hist_cnnlstm = hist2.history

# save model + histories
lstm.save(f"{NP_DIR}/lstm_model.keras")
with open(f"{NP_DIR}/lstm_hist.json", "w") as f:
    json.dump(hist_lstm, f)
with open(f"{NP_DIR}/cnnlstm_hist.json", "w") as f:
    json.dump(hist_cnnlstm, f)
with open(f"{NP_DIR}/all_results.json", "w") as f:
    json.dump(results, f)

# ---- Loss curves figure ----
fig, ax = plt.subplots(figsize=(6.5, 3.6))
ax.plot(hist_lstm["loss"], label="LSTM train", color="#1f77b4")
ax.plot(hist_lstm["val_loss"], label="LSTM val", color="#1f77b4", ls="--")
ax.plot(hist_cnnlstm["loss"], label="CNN-LSTM train", color="#d62728")
ax.plot(hist_cnnlstm["val_loss"], label="CNN-LSTM val", color="#d62728", ls="--")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss (MSE, normalised)")
ax.set_title("Training and validation loss")
ax.legend(fontsize=8, ncol=2)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig4_loss_curves.png")
plt.close()
print("Saved fig4_loss_curves.png")

print("\nSTAGE C DONE")
