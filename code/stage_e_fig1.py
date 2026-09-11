"""Generate Figure 1: methodology / model architecture flowchart."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
                      "savefig.dpi": 300, "savefig.bbox": "tight"})

fig, ax = plt.subplots(figsize=(8.5, 4.3))
ax.set_xlim(0, 100); ax.set_ylim(0, 52)
ax.axis("off")

def box(x, y, w, h, text, fc="#dbe5f1", ec="#3b6fb6", fs=8, bold=False):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3",
                       fc=fc, ec=ec, lw=1.2)
    ax.add_patch(b)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fs, fontweight="bold" if bold else "normal",
            wrap=True)

def arrow(x1, y1, x2, y2, color="#444"):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle="-|>", mutation_scale=11,
                        color=color, lw=1.2)
    ax.add_patch(a)

# Row 1: Multi-sensor data sources
box(2, 44, 18, 6, "Battery\nsensors\n(V, I, T, SoC,\ncycles)", fc="#fde2cf", ec="#d97706")
box(22, 44, 18, 6, "Motor\nsensors\n(temp, vib,\ntorque, RPM)", fc="#fde2cf", ec="#d97706")
box(42, 44, 18, 6, "Brake\nsensors\n(wear, press,\nregen)", fc="#fde2cf", ec="#d97706")
box(62, 44, 18, 6, "Tire &\nsuspension\n(press, temp,\nload)", fc="#fde2cf", ec="#d97706")
box(82, 44, 16, 6, "Operating\nconditions\n(speed, load,\nambient)", fc="#fde2cf", ec="#d97706")

# Row 2: preprocessing
box(20, 33, 60, 6, "Preprocessing:  Min-Max normalisation  →  Sliding window (W = 48 steps ≈ 12 h, stride = 4)",
    fc="#e2efda", ec="#2ca02c", fs=8.5, bold=True)
# arrows from sensors to preprocessing
for x in [11, 31, 51, 71, 90]:
    arrow(x, 44, x, 39.2)

# Row 3: window tensor
box(30, 24, 40, 6, "Multi-sensor window tensor  X ∈ R^(B × 48 × 23)",
    fc="#fff2cc", ec="#bf9000", fs=8.5, bold=True)
arrow(50, 33, 50, 30.2)

# Row 4: LSTM architecture (3 blocks)
box(8, 13, 22, 7, "LSTM layer 1\n(64 units, return seq.)\n+ Dropout 0.2", fc="#dbe5f1", ec="#1f4e79", fs=8)
box(39, 13, 22, 7, "LSTM layer 2\n(32 units)\n+ Dropout 0.2", fc="#dbe5f1", ec="#1f4e79", fs=8)
box(70, 13, 22, 7, "Dense (ReLU, 32)\n→ Dense (1, linear)\nRUL prediction", fc="#dbe5f1", ec="#1f4e79", fs=8)
arrow(30, 24, 19, 20.2)
arrow(30, 24, 50, 20.2)
arrow(30, 24, 81, 20.2)
arrow(30, 16.5, 39, 16.5)
arrow(61, 16.5, 70, 16.5)

# Row 5: output
box(35, 3, 30, 6, "Predicted RUL  (steps)\n+ evaluation (RMSE, MAE, R², MAPE)",
    fc="#f4cccc", ec="#cc0000", fs=8.5, bold=True)
arrow(81, 13, 60, 9.2)

ax.text(50, 50.5, "LSTM-Based Multi-Sensor RUL Prediction Framework for HEV Powertrain",
        ha="center", fontsize=10.5, fontweight="bold")

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig1_methodology.png")
plt.close()
print("Saved fig1_methodology.png")
