# LSTM-Based Remaining Useful Life Prediction for HEV Powertrain Components Using Multi-Sensor Time Series Data

This repository contains the complete Python pipeline accompanying the paper
*"LSTM-Based Remaining Useful Life Prediction for Hybrid Electric Vehicle
Powertrain Components Using Multi-Sensor Time Series Data"* submitted to the
Journal of Engine Research (Iranian Society of Engine).

## Abstract

Remaining useful life (RUL) prediction of hybrid electric vehicle (HEV)
powertrain components is essential for proactive maintenance and fleet
reliability. This work proposes a two-layer long short-term memory (LSTM)
recurrent neural network for RUL prediction that ingests a twenty-three-channel
multi-sensor time-series stream spanning the battery, electric motor, brake,
tire-and-suspension, and operating-condition sub-systems of an HEV powertrain.
Because the raw RUL field of the dataset exhibited no statistical relationship
to the sensor channels, a physically-motivated composite health index is
derived from the real sensor stream through a nonlinear cumulative-damage
recurrence whose coefficients are grounded in physically meaningful
time-constants. On a chronological test set of 4,488 windows, the proposed
LSTM attains an R² of 0.494, an RMSE of 3.37 steps (~50 min of operation), and
a MAPE of 4.45%, outperforming SVR, Random Forest, MLP and CNN-LSTM baselines
by 19% in RMSE and more than doubling the R².

## Repository structure

```
.
├── code/                       # Python pipeline (5 stages)
│   ├── stage_a2_construct_rul.py    # Data prep + composite health-index RUL construction
│   ├── stage_b_baselines.py         # SVR / Random Forest / MLP baselines
│   ├── stage_c_lstm.py              # Proposed LSTM + CNN-LSTM training & evaluation
│   ├── stage_d_figures.py           # Result figures (loss, scatter, residual, time series)
│   └── stage_e_fig1.py             # Methodology flowchart
├── figures/                    # Generated publication figures
│   ├── fig1_methodology.png
│   ├── fig2_data_overview.png
│   ├── fig3_correlation_heatmap.png
│   ├── fig4_loss_curves.png
│   ├── fig5_pred_vs_actual.png
│   ├── fig6_timeseries_comparison.png
│   └── fig7_model_comparison.png
├── results/
│   └── metrics_summary.json    # Final test metrics for all 5 models
├── data/
│   └── README.md               # Instructions to obtain the dataset
├── requirements.txt
├── LICENSE
└── README.md
```

## Requirements

- Python 3.9+
- See `requirements.txt` for the full list of dependencies.

Install with:

```bash
pip install -r requirements.txt
```

## Dataset

The pipeline expects the file
`data/EV_Predictive_Maintenance_Dataset_15min.csv` (175,393 records, 30 columns,
sampled every 15 minutes from a hybrid electric vehicle over January 2020 –
January 2025). The dataset is publicly available on Kaggle:

> **EV IoT Predictive Maintenance Dataset**
> https://www.kaggle.com/datasets/datasetengineer/eviot-predictivemaint-dataset

It is not redistributed in this repository due to its size. Download the CSV
from the Kaggle link above and place it in the `data/` directory before
running the pipeline (see `data/README.md` for details).

## Reproducing the results

Run the pipeline stages in order from the repository root:

```bash
python code/stage_a2_construct_rul.py   # builds windows, saves to results/
python code/stage_b_baselines.py all    # SVR + Random Forest + MLP
python code/stage_c_lstm.py             # proposed LSTM + CNN-LSTM
python code/stage_d_figures.py          # result figures
python code/stage_e_fig1.py             # methodology flowchart
```

After Stage A, the intermediate arrays (`X_train.npy`, `y_train.npy`,
`X_test.npy`, `y_test.npy`, `CD.npy`, `stress_rate.npy`, `meta.json`) are
written to `results/`. Stages B–D read these arrays, so they must be run
after Stage A.

## Results

The final test-set metrics (4,488 windows, chronological split) are stored in
`results/metrics_summary.json`:

| Model            | RMSE  | MAE   | R²     | MAPE (%) |
|------------------|-------|-------|--------|----------|
| SVR (RBF)        | 4.847 | 3.866 | -0.045 | 6.42     |
| Random Forest    | 4.327 | 3.274 | 0.167  | 5.60     |
| MLP              | 4.267 | 3.185 | 0.190  | 5.50     |
| CNN-LSTM         | 4.150 | 3.443 | 0.234  | 5.66     |
| **Proposed LSTM**| **3.372** | **2.685** | **0.494** | **4.45** |

The proposed LSTM outperforms the strongest baseline by 19% in RMSE and more
than doubles the R².

## Methodology summary

1. **Multi-sensor feature taxonomy** — 23 channels grouped into 5 powertrain
   sub-systems (battery, motor, brake, tire/suspension, operating/ambient).
2. **Composite health-index RUL target** — a nonlinear cumulative-damage
   recurrence converts the real sensor stream into a window-observable RUL
   target, since the raw RUL field of the dataset carries no exploitable
   sensor correlation.
3. **Sliding-window construction** — W = 48 steps (~12 h), stride = 4,
   chronological train/test split (no temporal leakage).
4. **Two-layer LSTM** — 64 + 32 units, dropout 0.2, Adam optimiser, early
   stopping; 36,033 trainable parameters (compact for embedded-BMS
   deployability).

## Limitations (stated in the paper)

- The RUL target is a reconstructed engineering proxy, not a directly
  measured failure label.
- The achieved R² of 0.494 is below the threshold typically required for
  autonomous proactive-maintenance deployment (R² > 0.85).
- Validation against a physical run-to-failure benchmark (NASA PCoE, etc.)
  is identified as the necessary next step.

## License

MIT License — see `LICENSE`.

## Citation

If you use this code, please cite the paper:

```bibtex
@article{lstm_rul_hev_2026,
  title   = {LSTM-Based Remaining Useful Life Prediction for Hybrid Electric
             Vehicle Powertrain Components Using Multi-Sensor Time Series Data},
  journal = {Journal of Engine Research},
  year    = {2026},
  note    = {Iranian Society of Engine}
}
```
