# K48 Digital Twin Predictive Maintenance

This repository contains a Python-based predictive‐maintenance (PdM) pipeline for a Rieter K48 ring-spinning machine.  
It trains an LSTM model on historical sensor data, then continuously generates new synthetic data, predicts failure probability, updates Tecnomatix live feed, and logs feature‐importance histograms.

---

## 📂 Repository Structure

```

.
├── k48\_history.csv           # Initial historical dataset (timestamp,vibration,temperature,...,failure\_flag)
├── live\_feed.xlsx            # Overwritten each iteration with latest record
├── parameters\_importance.jpg # Last‐saved feature‐importance histogram
├── feature\_importance.csv    # Appended records of feature percentages per run
├── pdm\_pipeline.py           # Main PdM script (train & real-time loop)
└── README.md                 # This file

````

---

## Prerequisites

- **Python 3.8–3.12**  
- **pip** (Python package installer)

Install project dependencies:

```bash
pip install \
  numpy pandas matplotlib \
  scikit-learn tensorflow openpyxl
````

> **Note:**
> * `openpyxl` is required for writing Excel files.
> * Adjust TensorFlow version to match your platform (e.g. `pip install tensorflow==2.12.0`).

---

## Configuration

1. **Historical Data**
   Prepare `k48_history.csv` with columns (in this order):

   ```
   timestamp,vibration,temperature,motor_current,rpm,production_count,failure_flag
   ```

   * **timestamp**: ISO or `DD/MM/YY HH:MM`
   * **failure\_flag**: `0` (normal) or `1` (failure)

2. **Script Parameters**
   All file-paths and key settings live at the top of `pdm_pipeline.py`:

   ```python
   CSV_HISTORY    = "k48_history.csv"
   LIVE_XLSX      = "live_feed.xlsx"
   HISTOGRAM_FILE = "parameters_importance.jpg"
   FEATURE_CSV    = "feature_importance.csv"
   ```

---

##  Usage

1. **Train & run**

   ```
   python pdm_pipeline.py
   ```

   * The script will:

     1. Train an LSTM (up to 500 epochs with early-stopping on `val_loss`).
     2. Plot Epoch vs. Accuracy.
     3. Enter a continuous loop that:

        * Synthesizes a new data point each minute.
        * Predicts failure probability.
        * Writes latest record to `live_feed.xlsx`.
        * Calculates & saves feature-importance histogram (`parameters_importance.jpg`).
        * Logs rankings into `feature_importance.csv`.

2. **Stopping**
   Press **Ctrl+C** to gracefully stop the loop.

---

##  Outputs

* **Epoch vs. Accuracy plot**
  Shown once training completes, with the “optimal epoch” marked.

* **`live_feed.xlsx`**
  Always contains the most recent timestamped record plus:

  * vibration, temperature, motor\_current, rpm, production\_count
  * failure\_prob, alert (0/1), trend

* **`parameters_importance.jpg`**
  A bar chart of the normalized feature importances—overwritten every 3 minutes.

* **`feature_importance.csv`**
  Historical log of feature‐importance percentages with timestamps.

---

## Tuning & Extensions

* **Model architecture** : adjust LSTM units, add Dropout, or swap in 1D-CNN/TCN.
* **Data generation** : tweak noise levels or anomaly frequency.
* **PdM thresholds**: change `failure_prob > 0.5` for alerts.

---


