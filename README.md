# 🎬 Cinema Ticket Demand Forecasting System
### Algions AI Labs Private Limited — AI & Machine Learning Advanced Project Assessment

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Tests Passing](https://img.shields.io/badge/tests-7%20passed-brightgreen.svg)]()
[![Model](https://img.shields.io/badge/Best%20Model-Gradient%20Boosting%20Regressor-orange.svg)]()
[![WAPE](https://img.shields.io/badge/Test%20WAPE-14.82%25-success.svg)]()

---

## 📌 Executive Summary
This project provides an end-to-end, decision-support machine learning solution to forecast show-level cinema ticket sales and seat occupancy for a multiplex cinema. By predicting demand 24 to 48 hours prior to showtime ($T-24\text{h}$ to $T-48\text{h}$), the system assists theater operations in:
1. **Auditorium Screen Allocation:** Upgrading high-demand titles to larger auditoriums (e.g. Screen 1, 250 seats) or downsizing low-demand shows to boutique halls (Screen 3, 120 seats).
2. **Staff Scheduling Optimization:** Recommending optimal staffing levels for concession stands, floor ushers, cleaning crews, and ticket counter clerks to eliminate long queues without labor overspending.

---

## 📂 Repository File Structure
```text
ALGIONS AI ASSIGNMENT/
├── cinema_ticket_demand.csv         # Generated show-level dataset (2,160 records, 180 days)
├── generate_data.py                 # Realistic cinema schedule & synthetic data generator
├── eda_analysis.py                  # Exploratory Data Analysis, quality auditing & leakage checks
├── eda_plots/                       # 4 high-resolution EDA visualization charts
│   ├── 1_demand_by_showtime.png
│   ├── 2_occupancy_by_genre.png
│   ├── 3_weekend_holiday_impact.png
│   └── 4_cold_start_comparison.png
├── model_training.py                # Chronological split, baseline & candidate models trainer
├── cinema_model_artifacts.pkl       # Serialized production pipeline, encoders & benchmarks
├── predict.py                       # CLI inference interface with decision support & drivers
├── app.py                           # Interactive Streamlit decision-support dashboard
├── tests/
│   └── test_pipeline.py             # Pytest suite verifying preprocessing, leakage & inference
├── Technical_Report.md              # Formal 12-section technical report with mathematical details
├── requirements.txt                 # Modern package dependency specifications
└── README.md                        # Project documentation, benchmarks & walkthrough guide
```

---

## 🏆 Held-Out Test Set Performance Benchmark
Evaluated on the held-out test set (final 15% of the chronological timeline, June 3 to June 29, 2025 — 324 shows):

| Model Architecture | Test MAE (tickets) | Test RMSE | Test WAPE (%) | Error Reduction vs Baseline |
| :--- | :---: | :---: | :---: | :---: |
| **Capacity-Adjusted Baseline** | 21.88 | 28.73 | 26.29% | *Baseline reference* |
| **Ridge Regression (L2 Linear)** | 17.18 | 22.00 | 20.64% | +21.5% improvement |
| **Random Forest Regressor** | 14.53 | 20.47 | 17.46% | +33.6% improvement |
| **Gradient Boosting Regressor (Winning)** | **12.33** | **16.90** | **14.82%** | **+43.6% improvement** |

*Evaluation Metric: Weighted Absolute Percentage Error $\text{WAPE} = \frac{\sum |y - \hat{y}|}{\sum y} \times 100\%$.*

### ❄️ Cold-Start vs Mature Shows Error Breakdown
* **Mature Shows ($>3$ days since release, $n=292$):** $\text{MAE} = \mathbf{11.83} \text{ tickets} \mid \text{WAPE} = \mathbf{14.73\%}$
* **Cold-Start Shows ($\le 3$ days since release, $n=32$):** $\text{MAE} = \mathbf{16.94} \text{ tickets} \mid \text{WAPE} = \mathbf{15.38\%}$
* **Operational Risk Strategy:** Newly released titles exhibit higher variance (+43.2% MAE) due to lack of historical word-of-mouth. The system automatically tags cold-start shows, broadens uncertainty margins ($\pm 1.35 \times \text{RMSE}$), and requests human supervisory review.

---

## 🚀 Quickstart & Reproduction Guide

### 1. Environment Setup
Install dependencies into your Python 3 environment:
```bash
pip install -r requirements.txt
```

### 2. Generate Synthetic Dataset
Produces 2,160 show records across 180 days with realistic lifecycle decay curves and zero target leakage:
```bash
python generate_data.py
```

### 3. Run Exploratory Data Analysis & Quality Audit
Validates physical capacity constraints, prints summary statistics, and saves charts to `eda_plots/`:
```bash
python eda_analysis.py
```

### 4. Execute Model Training & Benchmarking Pipeline
Trains the Capacity-Adjusted Baseline and 3 candidate models across chronological Train/Val/Test splits, prints the benchmark table, and serializes `cinema_model_artifacts.pkl`:
```bash
python model_training.py
```

### 5. Run Unit & Acceptance Tests
Executes the test suite covering data integrity, target leakage safeguards, artifact consistency, and inference pipelines:
```bash
pytest tests/
```

### 6. Make a Real CLI Prediction
Loads `cinema_model_artifacts.pkl` and outputs real model predictions, uncertainty ranges, and staffing recommendations:
```bash
python predict.py --genre Action --show_time Evening --day_of_week Saturday --capacity 250
```

### 7. Launch Interactive Streamlit Dashboard
```bash
streamlit run app.py
```

---

## ✅ Acceptance Tests Compliance Checklist

| Test # | Acceptance Test Specification | Evidence & Verification Command | Result |
| :---: | :--- | :--- | :---: |
| **1** | Reproduce dataset split and training run from README | Run `python model_training.py` — reproduces chronological cutoffs and exact metrics | **PASS** |
| **2** | Show baseline and candidate model metrics on held-out test set | Baseline (26.29% WAPE) vs GBR (14.82% WAPE) printed on 324-show held-out test set | **PASS** |
| **3** | Submit real prediction through API/App loading saved model | `python predict.py` loads `cinema_model_artifacts.pkl` and outputs real predictions | **PASS** |
| **4** | Explain prediction, most influential inputs & uncertainty | Output displays top feature importances, 80% confidence interval, and local drivers | **PASS** |
| **5** | Demonstrate no future/target leakage or sensitive data | Programmatic leakage tests pass in `tests/test_pipeline.py`; zero PII collected | **PASS** |

---

## 🛠️ Operational Decision-Support Heuristics

### Screen Allocation Rules
* **High Demand Risk ($\ge 85\%$ Occupancy):** Forecasted demand approaches auditorium limits. Recommend upgrading to Screen 1 (250 seats) or opening an extra late-night show.
* **Low Occupancy Warning ($\le 30\%$ Occupancy):** Substantial excess seats. Recommend downsizing to Screen 3 (120 seats) to minimize utility and cleaning costs.
* **Optimal Allocation ($30\% - 85\%$ Occupancy):** Auditorium capacity comfortably matches demand.

### Staff Scheduling Formulations
$$\text{Concession Staff} = \max\left(1, \left\lceil \frac{\text{Predicted Tickets}}{45} \right\rceil\right)$$
$$\text{Auditorium Ushers / Cleaning} = \max\left(1, \left\lceil \frac{\text{Predicted Tickets}}{60} \right\rceil\right)$$
$$\text{Ticket Counter Clerks} = \max\left(1, \left\lceil \frac{\text{Predicted Tickets}}{55} \right\rceil\right)$$

---

## ⚖️ Responsible AI & Governance
* **Decision Support Only:** This model is designed strictly to aid managerial decision-making, never to execute fully automated staffing or scheduling changes without human oversight.
* **Data Privacy:** Operates entirely on aggregated show-level records; zero personal or customer identifiable data is collected or stored.
* **External Shock Handling:** Unpredictable external events (e.g. extreme storm weather, citywide transport strikes) fall outside historical feature patterns; managers are instructed to apply situational overrides.

---

## 🎓 Technical Walkthrough & Live Debugging Interview Guide

Algions AI Labs will conduct a **15-minute technical walkthrough** and a **10-minute live debugging task**. Here are the key questions you should be ready to answer:

1. **Why chronological splitting instead of random K-Fold?**  
   *Answer:* Cinema demand is inherently time-dependent. Random K-fold causes future target leakage because models train on future sales to predict past shows. Chronological split strictly tests the model on unseen future dates ($T+1$), mirroring real-world deployment.
2. **Why use WAPE alongside MAE and RMSE?**  
   *Answer:* Standard percentage error (MAPE) divides by individual actuals, which explodes when actual ticket sales are small (e.g., 5-10 tickets). WAPE weights total absolute error by total actual sales ($\frac{\sum |y - \hat{y}|}{\sum y}$), giving an interpretable and robust volume-weighted error metric.
3. **How does the system handle the cold-start problem?**  
   *Answer:* For titles in their first 3 days of release, historical title attendance is zero. The system utilizes genre priors and historical timeslot baselines, widens the uncertainty prediction interval by $35\%$, and flags the show with a warning badge for manual review.
4. **How would you debug a high test error on weekend shows?**  
   *Answer:* Check if interaction terms between `is_weekend` and `screen_capacity` are underrepresented, inspect residual distributions across Saturday vs Sunday, and verify if external calendar holidays coincided with the test period.
