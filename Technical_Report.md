# Technical Report: Cinema Ticket Demand Forecasting
### Advanced Machine Learning Project Assessment | Algions AI Labs Private Limited
**Domain:** Cinema Hall Ticket Booking & Operational Resource Allocation  
**Author:** AI & Machine Learning Intern Candidate  
**Date:** November 2026  
**Status:** Complete & Production Ready  

---

## 1. Executive Summary & Problem Framing

### 1.1 Business Context & Decision Objective
Modern cinema multiplex operations face significant revenue and resource optimization challenges. Box-office demand fluctuates heavily depending on movie buzz, day of the week, show timing, calendar holidays, and the release lifecycle. Multiplex managers must make two crucial operational decisions on a daily basis:
1. **Auditorium Screen Allocation:** Deciding which movie should play on which screen (e.g., allocating a high-capacity 250-seat IMAX auditorium vs. a 120-seat boutique screen) to avoid lost revenue from sold-out shows or high energy and overhead costs from near-empty auditoriums.
2. **Staff Scheduling:** Accurately deploying concession stand staff, auditorium ushers, cleaning crews, and box office ticket clerks based on anticipated footfall to prevent long customer queues while avoiding labor overstaffing.

### 1.2 Formal Prediction Unit, Target & Horizon
* **Prediction Unit:** Show-level granularity (`show_id`, or a unique combination of `date`, `screen_id`, and `show_time`).
* **Target Variables:**
  * Primary: `tickets_sold` (Continuous integer target bounded strictly by physical screen capacity: $0 \le \text{tickets\_sold} \le \text{screen\_capacity}$).
  * Secondary (derived): `occupancy_rate` ($\frac{\text{tickets\_sold}}{\text{screen\_capacity}}$).
* **Prediction Horizon:** **T-24h to T-48h** (predictions are generated at 06:00 AM on the day prior to the scheduled show to give operations teams sufficient lead time to adjust screen schedules and labor shifts).
* **Point-in-Time Constraint:** At inference time $T$, the model has access only to historical data strictly prior to $T$. Future sales and same-day target values are strictly unavailable.

### 1.3 Live Cloud Deployment
* **Interactive Web Application:** [https://cinema-ticket-demand-forecasting.streamlit.app/](https://cinema-ticket-demand-forecasting.streamlit.app/)
* Evaluators and operations managers can interactively simulate demand forecasts, test cold-start variance buffers, and view real-time staffing and screen allocation recommendations directly in their web browser.

---

## 2. Dataset Sourcing & Synthetic Generation Methodology

To ensure full reproducibility without compromising proprietary or sensitive audience booking data, a synthetic data generator was developed (`generate_data.py`). The generator reflects realistic physical and behavioral dynamics of multiplex operations.

### 2.1 Theater Infrastructure & Show Schedule
The theater environment models a 3-screen multiplex operating across **180 calendar days** (January 1, 2025 to June 29, 2025):
* **Screen 1 (IMAX / Large Screen):** Capacity = 250 seats.
* **Screen 2 (Standard Main Screen):** Capacity = 180 seats.
* **Screen 3 (Boutique / Gold Screen):** Capacity = 120 seats.

Each screen operates **4 standard daily showtime slots**:
1. Morning (10:30 AM) — Multiplier: $0.60\times$
2. Matinee (02:00 PM) — Multiplier: $0.85\times$
3. Evening (06:00 PM) — Multiplier: $1.35\times$
4. Night (09:30 PM) — Multiplier: $1.50\times$

Total daily shows = $3 \text{ screens} \times 4 \text{ slots} = 12 \text{ shows/day}$. Across 180 days, this yields **2,160 unique show records**, well exceeding the 1,000-record assessment threshold.

### 2.2 Movie Catalog & Staggered Lifecycle
Nine distinct movies spanning 8 diverse genres were scheduled with staggered release dates across the 6-month timeline:
* **Lifecycle Decay Formulation:** Post-release ticket demand follows an exponential decay curve from opening-day buzz:
  $$\text{Buzz}(t) = \text{BaseBuzz} \times \exp(-0.022 \times \max(0, \text{days\_since\_release}))$$
* **Cold-Start Releases:** The first 3 days of a movie's release are explicitly designated as cold-start shows ($\text{days\_since\_release} \le 3$), enabling targeted evaluation of newly released titles.

---

## 3. Exploratory Data Analysis (EDA) & Data Quality Profiling

The full exploratory audit was executed via `eda_analysis.py`, generating key summary statistics and high-resolution visual plots in `eda_plots/`.

### 3.1 Data Quality Audit
| Integrity Check | Evaluated Criterion | Result | Status |
| :--- | :--- | :--- | :--- |
| **Missing Values** | Count of null or NaN values across all columns | 0 | **PASS** |
| **Schedule Collisions** | Duplicate shows at identical (date, screen, slot) | 0 | **PASS** |
| **Capacity Violations** | $\text{tickets\_sold} > \text{screen\_capacity}$ | 0 | **PASS** |
| **Negative Sales** | $\text{tickets\_sold} < 0$ | 0 | **PASS** |
| **Occupancy Boundary** | $\text{occupancy\_rate} \notin [0.0, 1.0]$ | 0 | **PASS** |

### 3.2 Target Distribution Summary
* **Tickets Sold:** Mean = **78.60**, Standard Deviation = **51.10**, Median = **65.0**, Min = **5**, Max = **250**.
* **Occupancy Rate:** Mean = **42.35%**, Standard Deviation = **22.52%**, Median = **38.0%**, Min = **2.0%**, Max = **100.0%**.

### 3.3 Key Segmented Operational Insights
1. **Showtime Impact:** Evening (avg 98.5 tickets, 53.1% occupancy) and Night shows (avg 108.3 tickets, 58.4% occupancy) exhibit more than double the demand of Morning shows (avg 44.5 tickets, 23.9% occupancy).
2. **Weekend Lift:** Weekend shows (Fri-Sun) average **99.5 tickets (53.7% occupancy)** compared to weekday shows (Mon-Thu) at **62.6 tickets (33.7% occupancy)** — representing an average demand surge of **+59.0%**.
3. **Cold-Start Behavior:** Movies during their initial 3 days of release achieve an average occupancy of **58.3%** compared to **40.8%** for mature shows, driven by opening-weekend marketing buzz.
4. **Genre Differentiation:** Action (52.6% avg occupancy), Sci-Fi (46.0%), and Animation (44.6%) drive the highest sustained occupancy rates.

---

## 4. Feature Engineering & Target Leakage Prevention

### 4.1 Feature Dictionary
| Feature Name | Type | Description | Leakage Prevention Safeguard |
| :--- | :--- | :--- | :--- |
| `screen_capacity` | Numeric | Physical seating capacity (120, 180, 250) | Static property known in advance |
| `days_since_release` | Numeric | Days elapsed since movie theatrical premiere | Deterministic calendar difference |
| `is_cold_start` | Binary | 1 if $\text{days\_since\_release} \le 3$, else 0 | Calculated from release date |
| `lag_7d_tickets` | Numeric | Tickets sold for same screen and slot 7 days prior | Strictly references $T - 7 \text{ days}$ |
| `movie_hist_avg_occ` | Numeric | Cumulative avg occupancy of movie up to $T - 1$ | Excludes all shows on date $T$ or later |
| `timeslot_hist_avg_occ`| Numeric | Cumulative avg occupancy for (dow, slot) up to $T - 1$ | Excludes all shows on date $T$ or later |
| `screen_hist_avg_occ` | Numeric | Cumulative avg occupancy of screen up to $T - 1$ | Excludes all shows on date $T$ or later |
| `genre` | Categorical | Movie genre (Action, Sci-Fi, Comedy, etc.) | Static metadata |
| `show_time` | Categorical | Slot label (Morning, Matinee, Evening, Night) | Pre-scheduled slot |
| `day_of_week` | Categorical | Day name (Monday to Sunday) | Calendar feature |
| `is_weekend` | Binary | 1 for Fri/Sat/Sun, 0 for Mon-Thu | Calendar feature |
| `is_holiday` | Binary | 1 for gazetted holiday, 0 otherwise | Calendar feature |

### 4.2 Rigorous Leakage Audit
A common pitfall in ML assessments is computing rolling averages or global group aggregations over the entire dataset, which inadvertently leaks future sales information into past rows. In this implementation:
* All historical rolling metrics (`movie_hist_avg_occ`, `timeslot_hist_avg_occ`, `screen_hist_avg_occ`) are computed strictly using shows occurring before the current date (`day_idx < cur_day_idx`).
* For Day 0 and new movie releases, informative **domain priors** (e.g., genre baseline expectations) are used as fallback anchors, preventing any future lookahead.

---

## 5. Chronological Validation Strategy

Standard random K-Fold cross-validation is fundamentally flawed for time-dependent demand forecasting because it leaks future temporal patterns into training folds. To adhere to Algions AI Labs guidelines, a **strict chronological split** was applied across the 180 days:

```
[---------- 70% TRAIN SET ----------] [--- 15% VAL SET ---] [--- 15% TEST SET ---]
2025-01-01 to 2025-05-05              2025-05-06 to 2025-06-02 2025-06-03 to 2025-06-29
1,500 shows (125 days)                336 shows (28 days)     324 shows (27 days)
```

The **Held-Out Test Set** (June 3 to June 29, 2025) was completely locked and held out during hyperparameter exploration, serving strictly for final acceptance verification.

---

## 6. Model Architectures & Benchmarking

Four distinct modeling approaches were implemented and compared:

### 6.1 Capacity-Adjusted Historical Baseline
As required by the assignment brief, a simple yet sensible operational baseline was constructed:
$$\hat{y}_{\text{baseline}} = \text{ScreenCapacity} \times \bar{\text{Occupancy}}_{\text{train}}(\text{day\_of\_week}, \text{show\_time})$$
Where $\bar{\text{Occupancy}}_{\text{train}}$ is the historical average occupancy for that specific day and time slot computed strictly on the training set.

### 6.2 Candidate Model 1: Ridge Regression (L2 Linear)
A standardized, L2-regularized linear model ($\alpha = 10.0$) with one-hot encoded categorical features. This establishes whether linear relationships between lagged sales, capacity, and time slots are sufficient.

### 6.3 Candidate Model 2: Random Forest Regressor
An ensemble of 150 regression trees (`max_depth=12`, `min_samples_leaf=2`). Random forests naturally capture non-linear interactions between auditorium capacity, movie buzz, and weekend multipliers.

### 6.4 Candidate Model 3: Gradient Boosting Regressor (Winning Model)
A sequential gradient boosted tree ensemble (150 estimators, learning rate = 0.08, `max_depth=5`, subsampling ratio = 0.85). Gradient boosting optimizes directly over the residuals of previous trees, providing sharper adaptation to demand shifts.

---

## 7. Experimental Results & Held-Out Test Evaluation

All models were evaluated on the held-out test set (324 shows) using standard business metrics:
* **MAE (Mean Absolute Error):** Average ticket misforecast per show.
* **RMSE (Root Mean Squared Error):** Penalizes large misforecasts that disrupt theater staffing.
* **WAPE (Weighted Absolute Percentage Error):**
  $$\text{WAPE} = \frac{\sum_{i=1}^{N} |y_i - \hat{y}_i|}{\sum_{i=1}^{N} y_i} \times 100\%$$

### 7.1 Comprehensive Benchmark Table (Acceptance Test 2)
| Model Architecture | Val MAE | Val RMSE | Val WAPE (%) | Test MAE | Test RMSE | Test WAPE (%) | Error Reduction |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Capacity-Adjusted Baseline** | 16.10 | 21.17 | 22.75% | 21.88 | 28.73 | 26.29% | Baseline |
| **Ridge Regression (L2 Linear)** | 17.46 | 21.41 | 24.67% | 17.18 | 22.00 | 20.64% | +21.5% |
| **Random Forest Regressor** | 13.34 | 17.87 | 18.85% | 14.53 | 20.47 | 17.46% | +33.6% |
| **Gradient Boosting Regressor** | **12.11** | **16.71** | **17.11%** | **12.33** | **16.90** | **14.82%** | **+43.6%** |

**Conclusion:** The **Gradient Boosting Regressor** achieved superior accuracy across all metrics, reducing forecasting error by **43.6% over the historical baseline** and achieving a Test WAPE of **14.82%**.

---

## 8. Cold-Start & Low-Data Show Error Analysis

The assignment specifically requires evaluating model performance on newly released movies and low-data shows. On the held-out test set, shows were segmented into Mature Shows vs. Cold-Start Shows ($\text{days\_since\_release} \le 3$):

| Cohort | Shows ($n$) | Test MAE | Test RMSE | Test WAPE (%) | Operational Risk Profile |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Mature Shows (> 3 Days)** | 292 | **11.83** | **16.48** | **14.73%** | Low; stable box-office momentum |
| **Cold-Start Shows ($\le$ 3 Days)** | 32 | **16.94** | **20.31** | **15.38%** | Moderate-High; unobserved word-of-mouth |

### 8.1 Analysis of Cold-Start Discrepancy
Cold-start shows exhibit approximately **43.2% higher MAE** (16.94 vs. 11.83 tickets). This occurs because brand-new releases lack historical attendance signals and rely primarily on genre priors, screen capacity, and initial buzz estimates. To safeguard operations:
1. The inference interface surfaces a **Cold-Start Alert badge** for titles in their first 3 days of release.
2. The uncertainty range is widened by $35\%$ ($\pm 1.35 \times \text{RMSE}$) to communicate higher variance to management.
3. Theater managers are instructed to review early online ticket reservations ($T-6\text{h}$) before locking final staff schedules.

---

## 9. Model Explainability & Influential Input Analysis

Feature importances extracted from the winning Gradient Boosting ensemble reveal the primary drivers of cinema ticket demand:

| Rank | Feature | Importance | Operational Interpretation |
| :---: | :--- | :---: | :--- |
| **1** | `lag_7d_tickets` | **72.0%** | Habitual attendance patterns: last week's sales for the same slot serve as the primary demand anchor. |
| **2** | `days_since_release` | **10.0%** | Theatrical lifecycle: demand decays steadily with each passing week since premiere. |
| **3** | `timeslot_hist_avg_occ`| **4.8%** | Time slot dynamics: Evening/Night slots consistently drive higher occupancy than mornings. |
| **4** | `screen_capacity` | **2.9%** | Physical capacity ceiling constraining total sales potential. |
| **5** | `screen_hist_avg_occ` | **2.8%** | Auditorium preference and screen quality index. |
| **6** | `movie_hist_avg_occ` | **2.2%** | Sustained title popularity and word-of-mouth strength. |
| **7** | `genre_Action` | **1.2%** | Genre-specific demand boost for high-energy blockbusters. |
| **8** | `is_holiday_1` | **0.8%** | Public holiday footfall surge. |

---

## 10. Operational Decision-Support Framework

The model is intentionally designed as a **decision-support tool** to assist human managers rather than an autonomous decision maker.

### 10.1 Screen Allocation Optimization Heuristics
* **High Demand Risk ($\ge 85\%$ Occupancy):** Projected sales are constrained by screen capacity. Recommendation: Upgrade to a higher-capacity screen (e.g., move from Screen 2 to Screen 1) or open a simultaneous parallel screening.
* **Low Occupancy Warning ($\le 30\%$ Occupancy):** Significant empty capacity. Recommendation: Downsize to Screen 3 (120 seats) to conserve energy and cleaning overhead, or run promotional combo packages.
* **Optimal Range ($30\% - 85\%$ Occupancy):** Efficient balance of auditorium utilization.

### 10.2 Staff Scheduling Rules of Thumb
To translate ticket predictions into actionable labor scheduling:
$$\text{Concession Staff} = \max\left(1, \left\lceil \frac{\hat{y}}{45} \right\rceil\right)$$
$$\text{Auditorium Ushers} = \max\left(1, \left\lceil \frac{\hat{y}}{60} \right\rceil\right)$$
$$\text{Ticket Counter Clerks} = \max\left(1, \left\lceil \frac{\hat{y}}{55} \right\rceil\right)$$

---

## 11. Responsible AI, Limitations & Ethical Considerations

### 11.1 Human-in-the-Loop Governance
The model is strictly non-autonomous. Staff schedules and screen alterations must be signed off by duty managers. Model forecasts should inform, not dictate, workforce scheduling.

### 11.2 Privacy by Design
The pipeline operates strictly on aggregated show-level metrics (`tickets_sold`, `screen_id`, `show_time`). Zero Personally Identifiable Information (PII) or customer demographic attributes are collected, stored, or processed, ensuring full compliance with modern data privacy regulations.

### 11.3 Residual Error & External Shock Vulnerabilities
The model cannot foresee unmodeled external anomalies, including:
* Severe weather disturbances (flash flooding, severe blizzards).
* Local transit disruptions or regional public curfews.
* Sudden viral social media trends or negative boycotts.

During known external events, managers are trained to discount the model's output in favor of on-the-ground situational awareness.

---

## 12. Verification of Acceptance Tests

The submission package directly fulfills all 5 mandatory Acceptance Tests detailed in the Algions AI Labs assessment brief:

1. **Reproduce Dataset Split & Training Run:** Executing `python model_training.py` trains all 4 models, prints chronological split cutoffs, and displays the full benchmark comparison table.
2. **Show Baseline & Candidate Metrics on Held-Out Test Set:** Evaluated on June 3 - June 29 test set: Baseline (WAPE: 26.29%), Ridge (WAPE: 20.64%), Random Forest (WAPE: 17.46%), Gradient Boosting (WAPE: 14.82%).
3. **Submit Real Prediction Through API/App:** `python predict.py` and the Streamlit web app ([Live Demo](https://cinema-ticket-demand-forecasting.streamlit.app/) / local `streamlit run app.py`) dynamically load `cinema_model_artifacts.pkl` and output real model predictions with uncertainty bounds.
4. **Explain Prediction, Influential Inputs & Limitations:** The CLI and web app surface the top 4 local drivers, global feature importances, cold-start warnings, and 80% confidence intervals.
5. **Demonstrate No Future Leakage or Sensitive Data:** Verified via automated unit tests in `tests/test_pipeline.py` (`7 passed in 5.36s`) and programmatic leakage checks in `eda_analysis.py`.

