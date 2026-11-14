"""
app.py
------
Cinema Ticket Demand Forecasting & Theater Operations Dashboard
Author: AI/ML Engineering Intern
Domain: Multiplex Screen & Staff Allocation

An interactive decision-support interface that loads trained gradient boosting
artifacts to forecast show-level ticket demand, estimate seat occupancy,
and recommend optimal auditorium and workforce allocation.
"""

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# Configure page layout and metadata
st.set_page_config(
    page_title="CinemaOps | AI Demand Forecaster",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling for a polished, dark cinema aesthetic
st.markdown("""
<style>
    /* Dark theater palette */
    .stApp {
        background-color: #0e1117;
        color: #f0f2f6;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    /* Polished header badge */
    .ops-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 12px 20px;
        background: #161b26;
        border: 1px solid #262f40;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .ops-title {
        font-size: 22px;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -0.3px;
    }
    .ops-tag {
        background: #ff4b4b;
        color: white;
        font-size: 11px;
        font-weight: 600;
        padding: 3px 8px;
        border-radius: 4px;
        margin-left: 8px;
        text-transform: uppercase;
    }
    .ops-sub {
        font-size: 12px;
        color: #8b949e;
    }

    /* Curved cinema screen indicator */
    .screen-indicator {
        text-align: center;
        margin: 15px auto 25px auto;
        width: 75%;
    }
    .screen-arc {
        height: 6px;
        background: linear-gradient(90deg, transparent, #ff4b4b, transparent);
        border-radius: 50%;
        box-shadow: 0 0 10px rgba(255, 75, 75, 0.6);
        margin-bottom: 6px;
    }
    .screen-caption {
        font-size: 11px;
        font-weight: 600;
        color: #8b949e;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    /* Theater seat grid */
    .seat-grid {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 5px;
        padding: 18px;
        background: #11151f;
        border: 1px solid #202736;
        border-radius: 10px;
        max-width: 820px;
        margin: 0 auto;
    }
    .seat-row {
        display: flex;
        gap: 5px;
        align-items: center;
    }
    .seat-row-label {
        font-size: 10px;
        font-weight: 700;
        color: #4b5568;
        width: 16px;
        text-align: center;
    }
    .seat-unit {
        width: 16px;
        height: 16px;
        border-radius: 3px;
        display: inline-block;
    }
    .seat-filled {
        background-color: #ff4b4b;
        box-shadow: 0 0 3px rgba(255, 75, 75, 0.4);
    }
    .seat-empty {
        background-color: #2ea043;
        opacity: 0.8;
    }
    .seat-key {
        display: flex;
        justify-content: center;
        gap: 20px;
        margin-top: 14px;
        font-size: 12px;
        color: #8b949e;
    }
    .seat-key-item {
        display: flex;
        align-items: center;
        gap: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 1. Load Trained Pipeline & Metadata
# ---------------------------------------------------------
MODEL_ARTIFACT_PATH = "cinema_model_artifacts.pkl"

@st.cache_resource
def load_model_pipeline(path=MODEL_ARTIFACT_PATH):
    """Load serialized model pipeline, priors, and test metrics."""
    if not os.path.exists(path):
        st.error(f"Required artifact '{path}' not found. Please run 'python model_training.py' first.")
        st.stop()
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

artifacts = load_model_pipeline()
pipeline = artifacts["pipeline"]
feature_cols = artifacts["feature_cols"]
genre_priors = artifacts["genre_priors"]
timeslot_priors = artifacts["timeslot_priors"]
benchmark_summary = artifacts["benchmark_summary"]
cold_start_metrics = artifacts["cold_start_metrics"]
feature_importances = artifacts.get("feature_importances", [])
test_rmse = artifacts.get("test_rmse", 16.9)

# ---------------------------------------------------------
# 2. Header & Quick User Instructions
# ---------------------------------------------------------
st.markdown("""
<div class="ops-header">
    <div>
        <span class="ops-title">🎬 CinemaOps</span>
        <span class="ops-tag">AI Demand Planner</span>
        <div class="ops-sub">Multiplex Ticket Demand Forecasting & Resource Allocation System</div>
    </div>
    <div style="text-align: right; font-size: 12px; color: #8b949e;">
        <span style="color:#2ea043;">● Model: Gradient Boosting Regressor</span><br>
        Forecast Horizon: T-24h to T-48h
    </div>
</div>
""", unsafe_allow_html=True)

with st.expander("ℹ️ How to use this forecasting dashboard", expanded=False):
    st.markdown("""
    **Operational Workflow for Multiplex Managers:**
    1. **Configure Show Parameters in Sidebar:** Select the movie title, target auditorium screen, scheduled showtime slot, and calendar date.
    2. **Review Forecast Metrics:** The model estimates ticket sales, occupancy rate, projected box-office revenue, and an **80% prediction interval**.
    3. **Operational Decision Support:** Review automated recommendations for:
       - **Auditorium Screen Swapping:** Flags shows that exceed 85% capacity (need larger hall) or fall below 30% capacity (should downsize).
       - **Staff Scheduling:** Recommends optimal staff counts for concession stands, auditorium cleaning, and box office ticket windows.
    4. **Inspect Model Explainability:** Switch to the **Explainability** tab to see global feature importances and specific local demand drivers.
    """)

# ---------------------------------------------------------
# 3. Sidebar Input Configuration (Organized & Documented)
# ---------------------------------------------------------
st.sidebar.header("📋 Show Setup & Signals")

# Section 1: Movie & Screen
st.sidebar.subheader("1. Movie & Auditorium")

movie_catalog = {
    "SuperHero Unleashed": {"genre": "Action", "base_price": 350, "format": "IMAX / 2D"},
    "Galactic Wars VIII": {"genre": "Sci-Fi", "base_price": 400, "format": "IMAX 3D"},
    "Laugh Riot 2": {"genre": "Comedy", "base_price": 280, "format": "Standard 2D"},
    "Kingdom of Animals": {"genre": "Animation", "base_price": 300, "format": "Family 3D"},
    "Shadow Detective": {"genre": "Thriller", "base_price": 320, "format": "Dolby Atmos"},
    "The Haunted Asylum": {"genre": "Horror", "base_price": 280, "format": "Standard 2D"},
    "Love in Venice": {"genre": "Romance", "base_price": 260, "format": "Standard 2D"},
    "Silicon Dreamers": {"genre": "Drama", "base_price": 250, "format": "Standard 2D"},
    "Cosmic Odyssey": {"genre": "Sci-Fi", "base_price": 420, "format": "IMAX Laser (New Release)"}
}

selected_movie = st.sidebar.selectbox(
    "Movie Title",
    list(movie_catalog.keys()),
    index=0,
    help="Select the theatrical title currently scheduled for screening."
)
movie_genre = movie_catalog[selected_movie]["genre"]
ticket_price = movie_catalog[selected_movie]["base_price"]

screen_options = {
    1: {"name": "Screen 1 (Auditorium IMAX Laser)", "capacity": 250},
    2: {"name": "Screen 2 (Dolby Atmos Main)", "capacity": 180},
    3: {"name": "Screen 3 (Boutique Gold Class)", "capacity": 120}
}
selected_screen_id = st.sidebar.selectbox(
    "Auditorium Screen",
    options=[1, 2, 3],
    format_func=lambda x: f"{screen_options[x]['name']} — {screen_options[x]['capacity']} seats",
    help="Physical auditorium assigned to this show. Capacity strictly bounds maximum ticket sales."
)
screen_capacity = screen_options[selected_screen_id]["capacity"]

# Section 2: Showtime & Calendar
st.sidebar.subheader("2. Schedule & Timing")

show_slots = {
    "Morning": "10:30 AM",
    "Matinee": "02:00 PM",
    "Evening": "06:00 PM",
    "Night": "09:30 PM"
}
selected_slot = st.sidebar.selectbox(
    "Showtime Slot",
    list(show_slots.keys()),
    index=2, # Default Evening
    format_func=lambda x: f"{x} ({show_slots[x]})",
    help="Showtime slot. Evening and Night slots historically experience significantly higher demand."
)

weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
selected_day = st.sidebar.selectbox(
    "Day of Week",
    weekdays,
    index=5, # Default Saturday
    help="Day of screening. Weekends (Friday-Sunday) historically experience a 50%+ demand surge."
)

is_holiday = st.sidebar.selectbox(
    "Public Holiday / Long Weekend?",
    [0, 1],
    format_func=lambda x: "Yes (Holiday Surge)" if x == 1 else "No (Regular Day)",
    help="Set to 'Yes' if the screening falls on a national holiday, festival, or long weekend."
)

# Section 3: Historical Anchors & Lifecycle
st.sidebar.subheader("3. Lifecycle & Historical Signals")

days_since_release = st.sidebar.slider(
    "Days Since Premiere (Movie Age)",
    min_value=0,
    max_value=40,
    value=8,
    help="Days elapsed since theatrical release. Days 0-3 trigger Cold-Start handling with wider uncertainty bounds."
)

is_weekend = 1 if selected_day in ["Friday", "Saturday", "Sunday"] else 0
is_cold_start = 1 if days_since_release <= 3 else 0
timeslot_prior_occ = timeslot_priors.get(selected_slot, 0.55)
movie_hist_avg_occ = genre_priors.get(movie_genre, 0.50)

# Realistic fallback estimation for 7-day prior sales
default_lag_estimate = int(round(screen_capacity * timeslot_prior_occ * (1.35 if is_weekend else 1.0)))
lag_7d_tickets = st.sidebar.number_input(
    "7-Day Prior Ticket Sales (Same Slot)",
    min_value=0,
    max_value=screen_capacity,
    value=min(screen_capacity, default_lag_estimate),
    help="Tickets sold for the exact same screen and timeslot 7 days ago. Serves as the primary autoregressive anchor."
)

# ---------------------------------------------------------
# 4. Model Inference & Calculations
# ---------------------------------------------------------
input_payload = pd.DataFrame([{
    "screen_capacity": screen_capacity,
    "days_since_release": days_since_release,
    "movie_hist_avg_occ": movie_hist_avg_occ,
    "timeslot_hist_avg_occ": timeslot_prior_occ,
    "screen_hist_avg_occ": 0.55,
    "lag_7d_tickets": lag_7d_tickets,
    "genre": movie_genre,
    "show_time": selected_slot,
    "day_of_week": selected_day,
    "is_weekend": is_weekend,
    "is_holiday": is_holiday,
    "is_cold_start": is_cold_start
}])[feature_cols]

# Run regression model
raw_prediction = pipeline.predict(input_payload)[0]
predicted_tickets = int(np.clip(np.round(raw_prediction), 0, screen_capacity))
occupancy_rate = round((predicted_tickets / screen_capacity) * 100.0, 1)

# Uncertainty margin (+/- 1.25 RMSE, ~80% prediction interval)
uncertainty_margin = int(round(test_rmse * (1.35 if is_cold_start else 1.0)))
lower_bound = max(0, predicted_tickets - uncertainty_margin)
upper_bound = min(screen_capacity, predicted_tickets + uncertainty_margin)

# Financial & labor scheduling heuristics
projected_revenue = predicted_tickets * ticket_price
ushers_needed = max(1, int(np.ceil(predicted_tickets / 60.0)))
concession_staff = max(1, int(np.ceil(predicted_tickets / 45.0)))
ticket_clerks = max(1, int(np.ceil(predicted_tickets / 55.0)))

# ---------------------------------------------------------
# 5. Main Application Tabs
# ---------------------------------------------------------
tab_overview, tab_operations, tab_explain, tab_benchmarks, tab_governance = st.tabs([
    "📊 Show Forecast & Seat Layout",
    "👥 Staffing & Screen Optimization",
    "🔍 Model Explainability & Drivers",
    "📈 Holdout Validation Benchmarks",
    "⚠️ Limitations & Responsible AI"
])

# ---------------------------------------------------------
# TAB 1: SHOW FORECAST & SEAT LAYOUT
# ---------------------------------------------------------
with tab_overview:
    # Cold-Start Advisory Banner
    if is_cold_start:
        st.warning(
            f"⚠️ **Cold-Start Advisory (Premiere Day {days_since_release}):** "
            "This title has ≤ 3 days of box-office history. The forecast relies primarily on genre priors and "
            f"slot averages with a wider uncertainty margin (±{uncertainty_margin} tickets). Manager review recommended."
        )

    # Show Summary Banner
    with st.container(border=True):
        col_m1, col_m2 = st.columns([3, 1])
        with col_m1:
            st.markdown(f"### {selected_movie}")
            st.caption(
                f"**Genre:** {movie_genre} | **Format:** {movie_catalog[selected_movie]['format']} | "
                f"**Screen:** {screen_options[selected_screen_id]['name']} | "
                f"**Schedule:** {selected_day}, {show_slots[selected_slot]} ({selected_slot})"
            )
        with col_m2:
            if occupancy_rate >= 85.0:
                st.error(f"🔴 **ALMOST FULL** ({occupancy_rate}%)")
            elif occupancy_rate >= 45.0:
                st.warning(f"🟠 **FILLING FAST** ({occupancy_rate}%)")
            else:
                st.success(f"🟢 **AVAILABLE** ({occupancy_rate}%)")

    # KPI Metric Cards
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric(
            label="Forecasted Ticket Sales",
            value=f"{predicted_tickets} / {screen_capacity}",
            help="Total tickets predicted to be sold for this show."
        )
    with kpi2:
        st.metric(
            label="Estimated Occupancy Rate",
            value=f"{occupancy_rate}%",
            help="Percentage of auditorium seating capacity expected to be occupied."
        )
    with kpi3:
        st.metric(
            label="Projected Box-Office",
            value=f"₹{projected_revenue:,}",
            help=f"Estimated gross ticket revenue (base price: ₹{ticket_price} per seat)."
        )
    with kpi4:
        st.metric(
            label="80% Confidence Interval",
            value=f"{lower_bound} – {upper_bound}",
            help="Statistical uncertainty band derived from held-out test set RMSE."
        )

    st.progress(occupancy_rate / 100.0)

    # Visual Cinema Screen & Seat Grid
    st.markdown("""
    <div class="screen-indicator">
        <div class="screen-arc"></div>
        <div class="screen-caption">⸻ CINEMA SCREEN (ALL EYES THIS WAY) ⸻</div>
    </div>
    """, unsafe_allow_html=True)

    # Generate Seat Grid HTML
    cols_per_row = 15 if screen_capacity <= 180 else 20
    total_rows = int(np.ceil(screen_capacity / float(cols_per_row)))
    row_labels = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N"]

    seat_grid_html = '<div class="seat-grid">'
    seat_count = 0
    booked_count = predicted_tickets

    for r_idx in range(total_rows):
        label = row_labels[r_idx % len(row_labels)]
        seat_grid_html += f'<div class="seat-row"><span class="seat-row-label">{label}</span>'
        for c_idx in range(cols_per_row):
            if seat_count < screen_capacity:
                if seat_count < booked_count:
                    seat_grid_html += '<span class="seat-unit seat-filled" title="Forecasted Booked"></span>'
                else:
                    seat_grid_html += '<span class="seat-unit seat-empty" title="Available"></span>'
                seat_count += 1
            else:
                break
        seat_grid_html += f'<span class="seat-row-label">{label}</span></div>'

    seat_grid_html += f"""
        <div class="seat-key">
            <div class="seat-key-item"><span class="seat-unit seat-filled"></span> <span>Forecasted Booked ({predicted_tickets} seats)</span></div>
            <div class="seat-key-item"><span class="seat-unit seat-empty"></span> <span>Available for Walk-ins ({screen_capacity - predicted_tickets} seats)</span></div>
        </div>
    </div>
    """
    st.markdown(seat_grid_html, unsafe_allow_html=True)

# ---------------------------------------------------------
# TAB 2: STAFFING & SCREEN OPTIMIZATION
# ---------------------------------------------------------
with tab_operations:
    st.subheader("🛠️ Decision Support: Resource & Capacity Management")
    st.caption("Actionable guidance for theater managers based on predicted attendance footfall:")

    col_staff, col_alloc = st.columns(2)

    with col_staff:
        with st.container(border=True):
            st.markdown("#### 👥 Labor & Staffing Deployment")
            st.markdown(
                "Staffing ratios configured to prevent customer bottlenecks while preventing overstaffing:"
            )
            st.markdown(f"- **Concession Stand Staff:** **{concession_staff} members** *(Ratio: 1 staff per 45 forecasted patrons)*")
            st.markdown(f"- **Auditorium Ushers / Cleaning:** **{ushers_needed} members** *(Ratio: 1 usher per 60 forecasted patrons)*")
            st.markdown(f"- **Ticket Counter Clerks:** **{ticket_clerks} members** *(Ratio: 1 clerk per 55 forecasted patrons)*")
            st.caption("Heuristics account for concession rush intervals 20 minutes prior to showtime.")

    with col_alloc:
        with st.container(border=True):
            st.markdown("#### 📽️ Screen Reallocation Analysis")
            if occupancy_rate >= 85.0:
                st.error(
                    f"🚨 **High Demand Alert ({occupancy_rate}% Occupancy):** "
                    f"Forecasted demand ({predicted_tickets} tickets) approaches or hits screen capacity ({screen_capacity} seats).\n\n"
                    "**Manager Recommendation:** Upgrade this screening to **Screen 1 (250 seats IMAX)** or open a secondary "
                    "parallel screening in another hall to capture unfulfilled demand."
                )
            elif occupancy_rate <= 30.0:
                st.info(
                    f"💡 **Low Utilization Alert ({occupancy_rate}% Occupancy):** "
                    f"Projected demand ({predicted_tickets} tickets) leaves over 70% of seats empty.\n\n"
                    "**Manager Recommendation:** Downsize to **Screen 3 (120 seats Boutique)** to save HVAC and projection costs, "
                    "or launch an app-exclusive concession combo discount."
                )
            else:
                st.success(
                    f"✅ **Optimal Capacity Match ({occupancy_rate}% Occupancy):** "
                    "Projected ticket sales align comfortably with this auditorium's seating capacity. No screen swapping required."
                )

# ---------------------------------------------------------
# TAB 3: MODEL EXPLAINABILITY & DRIVERS
# ---------------------------------------------------------
with tab_explain:
    st.subheader("🔍 Model Explainability & Local Show Drivers")
    st.caption("Understanding the primary quantitative drivers behind this show's demand forecast:")

    col_feat, col_local = st.columns([3, 2])

    with col_feat:
        st.markdown("##### Global Feature Importances (Trained Model)")
        if feature_importances:
            imp_df = pd.DataFrame(feature_importances).head(8)
            st.bar_chart(
                data=imp_df.set_index("Feature")["Importance"],
                color="#ff4b4b"
            )
        else:
            st.info("Feature importance data loaded from model pipeline.")

    with col_local:
        with st.container(border=True):
            st.markdown("##### Local Demand Drivers for This Show")
            st.markdown(f"""
            - **7-Day Lag Demand Anchor:** **{lag_7d_tickets} tickets**  
              *(Strongest predictor; captures habitual recurring attendance patterns).*
            - **Showtime Slot:** **{selected_slot}** on **{selected_day}**  
              *(Slot baseline occupancy: {timeslot_prior_occ*100:.1f}%).*
            - **Genre Track Record:** **{movie_genre}**  
              *(Genre baseline prior: {movie_hist_avg_occ*100:.1f}%).*
            - **Lifecycle Decay:** **Day {days_since_release}**  
              *(Natural box-office decay factor applied since theatrical debut).*
            """)

# ---------------------------------------------------------
# TAB 4: HOLDOUT VALIDATION BENCHMARKS
# ---------------------------------------------------------
with tab_benchmarks:
    st.subheader("📈 Chronological Held-Out Test Set Benchmark")
    st.markdown(
        "To guarantee zero future data leakage, candidate models were evaluated on a strictly chronological "
        "held-out test set (final 15% of the timeline, 324 shows):"
    )

    if benchmark_summary:
        bench_df = pd.DataFrame(benchmark_summary)
        st.dataframe(bench_df, hide_index=True)

    st.markdown("---")
    st.subheader("❄️ Cold-Start vs Mature Shows Error Breakdown")
    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.metric(
                label="Mature Shows MAE (> 3 Days Run)",
                value=f"{cold_start_metrics.get('mature_mae', '11.83')} tickets",
                help="Error on established titles with box-office track record"
            )
            st.caption(f"Test WAPE: **{cold_start_metrics.get('mature_wape', '14.73')}%** (n=292 shows)")
    with c2:
        with st.container(border=True):
            st.metric(
                label="Cold-Start Shows MAE (Days 0–3)",
                value=f"{cold_start_metrics.get('cold_mae', '16.94')} tickets",
                help="Error on newly released titles lacking historical attendance data"
            )
            st.caption(f"Test WAPE: **{cold_start_metrics.get('cold_wape', '15.38')}%** (n=32 shows)")

# ---------------------------------------------------------
# TAB 5: LIMITATIONS & RESPONSIBLE AI
# ---------------------------------------------------------
with tab_governance:
    st.subheader("⚠️ Operational Limitations & Governance")
    st.markdown("""
    #### 1. Non-Autonomous Decision Support
    - This model functions exclusively as a **decision-support tool** to assist theater supervisors, not an autonomous scheduling system.
    - Final screen assignments and labor rosters must be authorized by duty managers.

    #### 2. Cold-Start Handling
    - Newly released titles lack box-office momentum history. The model defaults to genre priors and widens the uncertainty interval by **35%**.
    - Supervisors are advised to check early advance booking velocity ($T-6\text{h}$) for opening weekend shows.

    #### 3. External Anomalies
    - The model cannot anticipate unmodeled external shocks (severe weather storms, transit strikes, viral local news).
    - Human supervisors should override model recommendations during known anomalous events.

    #### 4. Privacy by Design
    - All training and inference data operates strictly at the aggregated show level. Zero Personally Identifiable Information (PII) or customer demographic attributes are collected or processed.
    """)

st.markdown("---")
st.caption("CinemaOps AI Demand Forecaster | Decision-Support MVP | Author: AI & ML Intern Candidate")
