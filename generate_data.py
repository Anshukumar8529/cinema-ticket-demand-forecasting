"""
generate_data.py
----------------
Synthetic dataset generator for cinema hall ticket demand forecasting.
Generates show-level attendance records across multiplex screens, modeling:
- Box-office lifecycle decay curves for theatrical releases
- Show timing multipliers and weekend demand surges
- Leakage-safe historical demand features (lagged sales, rolling occupancy)
- Boundary constraints (physical auditorium seating capacity limits)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_cinema_dataset(start_date="2025-01-01", days=180, random_seed=42):
    np.random.seed(random_seed)
    base_date = datetime.strptime(start_date, "%Y-%m-%d")
    date_range = [base_date + timedelta(days=i) for i in range(days)]

    # 1. Theater Screen Infrastructure
    screens = [
        {"screen_id": 1, "screen_type": "IMAX / Large", "capacity": 250},
        {"screen_id": 2, "screen_type": "Standard Main", "capacity": 180},
        {"screen_id": 3, "screen_type": "Boutique / Gold", "capacity": 120}
    ]

    # 2. Show Timing Slots
    show_slots = [
        {"show_time": "Morning", "time_slot": "10:30 AM", "multiplier": 0.60},
        {"show_time": "Matinee", "time_slot": "02:00 PM", "multiplier": 0.85},
        {"show_time": "Evening", "time_slot": "06:00 PM", "multiplier": 1.35},
        {"show_time": "Night",   "time_slot": "09:30 PM", "multiplier": 1.50}
    ]

    # 3. Movie Catalog with Staggered Releases across the 180 days (including in test set)
    movie_catalog = [
        {"title": "Galactic Wars VIII", "genre": "Sci-Fi", "base_buzz": 1.45, "release_day": 0, "run_days": 45},
        {"title": "Laugh Riot 2", "genre": "Comedy", "base_buzz": 1.10, "release_day": 0, "run_days": 35},
        {"title": "Love in Venice", "genre": "Romance", "base_buzz": 0.95, "release_day": 0, "run_days": 28},
        {"title": "Shadow Detective", "genre": "Thriller", "base_buzz": 1.20, "release_day": 25, "run_days": 35},
        {"title": "SuperHero Unleashed", "genre": "Action", "base_buzz": 1.55, "release_day": 55, "run_days": 45},
        {"title": "Kingdom of Animals", "genre": "Animation", "base_buzz": 1.30, "release_day": 85, "run_days": 42},
        {"title": "The Haunted Asylum", "genre": "Horror", "base_buzz": 1.05, "release_day": 120, "run_days": 32},
        {"title": "Silicon Dreamers", "genre": "Drama", "base_buzz": 0.90, "release_day": 140, "run_days": 35},
        {"title": "Cosmic Odyssey", "genre": "Sci-Fi", "base_buzz": 1.40, "release_day": 158, "run_days": 25}
    ]

    # 4. Calendar Holidays (New Year, Republic Day, Holi, Good Friday, May Day, Eid, etc.)
    holiday_dates = {
        "2025-01-01", "2025-01-26", "2025-02-14", "2025-03-14",
        "2025-03-31", "2025-04-18", "2025-05-01", "2025-06-07"
    }

    # First pass: Generate raw schedule and ground truth sales
    raw_shows = []
    record_id = 1

    for day_idx, cur_date in enumerate(date_range):
        date_str = cur_date.strftime("%Y-%m-%d")
        weekday_num = cur_date.weekday() # 0 = Monday, 6 = Sunday
        day_of_week = cur_date.strftime("%A")
        is_weekend = 1 if weekday_num in [4, 5, 6] else 0 # Fri, Sat, Sun
        is_holiday = 1 if date_str in holiday_dates else 0

        # Day-of-week multiplier
        dow_multipliers = {
            "Monday": 0.85,
            "Tuesday": 0.80, # Discount Tuesday effect
            "Wednesday": 0.90,
            "Thursday": 0.95,
            "Friday": 1.25,
            "Saturday": 1.60,
            "Sunday": 1.45
        }
        dow_mult = dow_multipliers.get(day_of_week, 1.0)
        holiday_mult = 1.35 if is_holiday else 1.0

        # Active movies in theaters on this day
        active_movies = [
            m for m in movie_catalog
            if m["release_day"] <= day_idx < (m["release_day"] + m["run_days"])
        ]
        if not active_movies:
            active_movies = [movie_catalog[0]]

        # Schedule movies on screens
        # Screen 1 (largest) gets the highest current buzz movie
        sorted_movies = sorted(
            active_movies,
            key=lambda m: m["base_buzz"] * np.exp(-0.02 * (day_idx - m["release_day"])),
            reverse=True
        )

        screen_movie_map = {
            1: sorted_movies[0],
            2: sorted_movies[1 % len(sorted_movies)],
            3: sorted_movies[2 % len(sorted_movies)]
        }

        for screen in screens:
            s_id = screen["screen_id"]
            capacity = screen["capacity"]
            assigned_movie = screen_movie_map[s_id]

            days_since_release = day_idx - assigned_movie["release_day"]
            # Lifecycle decay: exponential decay from opening buzz
            lifecycle_decay = np.exp(-0.022 * max(0, days_since_release))
            current_buzz = assigned_movie["base_buzz"] * lifecycle_decay

            # Cold start: first 3 days of release
            is_cold_start = 1 if days_since_release <= 3 else 0

            for slot in show_slots:
                # Base unconstrained demand calculation
                base_demand = capacity * 0.42
                expected_demand = base_demand * current_buzz * slot["multiplier"] * dow_mult * holiday_mult

                # Add realistic observational noise
                noise = np.random.normal(0, capacity * 0.06)
                simulated_sales = int(np.round(expected_demand + noise))

                # Physical capacity constraint
                tickets_sold = int(np.clip(simulated_sales, 5, capacity))
                occupancy_rate = round(tickets_sold / capacity, 4)

                raw_shows.append({
                    "show_id": f"SHW{record_id:05d}",
                    "date": date_str,
                    "day_idx": day_idx,
                    "day_of_week": day_of_week,
                    "is_weekend": is_weekend,
                    "is_holiday": is_holiday,
                    "screen_id": s_id,
                    "screen_type": screen["screen_type"],
                    "screen_capacity": capacity,
                    "movie_title": assigned_movie["title"],
                    "genre": assigned_movie["genre"],
                    "days_since_release": days_since_release,
                    "is_cold_start": is_cold_start,
                    "show_time": slot["show_time"],
                    "time_slot": slot["time_slot"],
                    "tickets_sold": tickets_sold,
                    "occupancy_rate": occupancy_rate
                })
                record_id += 1

    df_raw = pd.DataFrame(raw_shows)

    # 5. Compute Historical Demand Features (Strictly Past Information, t < current_date)
    # Guaranteed NO TARGET LEAKAGE: past stats only incorporate shows up to yesterday (day_idx < today).
    print("Computing leakage-safe historical demand features...")

    # Fallback priors (overall genre and timeslot baseline expectations)
    genre_prior_occupancy = {
        "Sci-Fi": 0.62, "Action": 0.65, "Comedy": 0.52, "Animation": 0.58,
        "Thriller": 0.50, "Horror": 0.46, "Romance": 0.44, "Drama": 0.40
    }
    timeslot_prior_occupancy = {
        "Morning": 0.32, "Matinee": 0.48, "Evening": 0.72, "Night": 0.78
    }

    movie_hist_avg_occ = []
    timeslot_hist_avg_occ = []
    screen_hist_avg_occ = []
    lag_7d_tickets_list = []

    # Map for fast 7-day lag lookup: (date_str, screen_id, show_time) -> tickets_sold
    slot_lookup = {}
    for _, row in df_raw.iterrows():
        slot_lookup[(row["date"], row["screen_id"], row["show_time"])] = row["tickets_sold"]

    for idx, row in df_raw.iterrows():
        cur_day_idx = row["day_idx"]
        cur_date_dt = datetime.strptime(row["date"], "%Y-%m-%d")

        if cur_day_idx == 0:
            # First day: pure cold start, use informative domain priors
            m_occ = genre_prior_occupancy.get(row["genre"], 0.50)
            t_occ = timeslot_prior_occupancy.get(row["show_time"], 0.55)
            s_occ = 0.55
        else:
            # Past shows strictly up to yesterday
            past_df = df_raw[df_raw["day_idx"] < cur_day_idx]

            # 1. Historical movie occupancy
            past_movie_shows = past_df[past_df["movie_title"] == row["movie_title"]]
            if len(past_movie_shows) >= 4:
                m_occ = round(past_movie_shows["occupancy_rate"].mean(), 4)
            else:
                # Cold-start / low data fallback
                m_occ = genre_prior_occupancy.get(row["genre"], 0.50)

            # 2. Historical timeslot occupancy for this (day_of_week, show_time)
            past_slot_shows = past_df[
                (past_df["day_of_week"] == row["day_of_week"]) &
                (past_df["show_time"] == row["show_time"])
            ]
            if len(past_slot_shows) >= 1:
                t_occ = round(past_slot_shows["occupancy_rate"].mean(), 4)
            else:
                t_occ = timeslot_prior_occupancy.get(row["show_time"], 0.55)

            # 3. Screen historical average
            past_screen_shows = past_df[past_df["screen_id"] == row["screen_id"]]
            if len(past_screen_shows) >= 1:
                s_occ = round(past_screen_shows["occupancy_rate"].mean(), 4)
            else:
                s_occ = 0.55

        # 4. Lag 7-Day Tickets (same screen and showtime exactly 7 days ago)
        lag_date_str = (cur_date_dt - timedelta(days=7)).strftime("%Y-%m-%d")
        lag_val = slot_lookup.get((lag_date_str, row["screen_id"], row["show_time"]), None)
        if lag_val is None:
            # Fallback for first 7 days: capacity * timeslot expected occupancy
            lag_val = int(round(row["screen_capacity"] * timeslot_prior_occupancy.get(row["show_time"], 0.50)))

        movie_hist_avg_occ.append(m_occ)
        timeslot_hist_avg_occ.append(t_occ)
        screen_hist_avg_occ.append(s_occ)
        lag_7d_tickets_list.append(lag_val)

    df_raw["movie_hist_avg_occ"] = movie_hist_avg_occ
    df_raw["timeslot_hist_avg_occ"] = timeslot_hist_avg_occ
    df_raw["screen_hist_avg_occ"] = screen_hist_avg_occ
    df_raw["lag_7d_tickets"] = lag_7d_tickets_list

    # Drop intermediate column
    df_final = df_raw.drop(columns=["day_idx"])
    return df_final

if __name__ == "__main__":
    print("==================================================================")
    print(" Cinema Ticket Demand Forecasting - Synthetic Dataset Generator")
    print(" Algions AI Labs Private Limited Assessment")
    print("==================================================================")
    df = generate_cinema_dataset(days=180, random_seed=42)
    output_file = "cinema_ticket_demand.csv"
    df.to_csv(output_file, index=False)
    print(f"\n[OK] Dataset generated successfully: {output_file}")
    print(f"Total show records: {len(df):,}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()} ({df['date'].nunique()} calendar days)")
    print(f"Unique screens: {df['screen_id'].nunique()} screens (Capacities: 120, 180, 250)")
    print(f"Movies covered: {df['movie_title'].nunique()} across {df['genre'].nunique()} genres")
    print(f"Cold-start shows (<=3 days of release): {df['is_cold_start'].sum()} shows ({df['is_cold_start'].mean():.1%})")
    print("\nSample Preview:")
    print(df[["show_id", "date", "movie_title", "genre", "show_time", "screen_capacity", "days_since_release", "is_cold_start", "tickets_sold", "occupancy_rate"]].head(6))
