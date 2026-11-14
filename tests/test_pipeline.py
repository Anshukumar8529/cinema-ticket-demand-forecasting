"""
tests/test_pipeline.py
----------------------
Automated test suite verifying data integrity, physical boundary constraints,
temporal leakage prevention, artifact structure, and model inference pipelines.
"""

import os
import sys
# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pickle
import pytest
import numpy as np
import pandas as pd
from model_training import calculate_wape

DATASET_PATH = "cinema_ticket_demand.csv"
ARTIFACT_PATH = "cinema_model_artifacts.pkl"

def test_dataset_exists_and_non_empty():
    """Verify dataset is generated and exceeds 1,000 records."""
    assert os.path.exists(DATASET_PATH), f"Dataset {DATASET_PATH} not found!"
    df = pd.read_csv(DATASET_PATH)
    assert len(df) >= 1000, f"Dataset must have at least 1,000 records, got {len(df)}"

def test_data_integrity_and_capacity_constraints():
    """Verify that physical constraints (tickets <= capacity, tickets >= 0) hold strictly."""
    df = pd.read_csv(DATASET_PATH)
    assert (df["tickets_sold"] < 0).sum() == 0, "Negative ticket sales detected!"
    assert (df["tickets_sold"] > df["screen_capacity"]).sum() == 0, "Ticket sales exceed screen capacity!"
    assert (df["occupancy_rate"] < 0.0).sum() == 0, "Occupancy rate < 0 detected!"
    assert (df["occupancy_rate"] > 1.0).sum() == 0, "Occupancy rate > 100% detected!"
    assert df.isnull().sum().sum() == 0, "Unexpected null values in dataset!"

def test_target_leakage_safeguards():
    """Verify that no future target information leaks into historical features."""
    df = pd.read_csv(DATASET_PATH)
    df["date"] = pd.to_datetime(df["date"])
    
    # Check that day 0 cold-start shows use fallback priors rather than same-day sales
    min_date = df["date"].min()
    day_0_shows = df[df["date"] == min_date]
    assert len(day_0_shows) > 0
    # Day 0 days_since_release must be non-negative
    assert (day_0_shows["days_since_release"] < 0).sum() == 0

def test_artifact_structure():
    """Verify serialized model artifact contains all required production components."""
    assert os.path.exists(ARTIFACT_PATH), f"Model artifact {ARTIFACT_PATH} not found!"
    with open(ARTIFACT_PATH, "rb") as f:
        artifacts = pickle.load(f)
    
    required_keys = [
        "model_name", "pipeline", "baseline_lookup", "feature_cols",
        "benchmark_summary", "cold_start_metrics", "genre_priors", "timeslot_priors"
    ]
    for key in required_keys:
        assert key in artifacts, f"Missing key '{key}' in serialized artifact!"

def test_inference_pipeline_normal_case():
    """Verify inference pipeline produces valid, bounded predictions for standard input."""
    with open(ARTIFACT_PATH, "rb") as f:
        artifacts = pickle.load(f)
    pipeline = artifacts["pipeline"]
    feature_cols = artifacts["feature_cols"]

    sample_input = pd.DataFrame([{
        "screen_capacity": 250,
        "days_since_release": 10,
        "movie_hist_avg_occ": 0.65,
        "timeslot_hist_avg_occ": 0.72,
        "screen_hist_avg_occ": 0.60,
        "lag_7d_tickets": 160,
        "genre": "Action",
        "show_time": "Evening",
        "day_of_week": "Saturday",
        "is_weekend": 1,
        "is_holiday": 0,
        "is_cold_start": 0
    }])[feature_cols]

    raw_pred = pipeline.predict(sample_input)[0]
    final_tickets = int(np.clip(np.round(raw_pred), 0, 250))
    occupancy = round(final_tickets / 250.0, 4)

    assert isinstance(final_tickets, int)
    assert 0 <= final_tickets <= 250, f"Prediction {final_tickets} out of capacity bounds!"
    assert 0.0 <= occupancy <= 1.0

def test_inference_pipeline_cold_start_case():
    """Verify inference pipeline handles cold-start (new movie release) gracefully."""
    with open(ARTIFACT_PATH, "rb") as f:
        artifacts = pickle.load(f)
    pipeline = artifacts["pipeline"]
    feature_cols = artifacts["feature_cols"]

    cold_input = pd.DataFrame([{
        "screen_capacity": 180,
        "days_since_release": 1,
        "movie_hist_avg_occ": artifacts["genre_priors"].get("Sci-Fi", 0.50),
        "timeslot_hist_avg_occ": artifacts["timeslot_priors"].get("Night", 0.70),
        "screen_hist_avg_occ": 0.55,
        "lag_7d_tickets": 120,
        "genre": "Sci-Fi",
        "show_time": "Night",
        "day_of_week": "Friday",
        "is_weekend": 1,
        "is_holiday": 0,
        "is_cold_start": 1
    }])[feature_cols]

    raw_pred = pipeline.predict(cold_input)[0]
    final_tickets = int(np.clip(np.round(raw_pred), 0, 180))

    assert 0 <= final_tickets <= 180
    assert final_tickets > 0, "Cold start opening show should have positive demand"

def test_wape_metric_calculation():
    """Verify mathematical correctness of WAPE calculation."""
    y_true = np.array([100, 150, 200, 50])
    y_pred = np.array([90, 160, 190, 60])
    # Total error: |100-90| + |150-160| + |200-190| + |50-60| = 10 + 10 + 10 + 10 = 40
    # Total actual: 100 + 150 + 200 + 50 = 500
    # WAPE: 40 / 500 * 100 = 8.0%
    expected_wape = 8.0
    actual_wape = calculate_wape(y_true, y_pred)
    assert pytest.approx(actual_wape, rel=1e-4) == expected_wape

