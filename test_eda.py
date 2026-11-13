"""
Matthew Tong & Thomas Astier
CSE 163 Final Project: The Aging Endurance Athlete
Module: test_eda.py

Unit testing suite verifying time-string parsing, age bracket classification,
multi-dataset merging, career decline calculation, and seven-number summary
statistics on small, deterministic toy datasets.
"""

import math
import numpy as np
import pandas as pd
from eda import (
    parse_time_to_seconds,
    assign_age_bracket,
    clean_and_join_datasets,
    compute_career_decline,
    seven_number_summary,
)


def test_parse_time_to_seconds():
    """
    Tests conversion of ASO duration strings, gap strings, winners' dashes,
    and DNF/NaN flags into numerical seconds.
    """
    # 82h 05' 45'' -> 82*3600 + 5*60 + 45 = 295200 + 300 + 45 = 295545.0
    assert math.isclose(parse_time_to_seconds("82h 05' 45''"), 295545.0)
    # + 00h 06' 26'' -> 6*60 + 26 = 386.0
    assert math.isclose(parse_time_to_seconds("+ 00h 06' 26''"), 386.0)
    # Winner gap representation '-' -> 0.0
    assert math.isclose(parse_time_to_seconds("-"), 0.0)
    # Non-finishers and missing values -> NaN
    assert np.isnan(parse_time_to_seconds("DNF"))
    assert np.isnan(parse_time_to_seconds("OTL"))
    assert np.isnan(parse_time_to_seconds(np.nan))
    print("test_parse_time_to_seconds passed.")


def test_assign_age_bracket():
    """
    Tests boundary conditions of physiological age bracket assignment.
    """
    assert assign_age_bracket(19) == "U23 (19-22)"
    assert assign_age_bracket(22) == "U23 (19-22)"
    assert assign_age_bracket(23) == "Prime-Early (23-26)"
    assert assign_age_bracket(26) == "Prime-Early (23-26)"
    assert assign_age_bracket(27) == "Prime-Late (27-30)"
    assert assign_age_bracket(30) == "Prime-Late (27-30)"
    assert assign_age_bracket(31) == "Veteran (31+)"
    assert assign_age_bracket(np.nan) == "Unknown"
    print("test_assign_age_bracket passed.")


def test_clean_and_join_and_decline():
    """
    Tests dataset joining, age calculation, missing-value preservation for
    DNF rows, and year-over-year career decline metrics on a toy dataset.
    """
    toy_gc = pd.DataFrame([
        {
            "Race_Year": 2024,
            "Gender": "Male",
            "GC_Rank_Raw": "1",
            "Rider_Name": "RIDER ALPHA",
            "Rider_No": "1",
            "Team": "TEAM A",
            "Times_Raw": "80h 00' 00''",
            "Gap_Raw": "-",
            "Bonus_Raw": "B : 10''",
            "Finish_Status": "DF",
        },
        {
            "Race_Year": 2025,
            "Gender": "Male",
            "GC_Rank_Raw": "4",
            "Rider_Name": "RIDER ALPHA",
            "Rider_No": "1",
            "Team": "TEAM A",
            "Times_Raw": "81h 00' 00''",
            "Gap_Raw": "+ 00h 05' 00''",
            "Bonus_Raw": "-",
            "Finish_Status": "DF",
        },
        {
            "Race_Year": 2025,
            "Gender": "Male",
            "GC_Rank_Raw": "DNF",
            "Rider_Name": "RIDER BETA",
            "Rider_No": "2",
            "Team": "TEAM B",
            "Times_Raw": np.nan,
            "Gap_Raw": np.nan,
            "Bonus_Raw": "-",
            "Finish_Status": "DNF",
        },
    ])
    toy_riders = pd.DataFrame([
        {
            "Rider_Name": "RIDER ALPHA",
            "Gender": "Male",
            "Birth_Year": 1994,
            "Nationality": "FR",
            "Primary_Team": "TEAM A",
        },
        {
            "Rider_Name": "RIDER BETA",
            "Gender": "Male",
            "Birth_Year": 2004,
            "Nationality": "ES",
            "Primary_Team": "TEAM B",
        },
    ])
    toy_elev = pd.DataFrame([
        {
            "Race_Year": 2024,
            "Gender": "Male",
            "Total_Distance_km": 3498.0,
            "Total_Elevation_m": 52300,
            "Mountain_Stages": 9,
            "Avg_Temp_C": 25.4,
        },
        {
            "Race_Year": 2025,
            "Gender": "Male",
            "Total_Distance_km": 3338.0,
            "Total_Elevation_m": 55000,
            "Mountain_Stages": 10,
            "Avg_Temp_C": 24.9,
        },
    ])

    j_riders, j_elev = clean_and_join_datasets(toy_gc, toy_riders, toy_elev)
    assert len(j_riders) == 3
    assert len(j_elev) == 3

    # Verify Rider Alpha's ages (2024 - 1994 = 30; 2025 - 1994 = 31)
    alpha_rows = j_elev[j_elev["Rider_Name"] == "RIDER ALPHA"].sort_values(
        "Race_Year"
    )
    assert list(alpha_rows["Rider_Age"]) == [30, 31]
    assert list(alpha_rows["Age_Bracket"]) == [
        "Prime-Late (27-30)",
        "Veteran (31+)",
    ]

    # Verify DNF row preserves NaN for numeric rank and seconds
    beta_row = j_elev[j_elev["Rider_Name"] == "RIDER BETA"].iloc[0]
    assert np.isnan(beta_row["GC_Rank"])
    assert np.isnan(beta_row["Total_Seconds"])

    # Verify career decline calculation for Rider Alpha (Rank 1 -> Rank 4)
    decline = compute_career_decline(j_riders)
    assert len(decline) == 1
    row = decline.iloc[0]
    assert math.isclose(row["Annual_Rank_Change"], 3.0)
    assert math.isclose(row["Rank_Pct_Drop"], 300.0)
    assert math.isclose(row["Annual_Gap_Change_Sec"], 300.0)
    print("test_clean_and_join_and_decline passed.")


def test_seven_number_summary():
    """
    Tests the seven-number summary output values against known arithmetic.
    """
    df = pd.DataFrame({"val": [10.0, 20.0, 30.0, 40.0, 50.0]})
    summary = seven_number_summary(df, ["val"])
    assert math.isclose(summary.loc["mean", "val"], 30.0)
    assert math.isclose(summary.loc["min", "val"], 10.0)
    assert math.isclose(summary.loc["25%", "val"], 20.0)
    assert math.isclose(summary.loc["50%", "val"], 30.0)
    assert math.isclose(summary.loc["75%", "val"], 40.0)
    assert math.isclose(summary.loc["max", "val"], 50.0)
    print("test_seven_number_summary passed.")


def main():
    """
    Runs all unit tests in the test suite.
    """
    test_parse_time_to_seconds()
    test_assign_age_bracket()
    test_clean_and_join_and_decline()
    test_seven_number_summary()
    print("All EDA unit tests passed!")


if __name__ == "__main__":
    main()
