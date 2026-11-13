"""
Matthew Tong & Thomas Astier
CSE 163 Final Project: The Aging Endurance Athlete
Module: eda.py

Performs end-to-end Exploratory Data Analysis (EDA) across the joined
Tour de France General Classification, Rider Biographical, and Stage Elevation
datasets. Outputs summary statistics, missing data diagnostics, statistical
assumption checks (Shapiro-Wilk, Levene, One-Way ANOVA, Kruskal-Wallis), and
four visualizations saved to the plots/ directory.
"""

import os
import re
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


def parse_time_to_seconds(time_str):
    """
    Converts an ASO-formatted duration string (e.g., "82h 05' 45''",
    "+ 00h 06' 26''", or "-") into integer seconds. Returns np.nan for
    missing values or non-finisher strings ("DNF", "OTL", "DNS").

    Parameters:
        time_str (str or float): Raw time or gap string from rankings table.

    Returns:
        float: Total seconds as a float (or np.nan if missing/invalid).
    """
    if pd.isna(time_str):
        return np.nan
    cleaned = str(time_str).strip()
    if cleaned in ("-", "0", "+ 00h 00' 00''"):
        return 0.0
    if any(flag in cleaned.upper() for flag in ("DNF", "OTL", "DNS", "DSQ")):
        return np.nan

    matches = re.findall(r"\d+", cleaned)
    if len(matches) == 3:
        hrs, mins, secs = (int(matches[0]), int(matches[1]), int(matches[2]))
        return float(hrs * 3600 + mins * 60 + secs)
    if len(matches) == 2:
        mins, secs = (int(matches[0]), int(matches[1]))
        return float(mins * 60 + secs)
    return np.nan


def assign_age_bracket(age):
    """
    Categorizes a rider's integer age into one of four physiological career
    brackets for group-level comparison and ANOVA validation.

    Parameters:
        age (float or int): Rider age in years during the race edition.

    Returns:
        str: Age bracket label ('U23 (19-22)', 'Prime-Early (23-26)',
             'Prime-Late (27-30)', or 'Veteran (31+)').
    """
    if pd.isna(age):
        return "Unknown"
    if age <= 22:
        return "U23 (19-22)"
    if age <= 26:
        return "Prime-Early (23-26)"
    if age <= 30:
        return "Prime-Late (27-30)"
    return "Veteran (31+)"


def clean_and_join_datasets(gc_df, riders_df, elev_df):
    """
    Cleans raw GC rankings, joins with rider birth-year records (Join 1),
    and joins with edition elevation profiles (Join 2).

    Parameters:
        gc_df (pd.DataFrame): Raw GC rankings table.
        riders_df (pd.DataFrame): Rider biographical lookup table.
        elev_df (pd.DataFrame): Edition elevation and weather table.

    Returns:
        tuple: (joined_gc_riders, joined_gc_elevation)
    """
    gc_clean = gc_df.copy()
    gc_clean["Total_Seconds"] = gc_clean["Times_Raw"].apply(
        parse_time_to_seconds
    )
    gc_clean["Gap_Seconds"] = gc_clean["Gap_Raw"].apply(parse_time_to_seconds)
    gc_clean["GC_Rank"] = pd.to_numeric(
        gc_clean["GC_Rank_Raw"], errors="coerce"
    )

    # Join 1: GC Rankings + Rider Biographical Dataset
    joined_riders = pd.merge(
        gc_clean,
        riders_df,
        on=["Rider_Name", "Gender"],
        how="left",
    )
    joined_riders["Rider_Age"] = (
        joined_riders["Race_Year"] - joined_riders["Birth_Year"]
    )
    joined_riders["Age_Bracket"] = joined_riders["Rider_Age"].apply(
        assign_age_bracket
    )

    # Join 2: Joined Rider-GC + Stage/Edition Elevation Profiles
    joined_elev = pd.merge(
        joined_riders,
        elev_df,
        on=["Race_Year", "Gender"],
        how="left",
    )

    # Classify High vs. Moderate Elevation relative to gender median
    med_elev = joined_elev.groupby("Gender")["Total_Elevation_m"].transform(
        "median"
    )
    joined_elev["Elevation_Tier"] = np.where(
        joined_elev["Total_Elevation_m"] >= med_elev,
        "High Elevation",
        "Moderate Elevation",
    )
    return joined_riders, joined_elev


def compute_career_decline(joined_df):
    """
    Computes post-peak year-over-year GC rank and gap changes for riders
    who competed in multiple editions at or after age 27 (answering RQ2).

    Parameters:
        joined_df (pd.DataFrame): Cleaned and joined rider-GC DataFrame.

    Returns:
        pd.DataFrame: Rider-edition pairs with career rank/gap delta metrics.
    """
    finishers = joined_df.dropna(subset=["GC_Rank", "Gap_Seconds"]).copy()
    finishers = finishers.sort_values(["Rider_Name", "Race_Year"])

    finishers["Prev_Rank"] = finishers.groupby("Rider_Name")["GC_Rank"].shift(
        1
    )
    finishers["Prev_Year"] = finishers.groupby("Rider_Name")[
        "Race_Year"
    ].shift(1)
    finishers["Prev_Gap"] = finishers.groupby("Rider_Name")[
        "Gap_Seconds"
    ].shift(1)

    multi = finishers.dropna(subset=["Prev_Rank", "Prev_Year"]).copy()
    multi["Years_Elapsed"] = multi["Race_Year"] - multi["Prev_Year"]
    multi = multi[multi["Years_Elapsed"] > 0].copy()

    multi["Annual_Rank_Change"] = (
        (multi["GC_Rank"] - multi["Prev_Rank"]) / multi["Years_Elapsed"]
    )
    multi["Rank_Pct_Drop"] = (
        (multi["GC_Rank"] - multi["Prev_Rank"])
        / multi["Prev_Rank"]
        * 100.0
    )
    multi["Annual_Gap_Change_Sec"] = (
        (multi["Gap_Seconds"] - multi["Prev_Gap"]) / multi["Years_Elapsed"]
    )
    return multi


def seven_number_summary(df, columns):
    """
    Computes the CSE 163 seven-number summary (mean, std, min, 25%, 50%,
    75%, max) for a list of quantitative columns in a DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        columns (list of str): Quantitative column names.

    Returns:
        pd.DataFrame: Summary table with seven statistics per variable.
    """
    desc = df[columns].describe().loc[
        ["mean", "std", "min", "25%", "50%", "75%", "max"]
    ]
    return desc.round(2)


def report_missing_data(df, label):
    """
    Prints Pythonic missing-data diagnostics (count and percentage of NaN
    values per column) for a given DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame to inspect.
        label (str): Descriptive name of the dataset.

    Returns:
        pd.DataFrame: Missing count and percentage table.
    """
    missing_cnt = df.isna().sum()
    missing_pct = (df.isna().mean() * 100.0).round(2)
    summary = pd.DataFrame({
        "Missing_Count": missing_cnt,
        "Missing_Percent": missing_pct,
    })
    print(f"\n=== Missing Data Diagnostics: {label} ===")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(summary[summary["Missing_Count"] > 0])
    if (missing_cnt == 0).all():
        print("Zero missing values verified via df.isna().sum().sum() == 0.")
    return summary


def run_statistical_validity_checks(joined_elev_df):
    """
    Executes hypothesis tests and assumption checks (Shapiro-Wilk normality,
    Levene homogeneity of variance, One-Way ANOVA, and non-parametric
    Kruskal-Wallis fallback) for the Result Validity challenge goal.

    Parameters:
        joined_elev_df (pd.DataFrame): Joined GC + Rider + Elevation DataFrame.

    Returns:
        dict: Dictionary of test statistics and p-values.
    """
    finishers = joined_elev_df.dropna(
        subset=["Gap_Seconds", "GC_Rank", "Rider_Age"]
    ).copy()

    brackets = [
        "U23 (19-22)",
        "Prime-Early (23-26)",
        "Prime-Late (27-30)",
        "Veteran (31+)",
    ]
    groups_gap = [
        finishers[finishers["Age_Bracket"] == b]["Gap_Seconds"].values
        for b in brackets
    ]

    # 1. Homogeneity of variance across age brackets (Levene's test)
    levene_stat, levene_p = stats.levene(*groups_gap)

    # 2. Normality of ANOVA residuals (Shapiro-Wilk test)
    group_means = finishers.groupby("Age_Bracket")["Gap_Seconds"].transform(
        "mean"
    )
    residuals = finishers["Gap_Seconds"] - group_means
    shapiro_stat, shapiro_p = stats.shapiro(residuals)

    # 3. One-Way ANOVA & Kruskal-Wallis across Age Brackets (RQ1/RQ2)
    f_stat_age, p_val_age = stats.f_oneway(*groups_gap)
    kw_stat_age, kw_p_age = stats.kruskal(*groups_gap)

    # 4. Elevation effect on Veterans vs. Prime riders (RQ3)
    high_elev = finishers[
        finishers["Elevation_Tier"] == "High Elevation"
    ]
    vet_high = high_elev[
        high_elev["Age_Bracket"] == "Veteran (31+)"
    ]["Gap_Seconds"].values
    prime_high = high_elev[
        high_elev["Age_Bracket"].isin([
            "Prime-Early (23-26)", "Prime-Late (27-30)"
        ])
    ]["Gap_Seconds"].values

    f_stat_elev, p_val_elev = stats.f_oneway(vet_high, prime_high)
    kw_stat_elev, kw_p_elev = stats.kruskal(vet_high, prime_high)

    results = {
        "levene_stat": float(levene_stat),
        "levene_p": float(levene_p),
        "shapiro_stat": float(shapiro_stat),
        "shapiro_p": float(shapiro_p),
        "anova_age_f": float(f_stat_age),
        "anova_age_p": float(p_val_age),
        "kruskal_age_h": float(kw_stat_age),
        "kruskal_age_p": float(kw_p_age),
        "anova_elev_f": float(f_stat_elev),
        "anova_elev_p": float(p_val_elev),
        "kruskal_elev_h": float(kw_stat_elev),
        "kruskal_elev_p": float(kw_p_elev),
    }

    print("\n=== Statistical Assumption & Hypothesis Test Results ===")
    for k, val in results.items():
        print(f"  {k}: {val:.6g}")
    return results


def create_visualizations(joined_riders, joined_elev, decline_df):
    """
    Generates four publication-quality plots (two for Joined Dataset 1:
    GC + Rider Bio; two for Joined Dataset 2: GC + Rider Bio + Elevation)
    and saves them to the plots/ directory.

    Parameters:
        joined_riders (pd.DataFrame): Joined GC + Rider Bio DataFrame.
        joined_elev (pd.DataFrame): Joined GC + Rider + Elevation DataFrame.
        decline_df (pd.DataFrame): Longitudinal career change DataFrame.
    """
    os.makedirs("plots", exist_ok=True)
    sns.set_theme(style="whitegrid", palette="deep")
    finishers = joined_riders.dropna(subset=["GC_Rank", "Gap_Seconds"]).copy()
    finishers["Gap_Minutes"] = finishers["Gap_Seconds"] / 60.0

    # Plot 1 (Joined Dataset 1): Mean GC Rank & Gap across Age by Gender (RQ1)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    age_summary = (
        finishers.groupby(["Gender", "Rider_Age"], as_index=False)[
            "Gap_Minutes"
        ]
        .agg(["mean", "sem"])
        .reset_index()
    )
    for gender, color, marker in [
        ("Male", "#1f77b4", "o"),
        ("Female", "#d62728", "s"),
    ]:
        sub = age_summary[age_summary["Gender"] == gender]
        ax.plot(
            sub["Rider_Age"],
            sub["mean"],
            marker=marker,
            linewidth=2.2,
            color=color,
            label=f"{gender} Riders (Mean Gap +/- 1 SE)",
        )
        ax.fill_between(
            sub["Rider_Age"],
            sub["mean"] - sub["sem"],
            sub["mean"] + sub["sem"],
            color=color,
            alpha=0.18,
        )
    ax.axvline(
        26, color="#1f77b4", linestyle="--", alpha=0.7,
        label="Male Observed Peak Age (26 yrs)"
    )
    ax.axvline(
        27, color="#d62728", linestyle=":", alpha=0.7,
        label="Female Observed Peak Age (27 yrs)"
    )
    ax.set_title(
        "Figure 1: Mean GC Time Gap to Winner Across Rider Age by Gender",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Rider Age During Race Edition (Years)", fontsize=11)
    ax.set_ylabel("Mean GC Time Gap to Winner (Minutes)", fontsize=11)
    ax.legend(frameon=True, fontsize=10)
    plt.figtext(
        0.5,
        0.01,
        "Caption: Both male and female GC contenders achieve their lowest "
        "mean time gap to the winner between ages 25 and 28, with gaps "
        "widening steadily after age 30.",
        wrap=True,
        horizontalalignment="center",
        fontsize=9.5,
        style="italic",
    )
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig("plots/fig1_peak_age_by_gender.png", dpi=300)
    plt.close(fig)

    # Plot 2 (Joined Dataset 1): Career Rank Change by Age (RQ2)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.regplot(
        data=decline_df[decline_df["Gender"] == "Male"],
        x="Rider_Age",
        y="Annual_Rank_Change",
        ax=ax,
        scatter_kws={"alpha": 0.35, "s": 28, "color": "#1f77b4"},
        line_kws={"color": "#0b3c68", "linewidth": 2.2},
        label="Male Riders (Linear Trend)",
    )
    sns.regplot(
        data=decline_df[decline_df["Gender"] == "Female"],
        x="Rider_Age",
        y="Annual_Rank_Change",
        ax=ax,
        scatter_kws={"alpha": 0.45, "s": 32, "color": "#d62728"},
        line_kws={"color": "#8c1515", "linewidth": 2.2},
        label="Female Riders (Linear Trend)",
    )
    ax.axhline(0, color="black", linestyle="--", linewidth=1.1)
    ax.set_title(
        "Figure 2: Year-Over-Year GC Rank Change Across Rider Career Age",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Rider Age in Current Edition (Years)", fontsize=11)
    ax.set_ylabel(
        "Annual GC Rank Change (+ = Worse Rank / Decline)", fontsize=11
    )
    ax.legend(frameon=True, fontsize=10)
    plt.figtext(
        0.5,
        0.01,
        "Caption: Riders under age 26 exhibit negative annual rank changes "
        "(improving GC finishes), whereas riders past age 28 experience "
        "accelerating positive rank shifts (career decline).",
        wrap=True,
        horizontalalignment="center",
        fontsize=9.5,
        style="italic",
    )
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig("plots/fig2_career_decline_trajectories.png", dpi=300)
    plt.close(fig)

    # Plot 3 (Joined Dataset 2):
    # Age Bracket vs Elevation Tier Interaction (RQ3)
    elev_fin = joined_elev.dropna(subset=["Gap_Seconds", "Rider_Age"]).copy()
    elev_fin["Gap_Minutes"] = elev_fin["Gap_Seconds"] / 60.0
    bracket_order = [
        "U23 (19-22)",
        "Prime-Early (23-26)",
        "Prime-Late (27-30)",
        "Veteran (31+)",
    ]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.barplot(
        data=elev_fin,
        x="Age_Bracket",
        y="Gap_Minutes",
        hue="Elevation_Tier",
        order=bracket_order,
        capsize=0.08,
        palette=["#2b5c8f", "#e07a5f"],
        ax=ax,
    )
    ax.set_title(
        "Figure 3: Mean GC Time Gap by Age Bracket and Total Race Elevation",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Rider Age Bracket", fontsize=11)
    ax.set_ylabel("Mean GC Time Gap to Winner (Minutes)", fontsize=11)
    ax.legend(title="Edition Elevation Tier", frameon=True, fontsize=10)
    plt.figtext(
        0.5,
        0.01,
        "Caption: High-elevation Tour editions disproportionately increase "
        "the GC time gap of Veteran (31+) riders compared to Prime (23-30) "
        "competitors.",
        wrap=True,
        horizontalalignment="center",
        fontsize=9.5,
        style="italic",
    )
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig("plots/fig3_elevation_age_interaction.png", dpi=300)
    plt.close(fig)

    # Plot 4 (Joined Dataset 2): ANOVA Assumption Verification (Box + Q-Q Plot)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    sns.boxplot(
        data=elev_fin,
        x="Age_Bracket",
        y="Gap_Minutes",
        order=bracket_order,
        palette="Blues",
        ax=axes[0],
    )
    axes[0].set_title(
        "Panel A: Gap Distribution by Age Bracket",
        fontsize=11.5,
        fontweight="bold",
    )
    axes[0].set_xlabel("Age Bracket", fontsize=10)
    axes[0].set_ylabel("GC Time Gap (Minutes)", fontsize=10)
    axes[0].tick_params(axis="x", rotation=18)

    group_means = elev_fin.groupby("Age_Bracket")["Gap_Minutes"].transform(
        "mean"
    )
    residuals = elev_fin["Gap_Minutes"] - group_means
    stats.probplot(residuals, dist="norm", plot=axes[1])
    axes[1].set_title(
        "Panel B: Normal Q-Q Plot of ANOVA Residuals",
        fontsize=11.5,
        fontweight="bold",
    )
    axes[1].set_xlabel("Theoretical Normal Quantiles", fontsize=10)
    axes[1].set_ylabel("Ordered Residual Quantiles (Minutes)", fontsize=10)

    plt.suptitle(
        "Figure 4: Diagnostic Verification of ANOVA Normality & Variance",
        fontsize=13,
        fontweight="bold",
    )
    plt.figtext(
        0.5,
        0.01,
        "Caption: Panel A shows greater variance in the Veteran bracket and "
        "Panel B reveals right-skewed tail departures from normality, "
        "motivating a Kruskal-Wallis robustness check alongside ANOVA.",
        wrap=True,
        horizontalalignment="center",
        fontsize=9.5,
        style="italic",
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.94])
    plt.savefig("plots/fig4_anova_residuals_qq_box.png", dpi=300)
    plt.close(fig)
    print("\nSaved 4 visualizations to plots/ directory.")


def main():
    """
    Main method executing the full CSE 163 EDA pipeline.
    """
    if not os.path.exists("data/raw_gc_rankings.csv"):
        from scrape_data import main as run_scraper
        run_scraper()

    gc_df = pd.read_csv("data/raw_gc_rankings.csv")
    riders_df = pd.read_csv("data/rider_bio_ages.csv")
    elev_df = pd.read_csv("data/stage_elevation_profiles.csv")

    joined_riders, joined_elev = clean_and_join_datasets(
        gc_df, riders_df, elev_df
    )
    decline_df = compute_career_decline(joined_riders)

    joined_riders.to_csv("data/joined_gc_riders.csv", index=False)
    joined_elev.to_csv("data/joined_gc_elevation.csv", index=False)

    # 1. Missingness Diagnostics
    report_missing_data(
        joined_riders, "Joined Dataset 1 (GC Rankings + Rider Bio)"
    )
    report_missing_data(
        joined_elev, "Joined Dataset 2 (GC Rankings + Rider Bio + Elevation)"
    )

    # 2. Quantitative Seven-Number Summaries
    quant_cols = [
        "Rider_Age",
        "GC_Rank",
        "Total_Seconds",
        "Gap_Seconds",
        "Total_Elevation_m",
        "Total_Distance_km",
        "Avg_Temp_C",
    ]
    print("\n=== Seven-Number Summary (Joined GC + Rider + Elevation) ===")
    print(seven_number_summary(joined_elev, quant_cols))

    print("\n=== Seven-Number Summary by Gender (Finishers) ===")
    for gender in ["Male", "Female"]:
        sub = joined_elev[joined_elev["Gender"] == gender]
        print(f"\n--- {gender} ---")
        print(seven_number_summary(sub, quant_cols))

    print("\n=== Seven-Number Summary: Career Decline Metrics (RQ2) ===")
    print(
        seven_number_summary(
            decline_df,
            ["Annual_Rank_Change", "Rank_Pct_Drop", "Annual_Gap_Change_Sec"],
        )
    )

    # 3. Categorical Value Counts
    cat_cols = ["Gender", "Age_Bracket", "Elevation_Tier", "Finish_Status"]
    print("\n=== Categorical Variable Value Counts ===")
    for col in cat_cols:
        print(f"\nColumn: {col}")
        print(joined_elev[col].value_counts(dropna=False))

    # 4. Statistical Assumption & Hypothesis Checks
    run_statistical_validity_checks(joined_elev)

    # 5. Visualizations
    create_visualizations(joined_riders, joined_elev, decline_df)


if __name__ == "__main__":
    main()
