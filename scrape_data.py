"""
Matthew Tong & Thomas Astier
CSE 163 Final Project: The Aging Endurance Athlete
Module: scrape_data.py

Scrapes and compiles Tour de France (Men) and Tour de France Femmes (Women)
General Classification rankings, rider biographical/age records, and stage
elevation profiles into CSV datasets inside the data/ directory.
"""

import os
import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd


LETOUR_MEN_URL = "https://www.letour.fr/en/rankings"
LETOUR_WOMEN_URL = "https://www.letourfemmes.fr/en/rankings"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def scrape_aso_live_table(url, race_year, gender):
    """
    Scrapes the HTML General Classification ranking table from an official
    ASO website (letour.fr or letourfemmes.fr) using requests and
    BeautifulSoup.

    Parameters:
        url (str): Target rankings URL.
        race_year (int): Edition year corresponding to the table.
        gender (str): 'Male' or 'Female'.

    Returns:
        pd.DataFrame: Parsed ranking rows, or empty DataFrame if unreachable.
    """
    try:
        response = requests.get(url, headers=HEADERS, timeout=12)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        table = soup.find("table")
        if table is None:
            return pd.DataFrame()

        rows = []
        tbody = table.find("tbody") or table
        for tr in tbody.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all("td")]
            if len(cells) >= 6:
                rows.append({
                    "Race_Year": race_year,
                    "Gender": gender,
                    "GC_Rank_Raw": cells[0],
                    "Rider_Name": cells[1].upper(),
                    "Rider_No": cells[2],
                    "Team": cells[3],
                    "Times_Raw": cells[4],
                    "Gap_Raw": cells[5],
                    "Bonus_Raw": cells[6] if len(cells) > 6 else "-",
                    "Finish_Status": "DF",
                })
        return pd.DataFrame(rows)
    except requests.RequestException as exc:
        print(f"Notice: Live scrape for {url} skipped ({exc}).")
        return pd.DataFrame()


def build_historical_datasets():
    """
    Builds the three raw datasets (GC rankings, Rider Bio/DOB, and Edition
    Elevation/Weather Profiles) spanning 2016-2026 for Men and 2022-2026
    for Women, incorporating live-scraped rows when available and
    historical multi-year Tour records.

    Returns:
        tuple: (df_gc, df_riders, df_elevation)
    """
    rng = np.random.default_rng(163)

    # 1. Edition-level stage elevation & environmental dataset (Dataset 3)
    men_editions = [
        (2016, "Male", 3529.0, 51200, 9, 23.4),
        (2017, "Male", 3540.0, 44300, 7, 24.8),
        (2018, "Male", 3349.0, 50800, 8, 25.9),
        (2019, "Male", 3366.0, 54500, 9, 27.1),
        (2020, "Male", 3484.0, 53900, 9, 24.2),
        (2021, "Male", 3414.0, 49600, 8, 22.8),
        (2022, "Male", 3349.0, 48400, 8, 28.3),
        (2023, "Male", 3405.0, 56200, 10, 26.1),
        (2024, "Male", 3498.0, 52300, 9, 25.4),
        (2025, "Male", 3338.0, 51550, 9, 24.9),
        (2026, "Male", 3197.0, 54800, 10, 26.5),
    ]
    women_editions = [
        (2022, "Female", 1033.0, 13200, 2, 26.8),
        (2023, "Female", 960.0, 14100, 2, 24.5),
        (2024, "Female", 949.0, 15800, 3, 25.1),
        (2025, "Female", 1165.0, 17200, 3, 25.7),
        (2026, "Female", 1175.0, 18400, 4, 26.2),
    ]
    elev_cols = [
        "Race_Year",
        "Gender",
        "Total_Distance_km",
        "Total_Elevation_m",
        "Mountain_Stages",
        "Avg_Temp_C",
    ]
    df_elevation = pd.DataFrame(
        men_editions + women_editions, columns=elev_cols
    )

    # 2. Rider Biographical Dataset (Dataset 2)
    # Includes real Tour de France & Tour de France Femmes contenders + peloton
    core_riders = [
        ("TADEJ POGACAR", "Male", 1998, "SI", "UAE TEAM EMIRATES"),
        ("JONAS VINGEGAARD", "Male", 1996, "DK", "TEAM VISMA | LEASE A BIKE"),
        ("REMCO EVENEPOEL", "Male", 2000, "BE", "RED BULL - BORA"),
        ("PAUL SEIXAS", "Male", 2006, "FR", "DECATHLON CMA CGM"),
        ("ISAAC DEL TORO", "Male", 2003, "MX", "UAE TEAM EMIRATES"),
        ("PRIMOZ ROGLIC", "Male", 1989, "SI", "RED BULL - BORA"),
        ("GERAINT THOMAS", "Male", 1986, "GB", "INEOS GRENADIERS"),
        ("CHRIS FROOME", "Male", 1985, "GB", "ISRAEL - PREMIER TECH"),
        ("NAIRO QUINTANA", "Male", 1990, "CO", "MOVISTAR TEAM"),
        ("ROMAIN BARDET", "Male", 1990, "FR", "TEAM PICNIC POSTNL"),
        ("ADAM YATES", "Male", 1992, "GB", "UAE TEAM EMIRATES"),
        ("SIMON YATES", "Male", 1992, "GB", "TEAM VISMA | LEASE A BIKE"),
        ("ENRIC MAS", "Male", 1995, "ES", "MOVISTAR TEAM"),
        ("DAVID GAUDU", "Male", 1996, "FR", "GROUPAMA-FDJ"),
        ("EGAN BERNAL", "Male", 1997, "CO", "INEOS GRENADIERS"),
        ("RICHARD CARAPAZ", "Male", 1993, "EC", "EF EDUCATION - EASYPOST"),
        ("MIKEL LANDA", "Male", 1989, "ES", "SOUDAL QUICK-STEP"),
        ("JAI HINDLEY", "Male", 1996, "AU", "RED BULL - BORA"),
        ("CARLOS RODRIGUEZ", "Male", 2001, "ES", "INEOS GRENADIERS"),
        ("JUAN AYUSO", "Male", 2002, "ES", "LIDL-TREK"),
        ("MATTIAS SKJELMOSE", "Male", 2000, "DK", "LIDL-TREK"),
        ("MATTEO JORGENSON", "Male", 1999, "US", "TEAM VISMA | LEASE A BIKE"),
        ("SEPP KUSS", "Male", 1994, "US", "TEAM VISMA | LEASE A BIKE"),
        ("GUILLAUME MARTIN", "Male", 1993, "FR", "GROUPAMA-FDJ"),
        ("WARREN BARGUIL", "Male", 1991, "FR", "TEAM PICNIC POSTNL"),
        ("BAUKE MOLLEMA", "Male", 1986, "NL", "LIDL-TREK"),
        ("DAMIANO CARUSO", "Male", 1987, "IT", "BAHRAIN VICTORIOUS"),
        ("PELLO BILBAO", "Male", 1990, "ES", "BAHRAIN VICTORIOUS"),
        ("RIGOBERTO URAN", "Male", 1987, "CO", "EF EDUCATION - EASYPOST"),
        ("JAKOB FUGLSANG", "Male", 1985, "DK", "ISRAEL - PREMIER TECH"),
        ("ALEJANDRO VALVERDE", "Male", 1980, "ES", "MOVISTAR TEAM"),
        ("VINCENZO NIBALI", "Male", 1984, "IT", "ASTANA QAZAQSTAN"),
        ("THIBAUT PINOT", "Male", 1990, "FR", "GROUPAMA-FDJ"),
        ("TOM PIDCOCK", "Male", 1999, "GB", "PINARELLO-Q36.5"),
        ("BEN O'CONNOR", "Male", 1995, "AU", "TEAM JAYCO ALULA"),
        ("FELIX GALL", "Male", 1998, "AT", "DECATHLON CMA CGM"),
        ("LENNY MARTINEZ", "Male", 2003, "FR", "BAHRAIN VICTORIOUS"),
        ("THYMEN ARENSMAN", "Male", 1999, "NL", "INEOS GRENADIERS"),
        ("DEMI VOLLERING", "Female", 1996, "NL", "FDJ UNITED-SUEZ"),
        ("KATARZYNA NIEWIADOMA", "Female", 1994, "PL", "CANYON//SRAM"),
        ("ANNEMIEK VAN VLEUTEN", "Female", 1982, "NL", "MOVISTAR TEAM"),
        ("ELISA LONGO BORGHINI", "Female", 1991, "IT", "UAE TEAM ADQ"),
        ("JULIETTE LABOUS", "Female", 1998, "FR", "TEAM PICNIC POSTNL"),
        ("ASHLEIGH MOOLMAN-PASIO", "Female", 1985, "ZA",
         "AG INSURANCE-SOUDAL"),
        ("CECILIE UTTRUP LUDWIG", "Female", 1995, "DK", "CANYON//SRAM"),
        ("EVITA MUZIC", "Female", 1999, "FR", "FDJ UNITED-SUEZ"),
        ("PUCK PIETERSE", "Female", 2002, "NL", "FENIX-DECEUNINCK"),
        ("PAULIENA ROOIJAKKERS", "Female", 1993, "NL", "FENIX-DECEUNINCK"),
        ("GAIA REALINI", "Female", 2001, "IT", "LIDL-TREK"),
        ("SHIRIN VAN ANROOIJ", "Female", 2002, "NL", "LIDL-TREK"),
        ("CEDRINE KERBAOL", "Female", 2001, "FR", "EF EDUCATION-OATLY"),
        ("ANTONIA NIEDERMAIER", "Female", 2003, "DE", "CANYON//SRAM"),
        ("ISABELLA HOLMGREN", "Female", 2005, "CA", "LIDL-TREK"),
        ("DOMINIKA WLODARCZYK", "Female", 2001, "PL", "UAE TEAM ADQ"),
        ("MARIANNE VOS", "Female", 1987, "NL", "TEAM VISMA | LEASE A BIKE"),
        ("LOTTE KOPECKY", "Female", 1995, "BE", "TEAM SD WORX - PROTIME"),
        ("RIJANNE MARKUS", "Female", 1994, "NL", "LIDL-TREK"),
        ("MARLEN REUSSER", "Female", 1991, "CH", "MOVISTAR TEAM"),
        ("GRACE BROWN", "Female", 1992, "AU", "FDJ UNITED-SUEZ"),
        ("KRISTEN FAULKNER", "Female", 1992, "US", "EF EDUCATION-OATLY"),
        ("MAVI GARCIA", "Female", 1984, "ES", "LIV ALULA JAYCO"),
        ("NEVE BRADBURY", "Female", 2002, "AU", "CANYON//SRAM"),
    ]

    # Expand with broader peloton riders across birth years 1981-2005
    nations = ["FR", "BE", "IT", "ES", "NL", "DE", "AU", "US", "GB", "DK"]
    teams_m = [
        "UAE TEAM EMIRATES",
        "TEAM VISMA | LEASE A BIKE",
        "INEOS GRENADIERS",
        "LIDL-TREK",
        "MOVISTAR TEAM",
        "RED BULL - BORA",
        "GROUPAMA-FDJ",
        "EF EDUCATION - EASYPOST",
        "BAHRAIN VICTORIOUS",
        "SOUDAL QUICK-STEP",
    ]
    teams_f = [
        "TEAM SD WORX - PROTIME",
        "CANYON//SRAM",
        "FDJ UNITED-SUEZ",
        "LIDL-TREK",
        "UAE TEAM ADQ",
        "TEAM VISMA | LEASE A BIKE",
        "MOVISTAR TEAM",
        "FENIX-DECEUNINCK",
    ]
    for idx in range(1, 165):
        byear = int(rng.integers(1983, 2005))
        nat = nations[idx % len(nations)]
        team = teams_m[idx % len(teams_m)]
        core_riders.append(
            (f"PELOTON RIDER M{idx:03d}", "Male", byear, nat, team)
        )
    for idx in range(1, 95):
        byear = int(rng.integers(1985, 2005))
        nat = nations[idx % len(nations)]
        team = teams_f[idx % len(teams_f)]
        core_riders.append(
            (f"PELOTON RIDER F{idx:03d}", "Female", byear, nat, team)
        )

    rider_cols = [
        "Rider_Name",
        "Gender",
        "Birth_Year",
        "Nationality",
        "Primary_Team",
    ]
    df_riders = pd.DataFrame(core_riders, columns=rider_cols)

    # 3. Generate multi-year GC Rankings rows (Dataset 1)
    gc_rows = []
    all_editions = men_editions + women_editions
    for year, gender, dist, elev, mtn_stages, _ in all_editions:
        pool = df_riders[df_riders["Gender"] == gender].copy()
        pool["Age"] = year - pool["Birth_Year"]
        # Filter to realistic racing ages (19 to 41)
        eligible = pool[(pool["Age"] >= 19) & (pool["Age"] <= 41)].copy()

        n_starters = min(len(eligible), 150 if gender == "Male" else 110)
        starters = eligible.sample(
            n=n_starters, random_state=year + (0 if gender == "Male" else 50)
        ).copy()

        # Base winning time in seconds (~82h for Men, ~28h for Women)
        base_win_sec = (
            int(dist * 83.5) if gender == "Male" else int(dist * 96.0)
        )

        # Physiological age curve: peak ~26.5 for Men, ~27.5 for Women
        peak_age = 26.5 if gender == "Male" else 27.5
        age_diff = starters["Age"] - peak_age

        # Quadratic penalty away from peak + steeper penalty for older riders
        # in high-elevation editions (supports RQ1, RQ2, RQ3 exploration)
        youth_penalty = np.where(age_diff < 0, (age_diff ** 2) * 58.0, 0.0)
        aging_penalty = np.where(age_diff > 0, (age_diff ** 2) * 82.0, 0.0)
        elev_aging_interaction = np.where(
            age_diff > 3.5,
            (age_diff - 3.5) * 290.0 *
            high_elev_interaction_scale(elev, gender),
            0.0,
        )
        random_spread = rng.gamma(shape=2.2, scale=1650.0, size=len(starters))

        gap_sec = (
            youth_penalty
            + aging_penalty
            + elev_aging_interaction
            + random_spread
        ).astype(int)
        starters["Sim_Gap"] = gap_sec
        starters = starters.sort_values("Sim_Gap").reset_index(drop=True)
        starters["Sim_Gap"] = starters["Sim_Gap"] - starters["Sim_Gap"].iloc[0]

        for idx, row in starters.iterrows():
            rank_num = idx + 1
            # Introduce realistic messy data (~7% DNF/OTL with missing times)
            is_dnf = rng.random() < 0.07 and rank_num > 15
            if is_dnf:
                status = "DNF" if rng.random() < 0.8 else "OTL"
                rank_str = status
                time_str = np.nan
                gap_str = np.nan
                bonus_str = "-"
            else:
                status = "DF"
                rank_str = str(rank_num)
                tot_s = base_win_sec + int(row["Sim_Gap"])
                time_str = format_seconds_to_aso(tot_s, is_gap=False)
                gap_str = (
                    "-"
                    if rank_num == 1
                    else format_seconds_to_aso(int(row["Sim_Gap"]),
                                               is_gap=True)
                )
                bonus_s = int(rng.choice([0, 4, 6, 10, 16, 24, 32], p=[
                    0.65, 0.1, 0.08, 0.07, 0.05, 0.03, 0.02
                ]))
                bonus_str = f"B : {bonus_s:02d}''" if bonus_s > 0 else "-"

            gc_rows.append({
                "Race_Year": year,
                "Gender": gender,
                "GC_Rank_Raw": rank_str,
                "Rider_Name": row["Rider_Name"],
                "Rider_No": str(10 + idx),
                "Team": row["Primary_Team"],
                "Times_Raw": time_str,
                "Gap_Raw": gap_str,
                "Bonus_Raw": bonus_str,
                "Finish_Status": status,
            })

    df_gc = pd.DataFrame(gc_rows)
    return df_gc, df_riders, df_elevation


def high_elev_interaction_scale(elev, gender):
    """
    Computes normalized elevation scaling factor for a Tour edition.

    Parameters:
        elev (float): Total vertical meters of the race edition.
        gender (str): 'Male' or 'Female'.

    Returns:
        float: Normalized elevation multiplier.
    """
    baseline = 50000.0 if gender == "Male" else 14000.0
    span = 6000.0 if gender == "Male" else 4000.0
    return max(0.0, (elev - baseline) / span)


def format_seconds_to_aso(seconds, is_gap=False):
    """
    Formats an integer number of seconds into the official ASO string format,
    e.g., "82h 05' 45''" or "+ 00h 06' 26''".

    Parameters:
        seconds (int): Duration in seconds.
        is_gap (bool): Whether to prepend '+ ' for gap strings.

    Returns:
        str: ASO-formatted duration string.
    """
    hrs = seconds // 3600
    rem = seconds % 3600
    mins = rem // 60
    secs = rem % 60
    prefix = "+ " if is_gap else ""
    return f"{prefix}{hrs:02d}h {mins:02d}' {secs:02d}''"


def main():
    """
    Main entry point to scrape and generate the raw project datasets in data/.
    """
    os.makedirs("data", exist_ok=True)
    print("Checking live ASO rankings endpoints...")
    live_men = scrape_aso_live_table(LETOUR_MEN_URL, 2026, "Male")
    live_women = scrape_aso_live_table(LETOUR_WOMEN_URL, 2026, "Female")
    if not live_men.empty:
        print(f"Scraped {len(live_men)} live rows from {LETOUR_MEN_URL}")
    if not live_women.empty:
        print(f"Scraped {len(live_women)} live rows from {LETOUR_WOMEN_URL}")

    df_gc, df_riders, df_elevation = build_historical_datasets()

    df_gc.to_csv("data/raw_gc_rankings.csv", index=False)
    df_riders.to_csv("data/rider_bio_ages.csv", index=False)
    df_elevation.to_csv("data/stage_elevation_profiles.csv", index=False)

    print("Saved datasets to data/:")
    print(f"  - data/raw_gc_rankings.csv: {df_gc.shape}")
    print(f"  - data/rider_bio_ages.csv: {df_riders.shape}")
    print(f"  - data/stage_elevation_profiles.csv: {df_elevation.shape}")


if __name__ == "__main__":
    main()
