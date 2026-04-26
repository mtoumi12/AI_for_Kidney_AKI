"""
data.py — MIMIC-IV cohort extraction
Builds the AKI feature table from raw MIMIC-IV tables via SQLite.
"""

import sqlite3
import pandas as pd

# ── Item IDs ─────────────────────────────────────────────────────────────────
CREATININE_ITEMID  = 50912
HEMOGLOBIN_ITEMID  = 50811
MAP_ITEMIDS        = (220052, 220181)   # arterial + non-invasive
HR_ITEMID          = 220045
FOLEY_ITEMID       = 226559

COHORT_QUERY = """
WITH
baseline_creat AS (
    SELECT hadm_id,
           valuenum AS creatinine,
           charttime
    FROM   labevents
    WHERE  itemid   = {creatinine}
      AND  valuenum BETWEEN 0.3 AND 15
    GROUP BY hadm_id
    HAVING charttime = MIN(charttime)
),
peak_creat AS (
    SELECT hadm_id,
           MAX(valuenum) AS peak_creatinine
    FROM   labevents
    WHERE  itemid   = {creatinine}
      AND  valuenum BETWEEN 0.3 AND 15
    GROUP BY hadm_id
),
first_hgb AS (
    SELECT hadm_id,
           valuenum AS hemoglobin
    FROM   labevents
    WHERE  itemid   = {hemoglobin}
      AND  valuenum BETWEEN 4 AND 20
    GROUP BY hadm_id
    HAVING charttime = MIN(charttime)
),
hemodynamics AS (
    SELECT i.hadm_id,
           ROUND(AVG(c.valuenum), 1)   AS mean_map,
           ROUND(MIN(c.valuenum), 1)   AS min_map,
           ROUND(
               100.0 * SUM(CASE WHEN c.valuenum < 65 THEN 1 ELSE 0 END)
               / COUNT(*), 1)          AS pct_hypotensive,
           ROUND(AVG(CASE WHEN c.itemid = {hr} THEN c.valuenum END), 1)
                                       AS mean_hr
    FROM   chartevents c
    JOIN   icustays    i ON c.stay_id = i.stay_id
    WHERE  c.itemid IN ({map1}, {map2}, {hr})
      AND  c.valuenum BETWEEN 20 AND 250
    GROUP BY i.hadm_id
),
icu_summary AS (
    SELECT hadm_id,
           COUNT(*)         AS n_icu_stays,
           ROUND(SUM(los), 1) AS total_icu_los_days
    FROM icustays
    GROUP BY hadm_id
)
SELECT
    a.hadm_id,
    a.subject_id,
    p.anchor_age                                            AS age,
    p.gender,
    a.admission_type,
    a.insurance,
    a.race,
    ROUND(julianday(a.dischtime) - julianday(a.admittime), 1) AS los_days,
    a.hospital_expire_flag                                  AS died_in_hospital,
    -- Baseline biology
    ROUND(b.creatinine, 2)                                  AS baseline_creatinine,
    ROUND(h.hemoglobin, 1)                                  AS baseline_hemoglobin,
    -- eGFR (MDRD)
    ROUND(
        186.0
        * POWER(b.creatinine, -1.154)
        * POWER(p.anchor_age, -0.203)
        * CASE WHEN p.gender = 'F' THEN 0.742 ELSE 1.0 END
    , 1)                                                    AS egfr,
    -- Peak / delta creatinine
    ROUND(pk.peak_creatinine, 2)                            AS peak_creatinine,
    ROUND(pk.peak_creatinine - b.creatinine, 2)             AS delta_creatinine,
    ROUND(pk.peak_creatinine / NULLIF(b.creatinine, 0), 2)  AS creatinine_ratio,
    -- ICU hemodynamics
    COALESCE(hd.mean_map, -1)                               AS mean_map,
    COALESCE(hd.min_map,  -1)                               AS min_map,
    COALESCE(hd.pct_hypotensive, -1)                        AS pct_hypotensive,
    COALESCE(hd.mean_hr, -1)                                AS mean_hr,
    -- ICU stay
    COALESCE(icu.n_icu_stays, 0)                            AS n_icu_stays,
    COALESCE(icu.total_icu_los_days, 0)                     AS icu_los_days,
    -- AKI label (KDIGO from creatinine)
    CASE
        WHEN pk.peak_creatinine >= 1.5 * b.creatinine       THEN 1
        WHEN pk.peak_creatinine - b.creatinine  >= 0.3      THEN 1
        ELSE 0
    END                                                     AS aki_label,
    -- AKI stage
    CASE
        WHEN pk.peak_creatinine >= 3.0 * b.creatinine       THEN 3
        WHEN pk.peak_creatinine >= 2.0 * b.creatinine       THEN 2
        WHEN pk.peak_creatinine >= 1.5 * b.creatinine       THEN 1
        WHEN pk.peak_creatinine - b.creatinine >= 0.3       THEN 1
        ELSE 0
    END                                                     AS aki_stage
FROM       admissions     a
JOIN       patients       p   ON a.subject_id = p.subject_id
LEFT JOIN  baseline_creat b   ON a.hadm_id    = b.hadm_id
LEFT JOIN  peak_creat     pk  ON a.hadm_id    = pk.hadm_id
LEFT JOIN  first_hgb      h   ON a.hadm_id    = h.hadm_id
LEFT JOIN  hemodynamics   hd  ON a.hadm_id    = hd.hadm_id
LEFT JOIN  icu_summary    icu ON a.hadm_id    = icu.hadm_id
WHERE b.creatinine IS NOT NULL
ORDER BY a.hadm_id
""".format(
    creatinine=CREATININE_ITEMID,
    hemoglobin=HEMOGLOBIN_ITEMID,
    map1=MAP_ITEMIDS[0],
    map2=MAP_ITEMIDS[1],
    hr=HR_ITEMID,
)


def load_mimic(base_path: str) -> sqlite3.Connection:
    """Load MIMIC-IV demo CSV files into an in-memory SQLite database."""
    tables = {
        "patients":      f"{base_path}/hosp/patients.csv",
        "admissions":    f"{base_path}/hosp/admissions.csv",
        "labevents":     f"{base_path}/hosp/labevents.csv",
        "prescriptions": f"{base_path}/hosp/prescriptions.csv",
        "diagnoses_icd": f"{base_path}/hosp/diagnoses_icd.csv",
        "icustays":      f"{base_path}/icu/icustays.csv",
        "chartevents":   f"{base_path}/icu/chartevents.csv",
        "d_items":       f"{base_path}/icu/d_items.csv",
    }
    conn = sqlite3.connect(":memory:")
    for name, path in tables.items():
        pd.read_csv(path).to_sql(name, conn, index=False, if_exists="replace")
        print(f"  Loaded {name}")
    return conn


def build_cohort(conn: sqlite3.Connection) -> pd.DataFrame:
    """Run the extraction query and return the analysis-ready cohort DataFrame."""
    df = pd.read_sql_query(COHORT_QUERY, conn)
    return df
