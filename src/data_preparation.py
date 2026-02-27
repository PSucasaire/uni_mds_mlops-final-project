from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def load_raw(raw_path: Path) -> pd.DataFrame:
    """Load raw parquet exported from ENDES harmonization step."""
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw file not found: {raw_path}")
    return pd.read_parquet(raw_path)


def filter_adults_complete(df: pd.DataFrame) -> pd.DataFrame:
    """Keep adults (>=18) with complete interview."""
    df = df.copy()
    df["HV105_num"] = pd.to_numeric(df["HV105"], errors="coerce")
    mask = (df["HV105_num"] >= 18) & (df["QSRESULT"] == "Completa")
    return df.loc[mask].copy()


def make_target(df: pd.DataFrame) -> pd.DataFrame:
    """Create binary target 'diabetes' from QS109 (Si/No)."""
    df = df.copy()
    df["diabetes"] = df["QS109"].map({"Si": 1, "No": 0})
    return df


def ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce selected columns to numeric if present."""
    df = df.copy()
    num_cols = ["QS203C", "QS205C", "QS213C", "QS219C", "HV105", "HV271", "HV005", "HV040"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering (minimal but aligned with your notebook logic).
    NOTE: This script does not try to rebuild .sav merges; it starts from the raw parquet.
    """
    df = df.copy()

    # Basic derived columns
    df["edad"] = pd.to_numeric(df["HV105"], errors="coerce")

    # Target-related binary maps
    df["mujer"] = df["HV104"].map({"Mujer": 1, "Hombre": 0})
    df["urbano"] = df["HV025"].map({"Urbano": 1, "Rural": 0})

    # Seguro
    df["tiene_seguro"] = df["QS26"].map({"Si": 1, "No": 0})

    # Riqueza
    map_riqueza = {
        "Los más pobres": 1,
        "Pobre": 2,
        "Medio": 3,
        "Rico": 4,
        "Más rico": 5,
    }
    if "HV270" in df.columns:
        df["riqueza_quintil"] = df["HV270"].map(map_riqueza)
    if "HV271" in df.columns:
        df["riqueza_score"] = df["HV271"]

    # Binarize selected Si/No variables (if present)
    bin_si_no = [
        "QS100", "QS102", "QS104", "QS106", "QS107", "QS111", "QS113",
        "QS202", "QS200", "QS201",
        "QS206", "QS209", "QS210",
        "QS26", "HV225"
    ]
    for col in bin_si_no:
        if col in df.columns:
            df[col + "_bin"] = df[col].map({"Si": 1, "No": 0})

    # Smoking block (optional features)
    if "QS202" in df.columns:
        df["fuma_diario"] = df["QS202"].map({"Si": 1, "No": 0})

    if "QS203C" in df.columns and "edad" in df.columns:
        df["edad_inicio_fumar"] = df["QS203C"]
        df["años_fumando"] = df["edad"] - df["edad_inicio_fumar"]
        df.loc[df["años_fumando"] < 0, "años_fumando"] = np.nan

    if "QS205C" in df.columns:
        df["cigs_dia"] = df["QS205C"]

    # Alcohol
    if "QS206" in df.columns:
        df["alcohol_ever"] = df["QS206"].map({"Si": 1, "No": 0})
    if "QS209" in df.columns:
        df["alcohol_12m"] = df["QS209"].map({"Si": 1, "No": 0})
    if "QS210" in df.columns:
        df["alcohol_30d"] = df["QS210"].map({"Si": 1, "No": 0})

    # Fruits / vegetables frequency
    if "QS213C" in df.columns:
        df["dias_fruta"] = df["QS213C"]
    if "QS219C" in df.columns:
        df["dias_verdura"] = df["QS219C"]

    # Altitude
    if "HV040" in df.columns:
        df["altitud"] = df["HV040"]

    # ubigeo as string (if present)
    if "UBIGEO" in df.columns:
        df["ubigeo"] = df["UBIGEO"].astype(str)

    return df


def select_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build final modeling dataframe:
    - keep rows with diabetes known (0/1)
    - keep a curated set of columns (can be expanded later)
    """
    y_col = "diabetes"

    # Keep set (minimal & stable). You can add more later.
    cols_keep = [
        y_col,
        "HV005", "ID1", "ubigeo",
        "edad", "mujer",
        "urbano",
        "riqueza_quintil", "riqueza_score",
        "tiene_seguro",
        # some health & habits
        "QS100_bin", "QS102_bin", "QS107_bin",
        "fuma_diario", "años_fumando",
        "alcohol_12m", "alcohol_30d",
        "dias_fruta", "dias_verdura",
        "altitud",
    ]
    cols_keep = [c for c in cols_keep if c in df.columns]

    out = df.loc[df[y_col].notna(), cols_keep].copy()

    # Ensure unique column names (avoid parquet write issues)
    out = out.loc[:, ~out.columns.duplicated()].copy()

    # Diabetes should be int for modeling
    out[y_col] = out[y_col].astype(int)

    return out


def main(raw_path: str, out_path: str) -> None:
    raw_path = Path(raw_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_raw(raw_path)
    df = filter_adults_complete(df)
    df = ensure_numeric(df)
    df = make_target(df)
    df = add_features(df)
    df_model = select_model_frame(df)

    df_model.to_parquet(out_path, index=False)

    print("✅ Data preparation completed")
    print("Input:", raw_path)
    print("Output:", out_path)
    print("Shape:", df_model.shape)
    print("Target counts:\n", df_model["diabetes"].value_counts())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare ENDES diabetes dataset for training.")
    parser.add_argument(
        "--raw-path",
        type=str,
        default="data/raw/endes_diabetes_model.parquet",
        help="Path to raw parquet file.",
    )
    parser.add_argument(
        "--out-path",
        type=str,
        default="data/training/endes_diabetes_features.parquet",
        help="Path to output parquet file.",
    )
    args = parser.parse_args()
    main(args.raw_path, args.out_path)
