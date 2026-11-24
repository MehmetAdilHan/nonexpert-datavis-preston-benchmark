#!/usr/bin/env python3
"""
Deterministic, Zero-Fabrication World Bank WDI & WHO GHO Dataset Pipeline (v5)
==============================================================================
Fixes the `scaled_full` variable typo in the Wasserstein-1 audit step ([4/6])
and reuses all raw JSON payloads already cached in:
    ./wdi_who_verified_output/raw_wdi_api_cache/

Usage:
    python3 reproduce_wdi_who_dataset-v5.py
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------------------------
# Configuration (Deterministic & Explicitly Documented)
# ---------------------------------------------------------------------------
RANDOM_SEED = 20261115
YEAR_START = 2015
YEAR_END = 2023  # 2015-2023: latest complete multi-indicator WDI/WHO window
MIN_OBSERVED_INDICATORS = 5  # Require >= 5/8 indicators per country for KNN
KNN_NEIGHBORS = 5
CV_FOLDS = 5
WARMUP_COUNTRIES = 15
API_PER_PAGE = 2000

OUT_DIR = Path("wdi_who_verified_output")
CACHE_DIR = OUT_DIR / "raw_wdi_api_cache"

INDICATORS: List[Dict[str, Optional[str]]] = [
    {
        "code": "SP.DYN.LE00.IN",
        "gho_fallback": None,
        "short": "LE",
        "name": "Life Expectancy at Birth (WDI)",
        "unit": "Years",
        "fmt": "{:.1f}",
    },
    {
        "code": "NY.GDP.PCAP.PP.CD",
        "gho_fallback": None,
        "short": "GDP_PC",
        "name": "GDP per Capita, PPP (WDI)",
        "unit": r"Intl \$",
        "fmt": "{:,.0f}",
    },
    {
        "code": "SH.XPD.CHEX.PC.CD",
        "gho_fallback": "GHED_CHEX_pc_US_SHA2011",
        "short": "CHEX_PC",
        "name": "Current Health Exp. / Capita (WHO)",
        "unit": r"Current US\$",
        "fmt": "{:,.0f}",
    },
    {
        "code": "SH.XPD.CHEX.GD.ZS",
        "gho_fallback": "GHED_CHEX_GD_SHA2011",
        "short": "CHEX_GDP",
        "name": r"Current Health Exp. (\% GDP) (WHO)",
        "unit": r"\% GDP",
        "fmt": "{:.2f}",
    },
    {
        "code": "SH.UHC.SRVS.CV.XD",
        "gho_fallback": "UHC_INDEX_REPORTED",
        "short": "UHC",
        "name": "UHC Service Coverage Index (WHO)",
        "unit": "0--100",
        "fmt": "{:.1f}",
    },
    {
        "code": "SH.XPD.OOPC.CH.ZS",
        "gho_fallback": "GHED_OOPSCHE_SHA2011",
        "short": "OOP",
        "name": "Out-of-Pocket Health Exp. (WHO)",
        "unit": r"\% CHE",
        "fmt": "{:.1f}",
    },
    {
        "code": "SH.DYN.MORT",
        "gho_fallback": "MDG_0000000007",
        "short": "U5M",
        "name": "Under-5 Mortality Rate (WDI/IGME)",
        "unit": "per 1,000",
        "fmt": "{:.1f}",
    },
    {
        "code": "SH.STA.MMRT",
        "gho_fallback": "MDG_0000000026",
        "short": "MMR",
        "name": "Maternal Mortality Ratio (WHO)",
        "unit": "per 100k",
        "fmt": "{:.1f}",
    },
]


class WbIndicatorUnavailableError(RuntimeError):
    """Raised when World Bank v2 API reports an archived or unavailable series."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def is_valid_wb_payload(payload: Any) -> bool:
    return (
        isinstance(payload, list)
        and len(payload) >= 2
        and isinstance(payload[0], dict)
        and isinstance(payload[1], list)
    )


def extract_wb_error(payload: Any) -> tuple[str, str]:
    if isinstance(payload, list) and len(payload) >= 1 and isinstance(payload[0], dict):
        if "id" in payload[0]:
            return str(payload[0].get("id", "")), json.dumps(payload[0])
        msgs = payload[0].get("message")
        if isinstance(msgs, list) and msgs:
            err_id = str(msgs[0].get("id", ""))
            return err_id, json.dumps(msgs)
    return "", f"Unexpected payload structure: {str(payload)[:300]}"


def fetch_raw_json(url: str, cache_path: Path, retries: int = 4) -> Any:
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            cache_path.unlink(missing_ok=True)

    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "FAST-NUCES-DataVis-Research-Audit/5.0"},
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read().decode("utf-8")
            parsed = json.loads(raw)
            cache_path.write_text(json.dumps(parsed, indent=2), encoding="utf-8")
            return parsed
        except Exception as exc:
            last_err = exc
            time.sleep(1.5 * attempt)
    raise RuntimeError(f"Failed to fetch {url} after {retries} attempts: {last_err}")


def fetch_wb_page(url: str, cache_path: Path, retries: int = 4) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if is_valid_wb_payload(cached):
                return cached[0], cached[1]
        except Exception:
            pass
        cache_path.unlink(missing_ok=True)

    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "FAST-NUCES-DataVis-Research-Audit/5.0"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read().decode("utf-8")
            parsed = json.loads(raw)
            if not is_valid_wb_payload(parsed):
                err_id, err_detail = extract_wb_error(parsed)
                if err_id in {"120", "175"}:
                    raise WbIndicatorUnavailableError(f"WB API {err_id} for {url}: {err_detail}")
                raise RuntimeError(f"World Bank API returned error for {url}: {err_detail}")
            cache_path.write_text(json.dumps(parsed, indent=2), encoding="utf-8")
            return parsed[0], parsed[1]
        except WbIndicatorUnavailableError:
            raise
        except Exception as exc:
            last_err = exc
            time.sleep(1.5 * attempt)
    raise RuntimeError(f"Failed to fetch {url} after {retries} attempts: {last_err}")


def fetch_wb_all_pages(base_url: str, cache_prefix: str) -> list[dict[str, Any]]:
    sep = "&" if "?" in base_url else "?"
    page1_url = f"{base_url}{sep}per_page={API_PER_PAGE}&page=1"
    meta, records = fetch_wb_page(page1_url, CACHE_DIR / f"{cache_prefix}_p1.json")
    all_records = list(records)
    total_pages = int(meta.get("pages", 1))
    for page in range(2, total_pages + 1):
        page_url = f"{base_url}{sep}per_page={API_PER_PAGE}&page={page}"
        _, page_records = fetch_wb_page(page_url, CACHE_DIR / f"{cache_prefix}_p{page}.json")
        all_records.extend(page_records)
    return all_records


def fetch_sovereign_countries() -> pd.DataFrame:
    entries = fetch_wb_all_pages(
        "https://api.worldbank.org/v2/country?format=json",
        "countries_metadata",
    )
    rows = []
    for entry in entries:
        region_id = entry.get("region", {}).get("id", "NA")
        if region_id == "NA":
            continue
        rows.append(
            {
                "iso3": entry["id"],
                "iso2": entry["iso2Code"],
                "country_name": entry["name"],
                "region": entry["region"]["value"],
                "income_level": entry["incomeLevel"]["value"],
            }
        )
    return pd.DataFrame(rows).sort_values("iso3").reset_index(drop=True)


def verify_indicator_metadata(code: str) -> str:
    try:
        entries = fetch_wb_all_pages(
            f"https://api.worldbank.org/v2/indicator/{code}?format=json",
            f"meta_{code}",
        )
        if entries:
            return str(entries[0].get("name", code))
    except Exception:
        pass
    return code


def fetch_who_gho_indicator(gho_code: str, wdi_code: str, valid_iso3: set[str]) -> tuple[pd.DataFrame, str]:
    """Fetch country-year series directly from official WHO GHO OData API."""
    url = f"https://ghoapi.azureedge.net/api/{gho_code}"
    payload = fetch_raw_json(url, CACHE_DIR / f"who_gho_{gho_code}.json")
    rows = payload.get("value", [])
    records = []
    for item in rows:
        if item.get("SpatialDimType") != "COUNTRY":
            continue
        iso3 = item.get("SpatialDim")
        if iso3 not in valid_iso3:
            continue
        dim1 = item.get("Dim1")
        if dim1 not in (None, "", "SEX_BTSX", "BTSX", "RESIDENCEAREATYPE_TOTL"):
            continue
        year_raw = item.get("TimeDim")
        if year_raw is None:
            continue
        yr = int(year_raw)
        if not (YEAR_START <= yr <= YEAR_END):
            continue
        val = item.get("NumericValue")
        if val is None and item.get("Value") is not None:
            try:
                val = float(item["Value"])
            except (ValueError, TypeError):
                val = None
        records.append(
            {
                "iso3": iso3,
                "year": yr,
                wdi_code: float(val) if val is not None else np.nan,
            }
        )
    df = pd.DataFrame(records)
    if not df.empty:
        df = df.groupby(["iso3", "year"], as_index=False)[wdi_code].mean()
    return df, url


def fetch_indicator_panel(
    ind: Dict[str, Optional[str]],
    valid_iso3: set[str],
) -> tuple[pd.DataFrame, str]:
    """Fetch country-year values via World Bank v2 API, with WHO GHO fallback."""
    code = str(ind["code"])
    gho_code = ind.get("gho_fallback")
    official_title = verify_indicator_metadata(code)

    wb_url = f"https://api.worldbank.org/v2/country/all/indicator/{code}?format=json"
    try:
        items = fetch_wb_all_pages(wb_url, f"indicator_{code}_default")
        records = []
        for item in items:
            iso3 = item.get("countryiso3code")
            if iso3 not in valid_iso3:
                continue
            yr = int(item["date"])
            if not (YEAR_START <= yr <= YEAR_END):
                continue
            val = item.get("value")
            records.append(
                {
                    "iso3": iso3,
                    "year": yr,
                    code: float(val) if val is not None else np.nan,
                }
            )
        df = pd.DataFrame(records)
        if not df.empty and df[code].notna().sum() > 0:
            print(f"    - {code} ({official_title}) -> fetched via World Bank v2 API")
            return df, wb_url
    except WbIndicatorUnavailableError:
        pass

    if gho_code:
        df_gho, gho_url = fetch_who_gho_indicator(gho_code, code, valid_iso3)
        if not df_gho.empty and df_gho[code].notna().sum() > 0:
            print(f"    - {code} ({official_title}) -> fetched via WHO GHO OData API [{gho_code}]")
            return df_gho, gho_url

    raise RuntimeError(f"Unable to retrieve non-empty data for {code} from WB API or WHO GHO.")


def run_leak_free_knn_cv(
    raw_matrix: np.ndarray,
    col_codes: List[str],
    seed: int = RANDOM_SEED,
    k: int = KNN_NEIGHBORS,
    n_splits: int = CV_FOLDS,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    obs_coords = np.argwhere(~np.isnan(raw_matrix))
    perm = rng.permutation(len(obs_coords))
    folds = np.array_split(perm, n_splits)

    norm_sq_errs: Dict[str, List[float]] = {c: [] for c in col_codes}
    orig_sq_errs: Dict[str, List[float]] = {c: [] for c in col_codes}
    orig_abs_errs: Dict[str, List[float]] = {c: [] for c in col_codes}

    for fold_idx in folds:
        masked = raw_matrix.copy()
        for r_i, c_i in obs_coords[fold_idx]:
            masked[r_i, c_i] = np.nan

        scaler = MinMaxScaler()
        scaled_train = scaler.fit_transform(masked)
        imputer = KNNImputer(n_neighbors=k, weights="distance", metric="nan_euclidean")
        scaled_imputed = imputer.fit_transform(scaled_train)
        orig_imputed = scaler.inverse_transform(scaled_imputed)
        true_scaled = scaler.transform(raw_matrix)

        for r_i, c_i in obs_coords[fold_idx]:
            code = col_codes[c_i]
            err_norm = float(scaled_imputed[r_i, c_i] - true_scaled[r_i, c_i])
            err_orig = float(orig_imputed[r_i, c_i] - raw_matrix[r_i, c_i])
            norm_sq_errs[code].append(err_norm ** 2)
            orig_sq_errs[code].append(err_orig ** 2)
            orig_abs_errs[code].append(abs(err_orig))

    per_col = {}
    all_norm_sq = []
    for code in col_codes:
        all_norm_sq.extend(norm_sq_errs[code])
        per_col[code] = {
            "n_cv_cells": len(norm_sq_errs[code]),
            "nrmse": float(np.sqrt(np.mean(norm_sq_errs[code]))),
            "rmse_original_units": float(np.sqrt(np.mean(orig_sq_errs[code]))),
            "mae_original_units": float(np.mean(orig_abs_errs[code])),
        }

    return {
        "overall_nrmse": float(np.sqrt(np.mean(all_norm_sq))),
        "per_indicator": per_col,
    }


def fit_preston_curve_ols(df_raw: pd.DataFrame) -> Dict[str, Any]:
    valid = df_raw.dropna(subset=["SP.DYN.LE00.IN", "SH.XPD.CHEX.PC.CD"]).copy()
    non_us = valid[valid["iso3"] != "USA"].copy()

    x = np.log(non_us["SH.XPD.CHEX.PC.CD"].to_numpy(dtype=float))
    y = non_us["SP.DYN.LE00.IN"].to_numpy(dtype=float)

    res = stats.linregress(x, y)
    r_squared = float(res.rvalue ** 2)

    usa_row = valid[valid["iso3"] == "USA"].iloc[0]
    usa_chex_pc = float(usa_row["SH.XPD.CHEX.PC.CD"])
    usa_chex_gdp = float(usa_row["SH.XPD.CHEX.GD.ZS"])
    usa_le_obs = float(usa_row["SP.DYN.LE00.IN"])
    usa_le_pred = float(res.intercept + res.slope * math.log(usa_chex_pc))
    usa_residual = float(usa_le_obs - usa_le_pred)

    peers = {}
    for iso in ["JPN", "CHE", "GBR", "DEU", "SGP"]:
        sub = valid[valid["iso3"] == iso]
        if not sub.empty:
            r = sub.iloc[0]
            pred_i = float(res.intercept + res.slope * math.log(float(r["SH.XPD.CHEX.PC.CD"])))
            peers[iso] = {
                "country_name": str(r["country_name"]),
                "chex_pc": float(r["SH.XPD.CHEX.PC.CD"]),
                "le_obs": float(r["SP.DYN.LE00.IN"]),
                "le_pred": pred_i,
                "residual": float(r["SP.DYN.LE00.IN"]) - pred_i,
            }

    return {
        "n_non_us_observed": int(len(non_us)),
        "intercept_beta0": float(res.intercept),
        "slope_beta1_ln": float(res.slope),
        "r_value": float(res.rvalue),
        "r_squared": r_squared,
        "p_value": float(res.pvalue),
        "std_err_slope": float(res.stderr),
        "usa": {
            "chex_pc_usd": usa_chex_pc,
            "chex_pct_gdp": usa_chex_gdp,
            "le_observed": usa_le_obs,
            "le_expected_preston": usa_le_pred,
            "residual_years": usa_residual,
        },
        "benchmark_peers": peers,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    retrieval_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    print("[1/6] Fetching World Bank sovereign economy catalog...")
    countries_df = fetch_sovereign_countries()
    valid_iso3 = set(countries_df["iso3"])
    print(f"    Cataloged {len(countries_df)} sovereign economies (excluding aggregate regions).")

    print("[2/6] Fetching 8 WDI/WHO indicators for 2015-2023...")
    panel_frames = []
    provenance_urls: Dict[str, str] = {}
    for ind in INDICATORS:
        code = str(ind["code"])
        df_ind, src_url = fetch_indicator_panel(ind, valid_iso3)
        panel_frames.append(df_ind)
        provenance_urls[code] = src_url

    panel = panel_frames[0]
    for df_next in panel_frames[1:]:
        panel = panel.merge(df_next, on=["iso3", "year"], how="outer")

    panel = countries_df.merge(panel, on="iso3", how="inner").sort_values(["iso3", "year"])
    raw_panel_path = OUT_DIR / f"wdi_who_raw_panel_{YEAR_START}_{YEAR_END}.csv"
    panel.to_csv(raw_panel_path, index=False)

    print("[3/6] Computing country-level multi-year means and filtering M sovereign nations...")
    col_codes = [str(ind["code"]) for ind in INDICATORS]
    country_means = (
        panel.groupby(["iso3", "iso2", "country_name", "region", "income_level"], as_index=False)[col_codes]
        .mean()
        .sort_values("iso3")
        .reset_index(drop=True)
    )

    obs_per_country = country_means[col_codes].notna().sum(axis=1)
    excluded_df = country_means[obs_per_country < MIN_OBSERVED_INDICATORS].copy()
    df_raw = country_means[obs_per_country >= MIN_OBSERVED_INDICATORS].reset_index(drop=True)

    M = len(df_raw)
    total_cells = M * len(col_codes)
    total_missing = int(df_raw[col_codes].isna().sum().sum())
    overall_missing_pct = float(total_missing / total_cells * 100.0)

    raw_means_path = OUT_DIR / "wdi_who_country_means_raw.csv"
    df_raw.to_csv(raw_means_path, index=False)

    print(f"    Retained M = {M} sovereign nations ({total_cells} cells, {total_missing} missing = {overall_missing_pct:.2f}%).")

    print("[4/6] Running leak-free 5-fold KNN CV and full-matrix imputation...")
    raw_mat = df_raw[col_codes].to_numpy(dtype=float)
    cv_audit = run_leak_free_knn_cv(raw_mat, col_codes)

    scaler_full = MinMaxScaler()
    scaled_full = scaler_full.fit_transform(raw_mat)
    imputer_full = KNNImputer(n_neighbors=KNN_NEIGHBORS, weights="distance", metric="nan_euclidean")
    imputed_scaled_full = imputer_full.fit_transform(scaled_full)
    imputed_orig_full = scaler_full.inverse_transform(imputed_scaled_full)

    df_imputed = df_raw.copy()
    df_imputed[col_codes] = imputed_orig_full
    imp_means_path = OUT_DIR / "wdi_who_country_means_imputed.csv"
    df_imputed.to_csv(imp_means_path, index=False)

    schema_stats = []
    for idx, ind in enumerate(INDICATORS):
        code = str(ind["code"])
        s_obs = df_raw[code].dropna()
        n_miss = int(df_raw[code].isna().sum())
        miss_pct = float(n_miss / M * 100.0)
        q25, q50, q75 = np.percentile(s_obs, [25, 50, 75])
        w1_norm = float(
            stats.wasserstein_distance(
                scaled_full[~np.isnan(raw_mat[:, idx]), idx],
                imputed_scaled_full[:, idx],
            )
        )
        schema_stats.append(
            {
                "code": code,
                "name": str(ind["name"]),
                "unit": str(ind["unit"]),
                "endpoint_url": provenance_urls[code],
                "n_observed": int(len(s_obs)),
                "missing_count": n_miss,
                "missing_pct": miss_pct,
                "mean": float(s_obs.mean()),
                "sd": float(s_obs.std(ddof=1)),
                "median": float(q50),
                "q25": float(q25),
                "q75": float(q75),
                "min": float(s_obs.min()),
                "max": float(s_obs.max()),
                "cv_nrmse": cv_audit["per_indicator"][code]["nrmse"],
                "cv_rmse_orig": cv_audit["per_indicator"][code]["rmse_original_units"],
                "wasserstein1_norm": w1_norm,
            }
        )

    print("[5/6] Fitting unimputed non-U.S. Preston Curve OLS regression...")
    preston_audit = fit_preston_curve_ols(df_raw)

    rng = np.random.default_rng(RANDOM_SEED)
    df_splits = df_imputed[["iso3", "country_name", "region", "income_level"]].copy()
    df_splits["rand_key"] = rng.random(M)
    df_splits = df_splits.sort_values(["region", "rand_key"]).reset_index(drop=True)
    warmup_indices = set(np.linspace(0, M - 1, WARMUP_COUNTRIES, dtype=int).tolist())
    df_splits["split"] = [
        "warmup_15" if i in warmup_indices else "evaluation" for i in range(M)
    ]
    df_splits = df_splits.drop(columns=["rand_key"]).sort_values("iso3").reset_index(drop=True)
    splits_path = OUT_DIR / "wdi_who_stimulus_splits.csv"
    df_splits.to_csv(splits_path, index=False)

    print("[6/6] Writing verification JSON and LaTeX Table I macros...")
    audit_payload = {
        "metadata": {
            "retrieval_utc": retrieval_utc,
            "source": "World Bank World Development Indicators (WDI) API v2 & WHO Global Health Observatory (GHO) OData API",
            "year_window": f"{YEAR_START}-{YEAR_END}",
            "random_seed": RANDOM_SEED,
            "min_observed_indicators_rule": MIN_OBSERVED_INDICATORS,
            "excluded_countries_iso3": excluded_df["iso3"].tolist(),
            "indicator_endpoints": provenance_urls,
        },
        "matrix_audit": {
            "M_countries": M,
            "K_indicators": len(col_codes),
            "total_cells": total_cells,
            "total_missing_cells": total_missing,
            "overall_missing_pct": overall_missing_pct,
            "warmup_count": int((df_splits["split"] == "warmup_15").sum()),
            "evaluation_count": int((df_splits["split"] == "evaluation").sum()),
        },
        "knn_imputation_audit": {
            "k": KNN_NEIGHBORS,
            "weights": "distance",
            "metric": "nan_euclidean",
            "cv_folds": CV_FOLDS,
            "overall_holdout_nrmse": cv_audit["overall_nrmse"],
        },
        "schema_table": schema_stats,
        "preston_curve_ols": preston_audit,
        "artifact_sha256": {
            raw_panel_path.name: sha256_file(raw_panel_path),
            raw_means_path.name: sha256_file(raw_means_path),
            imp_means_path.name: sha256_file(imp_means_path),
            splits_path.name: sha256_file(splits_path),
        },
    }

    audit_json_path = OUT_DIR / "wdi_who_verification_audit.json"
    audit_json_path.write_text(json.dumps(audit_payload, indent=2), encoding="utf-8")

    usa = preston_audit["usa"]
    tex_lines = [
        "% Auto-generated by reproduce_wdi_who_dataset-v5.py -- DO NOT EDIT MANUALLY",
        f"% Retrieved UTC: {retrieval_utc} | SHA256(imputed): {audit_payload['artifact_sha256'][imp_means_path.name]}",
        rf"\newcommand{{\WdiM}}{{{M}}}",
        rf"\newcommand{{\WdiEvalM}}{{{M - WARMUP_COUNTRIES}}}",
        rf"\newcommand{{\WdiTotalCells}}{{{total_cells:,}}}",
        rf"\newcommand{{\WdiMissingCells}}{{{total_missing}}}",
        rf"\newcommand{{\WdiMissingPct}}{{{overall_missing_pct:.2f}\%}}",
        rf"\newcommand{{\WdiKnnNrmse}}{{{cv_audit['overall_nrmse']:.4f}}}",
        rf"\newcommand{{\PrestonRsq}}{{{preston_audit['r_squared']:.3f}}}",
        rf"\newcommand{{\PrestonBetaZero}}{{{preston_audit['intercept_beta0']:.2f}}}",
        rf"\newcommand{{\PrestonBetaOne}}{{{preston_audit['slope_beta1_ln']:.2f}}}",
        rf"\newcommand{{\UsaChexPc}}{{\${usa['chex_pc_usd']:,.0f}}}",
        rf"\newcommand{{\UsaChexGdp}}{{{usa['chex_pct_gdp']:.1f}\%}}",
        rf"\newcommand{{\UsaLeObs}}{{{usa['le_observed']:.1f}}}",
        rf"\newcommand{{\UsaLePred}}{{{usa['le_expected_preston']:.2f}}}",
        rf"\newcommand{{\UsaLeResid}}{{{usa['residual_years']:+.2f}}}",
        "",
        r"\begin{table*}[!t]",
        rf"\caption{{World Bank WDI and WHO GHO cross-national macro-health dataset schema ($M = {M}$ nations, {YEAR_START}--{YEAR_END} averages) and preprocessing audit.}}",
        r"\label{tab:schema}",
        r"\centering",
        r"\footnotesize",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}llllllrr@{}}",
        r"\toprule",
        r"\textbf{Indicator Code} & \textbf{Indicator Name (Source)} & \textbf{Unit} & \textbf{Mean (SD)} & \textbf{Median [IQR]} & \textbf{Missing \%} & \textbf{KNN NRMSE} \\",
        r"\midrule",
    ]
    for row, ind in zip(schema_stats, INDICATORS):
        fmt = str(ind["fmt"])
        mean_sd = f"{fmt.format(row['mean'])} ({fmt.format(row['sd'])})"
        med_iqr = f"{fmt.format(row['median'])} [{fmt.format(row['q25'])}--{fmt.format(row['q75'])}]"
        tex_lines.append(
            rf"\texttt{{{row['code']}}} & {row['name']} & {row['unit']} & "
            rf"{mean_sd} & {med_iqr} & {row['missing_pct']:.1f}\% & {row['cv_nrmse']:.3f} \\"
        )
    tex_lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}}",
            r"\end{table*}",
        ]
    )
    tex_path = OUT_DIR / "section4_verified_macros.tex"
    tex_path.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")

    print("\n=== VERIFICATION SUMMARY ===")
    print(f"M = {M} sovereign nations | Total cells = {total_cells} | Missing = {total_missing} ({overall_missing_pct:.2f}%)")
    print(f"5-fold leak-free KNN (k=5) overall NRMSE = {cv_audit['overall_nrmse']:.4f}")
    print(
        f"Preston Curve OLS (n={preston_audit['n_non_us_observed']} non-U.S.): "
        f"LE = {preston_audit['intercept_beta0']:.2f} + {preston_audit['slope_beta1_ln']:.2f} ln(CHEX_PC), "
        f"R^2 = {preston_audit['r_squared']:.3f}, p = {preston_audit['p_value']:.2e}"
    )
    print(
        f"USA Observed LE = {usa['le_observed']:.2f} yrs vs Expected = {usa['le_expected_preston']:.2f} yrs "
        f"(Residual = {usa['residual_years']:+.2f} yrs at ${usa['chex_pc_usd']:,.0f} US$ / {usa['chex_pct_gdp']:.1f}% GDP)"
    )
    print(f"All verified artifacts written to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
