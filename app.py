"""
Sistema de alarmas FX — Streamlit app.

Cada visita: baja datos en vivo de Yahoo Finance (FX, DXY, VIX, ^TNX, commodities)
+ FRED (DFF, T10Y2Y), computa features, predice con LightGBM persistido,
detecta episodios sobre la marcha, y renderiza el dashboard.
"""
from __future__ import annotations
import json
from io import StringIO
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st
import lightgbm as lgb

st.set_page_config(page_title="Sistema de alarmas FX", page_icon="📊", layout="wide")

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
SNAPSHOTS = ROOT / "snapshots"

FX_COLS = ["USDARS", "USDBRL", "USDCLP", "USDCOP", "USDINR"]

YAHOO_TICKERS = {
    "USDARS": "USDARS=X", "USDBRL": "USDBRL=X", "USDCLP": "USDCLP=X",
    "USDCOP": "USDCOP=X", "USDINR": "USDINR=X",
    "DXY": "DX-Y.NYB", "VIX": "^VIX", "DGS10": "^TNX",
    "BRENT": "BZ=F", "WTI": "CL=F", "COPPER": "HG=F",
    "GOLD": "GC=F", "SOYBEANS": "ZS=F", "IRON": "TIO=F",
}

PRINCIPAL_COMM = {"USDARS": "SOYBEANS", "USDBRL": "IRON", "USDCLP": "COPPER",
                  "USDCOP": "BRENT", "USDINR": "BRENT"}
GLOBAL_DRIVERS = ["DXY", "VIX", "DGS10", "DFF", "T10Y2Y"]
DRIVERS_WITH_LEVEL = ["VIX", "DGS10", "DFF", "T10Y2Y"]

THRESHOLDS_1D = {"USDARS": 0.05, "USDBRL": 0.02, "USDCLP": 0.015, "USDCOP": 0.015, "USDINR": 0.008}
THRESHOLDS_5D = {"USDARS": 0.10, "USDBRL": 0.05, "USDCLP": 0.04, "USDCOP": 0.04, "USDINR": 0.02}

EXT_THR = {("USDARS","5d"):0.05,("USDARS","20d"):0.10,
           ("USDBRL","5d"):0.025,("USDBRL","20d"):0.05,
           ("USDCLP","5d"):0.020,("USDCLP","20d"):0.04,
           ("USDCOP","5d"):0.022,("USDCOP","20d"):0.045,
           ("USDINR","5d"):0.012,("USDINR","20d"):0.025}

DRIVER_AFFECTS = {
    "DXY": "Afecta a TODOS los flotantes. Si sube, las EM se debilitan.",
    "VIX": "Miedo global. BRL e INR son los más sensibles.",
    "DGS10": "Yield US 10Y. Sube → fuga de EM.",
    "BRENT": "Petróleo. Afecta a USDCOP y USDINR.",
    "COPPER": "Cobre. Driver casi exclusivo del USDCLP.",
    "GOLD": "Oro. Refugio de valor en risk-off.",
}

DRIVER_HUMAN_MAP = {
    "global_riskoff_strong": "Pánico global (VIX > 35)",
    "global_riskoff": "Aversión al riesgo (VIX 25-35)",
    "USD_strength": "Dólar global fortalecido",
    "USD_weakness": "Dólar global debilitado",
    "copper_drop": "Caída del cobre", "copper_rally": "Suba del cobre",
    "iron_drop": "Caída del hierro", "iron_rally": "Suba del hierro",
    "brent_drop": "Caída del petróleo", "brent_rally": "Suba del petróleo",
    "soybeans_drop": "Caída de la soja", "soybeans_rally": "Suba de la soja",
    "local_idiosyncratic": "Evento local (sin driver global identificado)",
}

# Detección de episodios — config
WINDOW_BDAYS = 30
SEPARATION_BDAYS = 45
RECOVERY_HORIZON = 250
RECOVERY_THRESHOLD = 0.5
MIN_MAG_PCT = {"USDARS": 8.0, "USDBRL": 5.0, "USDCLP": 3.5, "USDCOP": 4.0, "USDINR": 2.0}

KNOWN_EVENTS = {
    "USDARS": [("2014-01-23", "Devaluación Kicillof"),
               ("2018-04-25", "Corrida 2018 — inicio"),
               ("2018-08-30", "Corrida 2018 — profundización"),
               ("2019-08-12", "PASO 2019"),
               ("2020-09-15", "Cepo 2.0 / restricción ahorro USD"),
               ("2022-07-29", "Crisis Batakis"),
               ("2023-08-14", "Devaluación Massa post-PASO"),
               ("2023-12-12", "Salida cepo Milei"),
               ("2025-04-14", "Liberación cepo personas (acuerdo FMI)")],
    "USDBRL": [("2008-09-15", "Lehman"),
               ("2008-10-08", "Lehman — pico volatilidad"),
               ("2011-09-22", "Crisis deuda europea"),
               ("2015-09-24", "Downgrade S&P Brasil a junk"),
               ("2016-04-17", "Impeachment Dilma — votación cámara"),
               ("2017-05-18", "Joesley day"),
               ("2018-09-01", "Pre-elección Bolsonaro"),
               ("2020-03-23", "COVID — peak BRL"),
               ("2020-05-13", "COVID — pico USDBRL 5.89"),
               ("2022-10-30", "Elección Lula 2T"),
               ("2024-12-18", "Crisis fiscal Lula 2")],
    "USDCLP": [("2008-10-15", "Lehman"),
               ("2011-08-05", "Crisis deuda US"),
               ("2015-08-24", "Devaluación yuan"),
               ("2019-10-18", "Estallido social"),
               ("2019-11-15", "Estallido — pico"),
               ("2020-03-16", "COVID"),
               ("2021-07-04", "Convención constitucional"),
               ("2022-07-14", "Pico USDCLP 1050 + intervención BCCh"),
               ("2022-09-04", "Rechazo plebiscito")],
    "USDCOP": [("2008-09-29", "Lehman"),
               ("2014-12-16", "Caída petróleo"),
               ("2015-08-24", "Devaluación yuan + petróleo"),
               ("2016-01-20", "Mínimo Brent — pico USDCOP"),
               ("2018-06-17", "Elección Duque"),
               ("2020-03-19", "COVID — pico"),
               ("2022-06-19", "Elección Petro"),
               ("2022-11-08", "USDCOP pico 5106")],
    "USDINR": [("2008-10-27", "Lehman"),
               ("2011-12-15", "Crisis EU + flujos out India"),
               ("2013-08-28", "Taper tantrum — pico INR"),
               ("2018-10-09", "Crisis NBFC + petróleo"),
               ("2020-04-22", "COVID"),
               ("2022-07-19", "Fed hiking + petróleo"),
               ("2024-12-26", "Continuación debilitamiento")],
}

# ---------------- Fetch Yahoo ----------------

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_yahoo_one(ticker: str, days: int = 600) -> pd.Series:
    now = int(datetime.now(timezone.utc).timestamp())
    start = now - days * 24 * 3600
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {"period1": start, "period2": now, "interval": "1d"}
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=15)
        r.raise_for_status()
        j = r.json()
        res = j.get("chart", {}).get("result", [])
        if not res: return pd.Series(dtype=float)
        ts = res[0].get("timestamp", []) or []
        closes = res[0].get("indicators", {}).get("quote", [{}])[0].get("close", []) or []
        if not ts: return pd.Series(dtype=float)
        idx = pd.to_datetime([pd.Timestamp(t, unit="s", tz="UTC") for t in ts]).tz_convert(None).normalize()
        s = pd.Series(closes, index=idx)
        s = s[~s.index.duplicated(keep="last")]
        return s.dropna()
    except Exception:
        return pd.Series(dtype=float)


@st.cache_data(ttl=3600, show_spinner="Bajando datos en vivo de Yahoo Finance...")
def fetch_all_data():
    series = {}
    for name, ticker in YAHOO_TICKERS.items():
        s = fetch_yahoo_one(ticker)
        if not s.empty: series[name] = s
    if not series: return pd.DataFrame(), pd.DataFrame()
    all_idx = sorted(set().union(*[s.index for s in series.values()]))
    bdays = pd.bdate_range(min(all_idx), max(all_idx))
    aligned = {n: s.reindex(bdays, method="ffill").ffill() for n, s in series.items()}
    fx_df = pd.DataFrame({k: aligned[k] for k in FX_COLS if k in aligned})
    drv_cols = ["DXY", "VIX", "DGS10", "BRENT", "WTI", "COPPER", "GOLD", "SOYBEANS", "IRON"]
    drv_df = pd.DataFrame({k: aligned[k] for k in drv_cols if k in aligned})
    if "DGS10" in drv_df.columns and drv_df["DGS10"].median() > 30:
        drv_df["DGS10"] = drv_df["DGS10"] / 10
    return fx_df, drv_df


# ---------------- Fetch FRED ----------------

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_fred_one(series_id: str) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        if df.shape[1] < 2: return pd.Series(dtype=float)
        df.columns = ["date", "value"]
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["date"]).set_index("date")
        return df["value"].dropna().sort_index()
    except Exception:
        return pd.Series(dtype=float)


@st.cache_data(ttl=86400, show_spinner=False)
def load_snapshot_fred() -> pd.DataFrame:
    path = SNAPSHOTS / "fred_recent.json"
    if not path.exists(): return pd.DataFrame()
    j = json.loads(path.read_text())
    out = {}
    for col, days in j.items():
        s = pd.Series({pd.Timestamp(d): v for d, v in days.items()})
        out[col] = s.sort_index()
    return pd.DataFrame(out)


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_fred_drivers() -> tuple:
    """Devuelve (fred_df, source_label). Intenta FRED en vivo, fallback a snapshot."""
    out = {}
    for sid in ["DFF", "T10Y2Y", "DGS10"]:
        s = fetch_fred_one(sid)
        if not s.empty: out[sid] = s
    if out:
        return pd.DataFrame(out), "live"
    snap = load_snapshot_fred()
    return snap, "snapshot" if not snap.empty else "none"


# ---------------- Features ----------------

def safe_log(s):
    return np.log(s.where(s > 0))


def build_self(fx):
    p = fx; lp = safe_log(p); ret1 = lp.diff()
    out = pd.DataFrame(index=p.index)
    out["ret_1d"] = ret1
    out["ret_5d"] = lp.diff(5)
    out["ret_20d"] = lp.diff(20)
    out["vol_20d"] = ret1.rolling(20).std()
    out["vol_60d"] = ret1.rolling(60).std()
    sma20 = p.rolling(20).mean(); sma60 = p.rolling(60).mean(); std60 = p.rolling(60).std()
    out["dist_sma20"] = (p - sma20) / sma20
    out["zscore_60d"] = (p - sma60) / std60
    return out


def build_driver(s, name, with_level):
    out = pd.DataFrame(index=s.index)
    if with_level: out[f"{name}_lvl"] = s
    lp = safe_log(s)
    out[f"{name}_ret_5d"] = lp.diff(5)
    out[f"{name}_ret_20d"] = lp.diff(20)
    ret1 = s.diff() if with_level else lp.diff()
    out[f"{name}_vol_20d"] = ret1.rolling(20).std()
    return out


def build_features_for(ccy, fx_df, drv_df, fred):
    feats = [build_self(fx_df[ccy])]
    fred_a = fred.reindex(fx_df.index, method="ffill") if not fred.empty else pd.DataFrame()
    for d in GLOBAL_DRIVERS:
        if d in drv_df.columns:
            feats.append(build_driver(drv_df[d], d, d in DRIVERS_WITH_LEVEL))
        elif d in fred_a.columns:
            feats.append(build_driver(fred_a[d], d, d in DRIVERS_WITH_LEVEL))
    comm = PRINCIPAL_COMM[ccy]
    if comm in drv_df.columns:
        feats.append(build_driver(drv_df[comm], comm, with_level=False))
    return pd.concat(feats, axis=1)


# ---------------- Predict ----------------

@st.cache_resource
def load_models():
    out = {}
    feat_cols = json.loads((MODELS / "feature_columns.json").read_text())
    for ccy in FX_COLS:
        for h in ["5d", "20d"]:
            for kind in ["clf", "reg"]:
                name = f"gbm_{ccy}_{h}.txt" if kind == "clf" else f"gbm_reg_{ccy}_{h}.txt"
                path = MODELS / name
                if path.exists():
                    out[(ccy, h, kind)] = lgb.Booster(model_file=str(path))
    return out, feat_cols


def predict_today(fx_df, drv_df, fred):
    models, feat_cols_by_ccy = load_models()
    out = {}
    for ccy in FX_COLS:
        if ccy not in fx_df.columns or fx_df[ccy].dropna().empty: continue
        feat = build_features_for(ccy, fx_df, drv_df, fred)
        expected = feat_cols_by_ccy.get(ccy, list(feat.columns))
        for col in expected:
            if col not in feat.columns: feat[col] = np.nan
        feat = feat[expected]
        last_idx = feat.dropna().index.max() if not feat.dropna().empty else feat.index.max()
        x_now = feat.loc[[last_idx]].fillna(0)
        preds = {}
        for h in ["5d", "20d"]:
            clf = models.get((ccy, h, "clf")); reg = models.get((ccy, h, "reg"))
            preds[h] = {
                "prob": float(clf.predict(x_now)[0]) if clf else None,
                "ret": float(reg.predict(x_now)[0]) if reg else None,
            }
        out[ccy] = preds
    return out


# ---------------- Episode detection (live) ----------------

def detect_episodes(fx_series, ccy):
    s = fx_series.dropna()
    logs = np.log(s)
    roll_min = logs.rolling(WINDOW_BDAYS).min()
    move_up = logs - roll_min
    roll_max = logs.rolling(WINDOW_BDAYS).max()
    move_dn = roll_max - logs
    thr = np.log(1 + MIN_MAG_PCT[ccy] / 100)

    candidates = []
    for direction, series, sign in [("up", move_up, 1), ("dn", move_dn, -1)]:
        peaks = series[series > thr]
        i = 0; idx = peaks.index
        while i < len(idx):
            d = idx[i]; j = i
            window_end = d + pd.Timedelta(days=int(SEPARATION_BDAYS * 1.5))
            while j + 1 < len(idx) and idx[j + 1] <= window_end:
                j += 1
            sub = series.loc[d:idx[j]]
            peak_d = sub.idxmax(); mag = sub.max()
            start_window = logs.loc[:peak_d].tail(WINDOW_BDAYS + 1)
            if direction == "up":
                start_d = start_window.idxmin()
            else:
                start_d = start_window.idxmax()
            days_to_peak = max(1, len(s.loc[start_d:peak_d]) - 1)
            mag_pct = (np.exp(mag) - 1) * 100 * sign
            candidates.append({
                "peak_date": peak_d, "start_date": start_d, "direction": direction,
                "mag_pct": round(float(mag_pct), 2),
                "days_to_peak": int(days_to_peak),
            })
            next_d = peak_d + pd.Timedelta(days=int(SEPARATION_BDAYS * 1.5))
            while i < len(idx) and idx[i] <= next_d:
                i += 1
    if not candidates: return pd.DataFrame()
    df = pd.DataFrame(candidates).sort_values("peak_date").drop_duplicates("peak_date")
    df = df.sort_values("mag_pct", key=lambda x: x.abs(), ascending=False)
    keep = []; used = []
    for _, row in df.iterrows():
        d = row["peak_date"]
        if all(abs((d - u).days) > SEPARATION_BDAYS * 7 / 5 for u in used):
            keep.append(row); used.append(d)
    return pd.DataFrame(keep).sort_values("peak_date").reset_index(drop=True) if keep else pd.DataFrame()


def add_recovery(df, fx_series):
    logs = np.log(fx_series.dropna())
    rec_days = []; permanents = []
    for _, row in df.iterrows():
        peak_d = row["peak_date"]; start_d = row["start_date"]
        if peak_d not in logs.index or start_d not in logs.index:
            rec_days.append(np.nan); permanents.append(True); continue
        peak_v = logs.at[peak_d]; start_v = logs.at[start_d]
        target = peak_v - (peak_v - start_v) * RECOVERY_THRESHOLD
        future = logs.loc[peak_d:].iloc[1:RECOVERY_HORIZON + 1]
        if row["direction"] == "up":
            recovered = future[future <= target]
        else:
            recovered = future[future >= target]
        if recovered.empty:
            rec_days.append(np.nan); permanents.append(True)
        else:
            rec_d = recovered.index[0]
            rec_days.append(len(logs.loc[peak_d:rec_d]) - 1); permanents.append(False)
    df = df.copy()
    df["days_to_recover"] = rec_days
    df["permanent"] = permanents
    return df


def attribute_driver(df, drv, ccy):
    comm = PRINCIPAL_COMM[ccy]
    classes = []
    for _, row in df.iterrows():
        peak = row["peak_date"]; start = row["start_date"]
        wnd = drv.loc[start:peak + pd.Timedelta(days=10)] if not drv.empty else pd.DataFrame()
        vix_max = wnd["VIX"].max() if "VIX" in wnd.columns else np.nan
        dxy_ret = (wnd["DXY"].iloc[-1] / wnd["DXY"].iloc[0] - 1) * 100 if "DXY" in wnd.columns and len(wnd) > 1 else np.nan
        comm_ret = ((wnd[comm].iloc[-1] / wnd[comm].iloc[0] - 1) * 100
                    if comm in wnd.columns and len(wnd) > 1 and pd.notna(wnd[comm].iloc[0]) else np.nan)
        flags = []
        if pd.notna(vix_max):
            if vix_max > 35: flags.append("global_riskoff_strong")
            elif vix_max > 25: flags.append("global_riskoff")
        if pd.notna(dxy_ret):
            if dxy_ret > 2.5: flags.append("USD_strength")
            elif dxy_ret < -2.5: flags.append("USD_weakness")
        if pd.notna(comm_ret):
            if row["direction"] == "up" and comm_ret < -5: flags.append(f"{comm.lower()}_drop")
            elif row["direction"] == "dn" and comm_ret > 5: flags.append(f"{comm.lower()}_rally")
        if not flags: flags = ["local_idiosyncratic"]
        classes.append("+".join(flags))
    df = df.copy(); df["driver_class"] = classes
    return df


def match_known_events(df, ccy):
    events = KNOWN_EVENTS.get(ccy, [])
    df = df.copy(); matched = []
    for _, row in df.iterrows():
        peak = row["peak_date"]; best = ""; best_dist = 999
        for d_str, name in events:
            ed = pd.Timestamp(d_str)
            dist = abs((peak - ed).days)
            if dist <= 14 and dist < best_dist: best = name; best_dist = dist
        matched.append(best)
    df["known_event"] = matched
    return df


def detect_episodes_live(fx_df, drv_df, fred_df, days_back=180):
    out = []
    cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=days_back)
    if fred_df.empty:
        drv_full = drv_df
    else:
        fred_a = fred_df.reindex(drv_df.index, method="ffill") if not drv_df.empty else fred_df
        drv_full = pd.concat([drv_df, fred_a], axis=1)
    for ccy in FX_COLS:
        if ccy not in fx_df.columns: continue
        ep = detect_episodes(fx_df[ccy], ccy)
        if ep.empty: continue
        ep = add_recovery(ep, fx_df[ccy])
        ep = attribute_driver(ep, drv_full, ccy)
        ep = match_known_events(ep, ccy)
        ep = ep[ep["peak_date"] >= cutoff]
        for _, row in ep.iterrows():
            pd_ts = row["peak_date"]
            out.append({
                "ccy": ccy,
                "peak_date": pd_ts.strftime("%Y-%m-%d"),
                "mag_pct": float(row["mag_pct"]),
                "days_to_peak": int(row["days_to_peak"]),
                "permanent": bool(row["permanent"]),
                "driver_class": str(row["driver_class"]),
                "known_event": str(row.get("known_event", "")),
                "direction": "up" if row["direction"] == "up" else "dn",
            })
    out.sort(key=lambda x: x["peak_date"], reverse=True)
    return out[:20]


# ---------------- Card data builders ----------------

def card_data(ccy, fx_series, pred):
    s = fx_series.dropna()
    if s.empty: return None
    px = float(s.iloc[-1])
    ret_1d = float(np.log(s.iloc[-1]/s.iloc[-2])*100) if len(s) >= 2 else None
    ret_5d = float(np.log(s.iloc[-1]/s.iloc[-6])*100) if len(s) >= 6 else None
    ret_20d = float(np.log(s.iloc[-1]/s.iloc[-21])*100) if len(s) >= 21 else None
    sp = s.iloc[-60:]
    vals = [round(float(v), 4) for v in sp.tolist()]
    dates = [d.strftime("%Y-%m-%d") for d in sp.index]
    p5 = pred.get("5d", {}) if pred else {}
    p20 = pred.get("20d", {}) if pred else {}
    base5 = EXT_THR[(ccy, "5d")] * 100
    base20 = EXT_THR[(ccy, "20d")] * 100
    alarm_1d = abs(ret_1d) > THRESHOLDS_1D[ccy] * 100 if ret_1d else False
    alarm_5d = abs(ret_5d) > THRESHOLDS_5D[ccy] * 100 if ret_5d else False
    return {
        "ccy": ccy, "px": px,
        "ret_1d": ret_1d, "ret_5d": ret_5d, "ret_20d": ret_20d,
        "spark": vals, "spark_dates": dates,
        "spark_min": min(vals), "spark_max": max(vals),
        "spark_min_date": dates[vals.index(min(vals))],
        "spark_max_date": dates[vals.index(max(vals))],
        "spark_first_date": dates[0], "spark_last_date": dates[-1],
        "alarma_1d": alarm_1d, "alarma_5d": alarm_5d,
        "pred_5d": {"prob": ((p5.get("prob") or 0)*100), "ret": ((p5.get("ret") or 0)*100), "umbral": base5},
        "pred_20d": {"prob": ((p20.get("prob") or 0)*100), "ret": ((p20.get("ret") or 0)*100), "umbral": base20},
    }


def driver_card_data(name, series):
    s = series.dropna()
    if s.empty: return None
    sp = s.iloc[-60:]
    vals = [round(float(v), 4) for v in sp.tolist()]
    dates = [d.strftime("%Y-%m-%d") for d in sp.index]
    n = len(s); v_now = float(s.iloc[-1])
    chg_1d = float(s.iloc[-1] - s.iloc[-2]) if n >= 2 else None
    chg_5d_pct = ((v_now / float(s.iloc[-6]) - 1) * 100) if n >= 6 else None
    chg_20d_pct = ((v_now / float(s.iloc[-21]) - 1) * 100) if n >= 21 else None
    return {
        "name": name, "lvl": v_now,
        "chg_1d": chg_1d, "chg_5d_pct": chg_5d_pct, "chg_20d_pct": chg_20d_pct,
        "spark": vals, "spark_dates": dates,
        "spark_min": min(vals), "spark_max": max(vals),
    }


# ---------------- Render HTML ----------------

HTML_TEMPLATE = r"""
<style>
.fx-wrapper {font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif;}
.fx-wrapper .help-grid{display:grid;grid-template-columns:repeat(auto-fit, minmax(240px,1fr));gap:12px;margin-top:8px;}
.fx-wrapper .help-card{background:#eff6ff;border:1px solid #bfdbfe;border-radius:6px;padding:10px 12px;font-size:13px;color:#1e3a8a;line-height:1.55;}
.fx-wrapper .help-card b{color:#1e40af;}
.fx-wrapper .card-grid{display:grid;grid-template-columns:repeat(auto-fit, minmax(310px,1fr));gap:14px;}
.fx-wrapper .card{background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:16px;box-shadow:0 1px 2px rgba(0,0,0,0.04);}
.fx-wrapper .ccy-name{font-size:16px;font-weight:600;margin-bottom:6px;display:flex;justify-content:space-between;align-items:center;color:#0f172a;}
.fx-wrapper .px{font-size:22px;font-weight:600;color:#0f172a;}
.fx-wrapper .row{display:flex;justify-content:space-between;padding:3px 0;font-size:13px;color:#64748b;}
.fx-wrapper .row span:nth-child(2){color:#0f172a;font-weight:500;font-variant-numeric:tabular-nums;}
.fx-wrapper .ret-up{color:#dc2626;}
.fx-wrapper .ret-dn{color:#16a34a;}
.fx-wrapper .alarm-on{background:#dc2626;color:#fff;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600;}
.fx-wrapper .alarm-off{color:#94a3b8;font-size:11px;}
.fx-wrapper .prob-bar{height:6px;background:#e2e8f0;border-radius:3px;overflow:hidden;margin:4px 0 8px;}
.fx-wrapper .prob-bar-fill{height:100%;background:#3b82f6;}
.fx-wrapper .pred-block{background:#f8fafc;border:1px solid #e2e8f0;border-radius:5px;padding:8px 10px;margin:8px 0;font-size:12px;}
.fx-wrapper .pred-block .ph{font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:0.4px;margin-bottom:4px;}
.fx-wrapper .pred-block .pmain{font-size:13px;font-weight:500;margin-bottom:3px;color:#0f172a;}
.fx-wrapper .pred-block .psub{font-size:11px;color:#64748b;}
.fx-wrapper .dir-up{color:#dc2626;font-weight:600;}
.fx-wrapper .dir-dn{color:#16a34a;font-weight:600;}
.fx-wrapper .dir-flat{color:#64748b;}
.fx-wrapper .driver-grid{display:grid;grid-template-columns:repeat(auto-fit, minmax(260px,1fr));gap:12px;}
.fx-wrapper .driver-card{background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:14px;}
.fx-wrapper .driver-name{font-size:12px;color:#64748b;text-transform:uppercase;letter-spacing:0.5px;font-weight:600;}
.fx-wrapper .driver-val{font-size:24px;font-weight:600;margin:6px 0 4px 0;color:#0f172a;}
.fx-wrapper .driver-affects{font-size:11px;color:#64748b;margin-top:10px;line-height:1.4;border-top:1px solid #f1f5f9;padding-top:8px;}
.fx-wrapper .driver-chgs{display:flex;gap:14px;font-size:11px;margin-top:6px;color:#64748b;}
.fx-wrapper .driver-chgs span b{color:#0f172a;font-weight:500;font-variant-numeric:tabular-nums;}
.fx-wrapper .ep-table{width:100%;border-collapse:collapse;font-size:13px;background:#fff;border:1px solid #e2e8f0;border-radius:6px;overflow:hidden;margin-top:6px;}
.fx-wrapper .ep-table th{text-align:left;color:#64748b;padding:10px 8px;border-bottom:1px solid #e2e8f0;font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:0.4px;background:#f8fafc;}
.fx-wrapper .ep-table td{padding:8px;border-bottom:1px solid #f1f5f9;font-variant-numeric:tabular-nums;}
.fx-wrapper .ep-table td.num{text-align:right;}
.fx-wrapper .spark-wrap{position:relative;margin:8px 0 6px 0;}
.fx-wrapper .spark-svg{display:block;width:100%;height:90px;}
.fx-wrapper .spark-svg.driver-spark{height:60px;}
.fx-wrapper .spark-meta{display:flex;justify-content:space-between;font-size:10px;color:#94a3b8;margin-top:2px;}
.fx-wrapper .spark-extremes{display:flex;justify-content:space-between;font-size:11px;margin-top:4px;color:#475569;}
.fx-wrapper .spark-extremes b{font-variant-numeric:tabular-nums;}
.fx-wrapper .spark-tooltip{position:absolute;pointer-events:none;background:#0f172a;color:#fff;padding:6px 9px;border-radius:5px;font-size:11px;line-height:1.4;white-space:nowrap;display:none;z-index:10;transform:translate(-50%,-110%);box-shadow:0 4px 12px rgba(0,0,0,0.15);}
.fx-wrapper h3.sec-title{margin:24px 0 12px 0;font-size:16px;font-weight:600;color:#1a1a1a;}
</style>
<div class="fx-wrapper">

<h3 class="sec-title">¿Cómo leer este dashboard?</h3>
<div class="help-grid">
  <div class="help-card"><b>Banner verde / rojo</b><br>Verde = ningún movimiento extremo registrado hoy o esta semana. Rojo = una moneda ya cruzó un umbral histórico.</div>
  <div class="help-card"><b>Sparkline interactivo</b><br>Pasá el mouse sobre el gráfico para ver el valor de cada día. Las líneas punteadas marcan máx y mín del período de 60 días hábiles.</div>
  <div class="help-card"><b>Probabilidad de movimiento extremo</b><br>Probabilidad estimada por el modelo de que en los próximos 5 (o 20) días la moneda se mueva más que un umbral. Comparar contra el umbral. Ratio &gt; 2× = el modelo ve algo distinto a lo habitual.</div>
  <div class="help-card"><b>Dirección esperada</b><br><span class="dir-up">↑ USD sube</span> = la moneda local se debilita (mejor haber comprado dólares antes). <span class="dir-dn">↓ USD baja</span> = la moneda local se fortalece (mejor para vender dólares).</div>
  <div class="help-card"><b>Drivers globales</b><br>Variables que mueven los flotantes (BRL, CLP, COP, INR). Cada tarjeta muestra su evolución 60 días + cambios recientes.</div>
  <div class="help-card"><b>Limitación importante</b><br>El modelo acierta más que tirar moneda pero no es oráculo (AUC ≈ 0,57 a 5 días). Es señal complementaria al monitoreo, no sustituto. ARS sigue siendo difícil por su drift y régimen de cepo.</div>
</div>

<h3 class="sec-title">¿Qué mueve a cada moneda?</h3>
<div class="help-grid">
  <div class="help-card"><b>USDARS</b><br>Dominado por <b>política local</b>: cepos, devaluaciones administrativas, elecciones, FMI. Drivers globales explican &lt;5% de los movimientos. La soja ayuda en algunos episodios. Característica: el peso solo se devalúa, casi nunca se aprecia.</div>
  <div class="help-card"><b>USDBRL</b><br>Mezcla 60-40 entre <b>drivers globales</b> (DXY, VIX, hierro) y <b>política local</b> (fiscal, electoral). Sensible al "risk-off". Episodios: Lehman 2008, COVID 2020, elección Lula 2022.</div>
  <div class="help-card"><b>USDCLP</b><br>El más "limpio": <b>cobre + DXY</b>. Cuando el cobre cae fuerte, el peso chileno se debilita (Chile = ~50% exportaciones de cobre). Episodios: estallido social 2019, intervención BCCh 2022.</div>
  <div class="help-card"><b>USDCOP</b><br><b>Petróleo + risk-off</b>. Sensible al Brent (Colombia es petrolero neto). Política local mueve fuerte en eventos discretos pero no daily.</div>
  <div class="help-card"><b>USDINR</b><br><b>Managed float</b>: el RBI interviene activamente y suaviza shocks. Solo eventos globales muy fuertes (taper 2013, COVID, Fed hiking 2022) pasan.</div>
  <div class="help-card"><b>Drivers globales</b><br><b>DXY</b> afecta a todos los flotantes. <b>VIX</b> = BRL e INR sobre todo. <b>Cobre</b> = CLP. <b>Brent</b> = COP, INR. <b>Hierro</b> = BRL.</div>
</div>

<h3 class="sec-title">Estado por moneda</h3>
<div class="card-grid" id="fx-cards"></div>

<h3 class="sec-title">Drivers globales</h3>
<div class="driver-grid" id="fx-drivers"></div>

<h3 class="sec-title">Episodios recientes (últimos 180 días)</h3>
<div id="fx-episodes"></div>

</div>
<script>
const CARDS = __CARDS_DATA__;
const DRIVERS = __DRIVERS_DATA__;
const EPISODES = __EPISODES_DATA__;
const AFFECTS = __AFFECTS_DATA__;
const HUMAN_MAP = __HUMAN_MAP_DATA__;

const fmtRet = (v) => v == null ? '—' : (v > 0 ? '+' : '') + v.toFixed(2) + '%';
const cls = (v) => v == null ? '' : (v > 0 ? 'ret-up' : 'ret-dn');
function dirSpan(ret) {
  if (ret == null || Math.abs(ret) < 0.05) return '<span class="dir-flat">→ neutro</span>';
  if (ret > 0) return '<span class="dir-up">↑ USD sube</span>';
  return '<span class="dir-dn">↓ USD baja</span>';
}
function fmtDate(s){const m=["ene","feb","mar","abr","may","jun","jul","ago","sep","oct","nov","dic"];const p=s.split("-");return parseInt(p[2],10)+"-"+m[parseInt(p[1],10)-1];}
function fmtVal(v, ref){if (ref > 1000) return v.toFixed(2); if (ref > 100) return v.toFixed(2); if (ref > 10) return v.toFixed(3); return v.toFixed(4);}

function makeSparkline(vals, w, h, padTop, padBot, color, fill) {
  const padX = 4;
  const min = Math.min.apply(null, vals);
  const max = Math.max.apply(null, vals);
  const range = (max - min) || 1;
  const xFor = i => padX + (i / (vals.length - 1)) * (w - padX*2);
  const yFor = v => padTop + (1 - (v - min) / range) * (h - padTop - padBot);
  const pts = vals.map((v, i) => xFor(i).toFixed(1)+","+yFor(v).toFixed(1)).join(' ');
  const xN = xFor(vals.length-1);
  const minIdx = vals.indexOf(min), maxIdx = vals.indexOf(max);
  const yMin = yFor(min), yMax = yFor(max);
  const areaPath = "M "+xFor(0)+","+(h-padBot)+" L "+pts.split(' ').join(' L ')+" L "+xN+","+(h-padBot)+" Z";
  let svg = '';
  svg += '<path d="'+areaPath+'" fill="'+fill+'" stroke="none"/>';
  svg += '<line x1="'+padX+'" y1="'+yMax.toFixed(1)+'" x2="'+(w-padX)+'" y2="'+yMax.toFixed(1)+'" stroke="#cbd5e1" stroke-dasharray="2,3" stroke-width="0.7"/>';
  svg += '<line x1="'+padX+'" y1="'+yMin.toFixed(1)+'" x2="'+(w-padX)+'" y2="'+yMin.toFixed(1)+'" stroke="#cbd5e1" stroke-dasharray="2,3" stroke-width="0.7"/>';
  svg += '<polyline fill="none" stroke="'+color+'" stroke-width="1.6" points="'+pts+'"/>';
  svg += '<circle cx="'+xFor(maxIdx).toFixed(1)+'" cy="'+yMax.toFixed(1)+'" r="2.5" fill="#1e293b"/>';
  svg += '<circle cx="'+xFor(minIdx).toFixed(1)+'" cy="'+yMin.toFixed(1)+'" r="2.5" fill="#1e293b"/>';
  svg += '<circle cx="'+xN+'" cy="'+yFor(vals[vals.length-1])+'" r="3.5" fill="'+color+'" stroke="#fff" stroke-width="1.5"/>';
  svg += '<line class="hover-line" x1="0" y1="'+padTop+'" x2="0" y2="'+(h-padBot)+'" stroke="#94a3b8" stroke-width="1" stroke-dasharray="2,2" style="display:none"/>';
  svg += '<circle class="hover-dot" cx="0" cy="0" r="3" fill="'+color+'" stroke="#fff" stroke-width="1.5" style="display:none"/>';
  return svg;
}

function attachInteraction(svg, vals, dates, w, h, padTop, padBot, fmtFn) {
  const padX = 4;
  const wrap = svg.parentElement;
  const tooltip = wrap.querySelector('.spark-tooltip');
  const hoverLine = svg.querySelector('.hover-line');
  const hoverDot = svg.querySelector('.hover-dot');
  const min = Math.min.apply(null, vals), max = Math.max.apply(null, vals);
  const range = (max - min) || 1;
  const xFor = i => padX + (i / (vals.length - 1)) * (w - padX*2);
  const yFor = v => padTop + (1 - (v - min) / range) * (h - padTop - padBot);
  svg.addEventListener('mousemove', function(e) {
    const rect = svg.getBoundingClientRect();
    const xPx = e.clientX - rect.left;
    let idx = Math.round(((xPx / rect.width) * w - padX) / (w - padX*2) * (vals.length - 1));
    idx = Math.max(0, Math.min(vals.length - 1, idx));
    const cx = xFor(idx), cy = yFor(vals[idx]);
    hoverLine.setAttribute('x1', cx); hoverLine.setAttribute('x2', cx);
    hoverLine.style.display = '';
    hoverDot.setAttribute('cx', cx); hoverDot.setAttribute('cy', cy);
    hoverDot.style.display = '';
    const wrapRect = wrap.getBoundingClientRect();
    tooltip.style.display = 'block';
    tooltip.innerHTML = '<b>'+fmtDate(dates[idx])+'</b><br>'+fmtFn(vals[idx]);
    tooltip.style.left = (e.clientX - wrapRect.left) + 'px';
    tooltip.style.top = ((cy / h) * rect.height + 4) + 'px';
  });
  svg.addEventListener('mouseleave', function() {
    hoverLine.style.display = 'none';
    hoverDot.style.display = 'none';
    tooltip.style.display = 'none';
  });
}

const cardsDiv = document.getElementById('fx-cards');
for (const c of CARDS) {
  const lastUp = c.spark[c.spark.length-1] >= c.spark[0];
  const color = lastUp ? '#dc2626' : '#16a34a';
  const fill = lastUp ? 'rgba(220,38,38,0.08)' : 'rgba(22,163,74,0.08)';
  const sparkInner = makeSparkline(c.spark, 320, 90, 12, 18, color, fill);
  const p5 = c.pred_5d, p20 = c.pred_20d;
  const probWidth5 = Math.min(100, p5.prob * 2.5);
  const probWidth20 = Math.min(100, p20.prob * 2.5);
  const alarmHtml = (c.alarma_1d || c.alarma_5d) ? '<span class="alarm-on">ALARMA</span>' : '<span class="alarm-off">ok</span>';
  const html = ''
    + '<div class="card">'
    + '<div class="ccy-name">' + c.ccy + alarmHtml + '</div>'
    + '<div class="px">' + c.px.toFixed(c.px > 100 ? 2 : 4) + '</div>'
    + '<div class="spark-wrap" data-ccy="' + c.ccy + '">'
    +   '<svg class="spark-svg" viewBox="0 0 320 90" preserveAspectRatio="none">' + sparkInner + '</svg>'
    +   '<div class="spark-tooltip"></div>'
    +   '<div class="spark-meta"><span>' + fmtDate(c.spark_first_date) + '</span><span>últimos 60 días hábiles</span><span>' + fmtDate(c.spark_last_date) + '</span></div>'
    +   '<div class="spark-extremes"><span>Máx: <b>' + fmtVal(c.spark_max, c.px) + '</b> <span style="color:#94a3b8">(' + fmtDate(c.spark_max_date) + ')</span></span><span>Mín: <b>' + fmtVal(c.spark_min, c.px) + '</b> <span style="color:#94a3b8">(' + fmtDate(c.spark_min_date) + ')</span></span></div>'
    + '</div>'
    + '<div class="row"><span>Hoy (1d)</span><span class="' + cls(c.ret_1d) + '">' + fmtRet(c.ret_1d) + '</span></div>'
    + '<div class="row"><span>Esta semana (5d)</span><span class="' + cls(c.ret_5d) + '">' + fmtRet(c.ret_5d) + '</span></div>'
    + '<div class="row"><span>Este mes (20d)</span><span class="' + cls(c.ret_20d) + '">' + fmtRet(c.ret_20d) + '</span></div>'
    + '<div class="pred-block"><div class="ph">Próximos 5 días</div><div class="pmain">' + dirSpan(p5.ret) + ' · esperado ' + fmtRet(p5.ret) + '</div><div class="psub">P(movimiento extremo): ' + p5.prob.toFixed(1) + '% · umbral ' + p5.umbral + '%</div><div class="prob-bar"><div class="prob-bar-fill" style="width:' + probWidth5 + '%"></div></div></div>'
    + '<div class="pred-block"><div class="ph">Próximos 20 días</div><div class="pmain">' + dirSpan(p20.ret) + ' · esperado ' + fmtRet(p20.ret) + '</div><div class="psub">P(movimiento extremo): ' + p20.prob.toFixed(1) + '% · umbral ' + p20.umbral + '%</div><div class="prob-bar"><div class="prob-bar-fill" style="width:' + probWidth20 + '%"></div></div></div>'
    + '</div>';
  cardsDiv.insertAdjacentHTML('beforeend', html);
}
document.querySelectorAll('.spark-wrap[data-ccy]').forEach(wrap => {
  const ccy = wrap.dataset.ccy;
  const c = CARDS.find(x => x.ccy === ccy);
  const svg = wrap.querySelector('.spark-svg');
  attachInteraction(svg, c.spark, c.spark_dates, 320, 90, 12, 18, function(v){ return fmtVal(v, v); });
});

const drvDiv = document.getElementById('fx-drivers');
for (const name of Object.keys(DRIVERS)) {
  const d = DRIVERS[name];
  const lastUp = d.spark[d.spark.length-1] >= d.spark[0];
  const color = lastUp ? '#dc2626' : '#16a34a';
  const fill = lastUp ? 'rgba(220,38,38,0.08)' : 'rgba(22,163,74,0.08)';
  const sparkInner = makeSparkline(d.spark, 280, 60, 8, 12, color, fill);
  const c1 = d.chg_1d == null ? '' : (d.chg_1d > 0 ? 'ret-up' : 'ret-dn');
  const c5 = d.chg_5d_pct == null ? '' : (d.chg_5d_pct > 0 ? 'ret-up' : 'ret-dn');
  const c20 = d.chg_20d_pct == null ? '' : (d.chg_20d_pct > 0 ? 'ret-up' : 'ret-dn');
  const fmt = v => v == null ? '—' : (v > 100 ? v.toFixed(2) : v.toFixed(3));
  const fmtP = v => v == null ? '—' : (v > 0 ? '+' : '') + v.toFixed(2) + '%';
  const fmtC = v => v == null ? '—' : (v > 0 ? '+' : '') + v.toFixed(3);
  const html = ''
    + '<div class="driver-card">'
    + '<div class="driver-name">' + name + '</div>'
    + '<div class="driver-val">' + fmt(d.lvl) + '</div>'
    + '<div class="spark-wrap" data-drv="' + name + '">'
    +   '<svg class="spark-svg driver-spark" viewBox="0 0 280 60" preserveAspectRatio="none">' + sparkInner + '</svg>'
    +   '<div class="spark-tooltip"></div>'
    + '</div>'
    + '<div class="driver-chgs">'
    +   '<span>1d <b class="' + c1 + '">' + fmtC(d.chg_1d) + '</b></span>'
    +   '<span>5d <b class="' + c5 + '">' + fmtP(d.chg_5d_pct) + '</b></span>'
    +   '<span>20d <b class="' + c20 + '">' + fmtP(d.chg_20d_pct) + '</b></span>'
    + '</div>'
    + '<div class="driver-affects">' + (AFFECTS[name] || '') + '</div>'
    + '</div>';
  drvDiv.insertAdjacentHTML('beforeend', html);
}
document.querySelectorAll('.spark-wrap[data-drv]').forEach(wrap => {
  const name = wrap.dataset.drv;
  const d = DRIVERS[name];
  if (!d) return;
  const svg = wrap.querySelector('.spark-svg');
  attachInteraction(svg, d.spark, d.spark_dates, 280, 60, 8, 12, function(v){ return v.toFixed(v > 100 ? 2 : 3); });
});

const epDiv = document.getElementById('fx-episodes');
function humanize(s) {
  if (!s) return '—';
  return s.split('+').map(t => HUMAN_MAP[t] || t).join(' + ');
}
if (EPISODES.length === 0) {
  epDiv.innerHTML = '<p style="color:#64748b;font-size:13px;">Sin episodios &gt; umbral en los últimos 180 días — período tranquilo.</p>';
} else {
  let h = '<table class="ep-table"><thead><tr><th>Pico</th><th>Moneda</th><th>Dirección</th><th class="num">Magnitud</th><th class="num">Días al pico</th><th>¿Permanente?</th><th>Causa probable</th></tr></thead><tbody>';
  for (const e of EPISODES) {
    const dirText = e.direction === "up" ? '↑ USD subió' : '↓ USD bajó';
    const dirCls = e.direction === "up" ? 'dir-up' : 'dir-dn';
    const evNote = (e.known_event && e.known_event !== "" && e.known_event !== "-" && e.known_event !== "nan")
      ? '<br><small><b>Evento:</b> ' + e.known_event + '</small>' : '';
    h += '<tr>'
      + '<td>' + e.peak_date + '</td>'
      + '<td><strong>' + e.ccy + '</strong></td>'
      + '<td><span class="' + dirCls + '">' + dirText + '</span></td>'
      + '<td class="num ' + (e.mag_pct > 0 ? 'ret-up' : 'ret-dn') + '">' + (e.mag_pct > 0 ? '+' : '') + e.mag_pct.toFixed(2) + '%</td>'
      + '<td class="num">' + e.days_to_peak + '</td>'
      + '<td>' + (e.permanent ? 'Sí (no se corrigió)' : 'No (se revirtió)') + '</td>'
      + '<td style="font-size:12px;color:#475569;">' + humanize(e.driver_class) + evNote + '</td>'
      + '</tr>';
  }
  h += '</tbody></table>';
  epDiv.innerHTML = h;
}
</script>
"""


def render_html(cards_data, drivers_data, episodes_data):
    return (HTML_TEMPLATE
        .replace("__CARDS_DATA__", json.dumps(cards_data))
        .replace("__DRIVERS_DATA__", json.dumps(drivers_data))
        .replace("__EPISODES_DATA__", json.dumps(episodes_data))
        .replace("__AFFECTS_DATA__", json.dumps(DRIVER_AFFECTS))
        .replace("__HUMAN_MAP_DATA__", json.dumps(DRIVER_HUMAN_MAP))
    )


# ---------------- Main ----------------

def main():
    st.title("Sistema de alarmas FX")

    with st.sidebar:
        st.markdown("### Acerca")
        st.markdown(
            "Dashboard predictivo de tipo de cambio: USD vs ARS / BRL / CLP / COP / INR. "
            "Cada visita baja datos en vivo de Yahoo Finance + FRED, computa features, "
            "predice con LightGBM y detecta episodios sobre la marcha."
        )
        st.markdown("### Limitaciones")
        st.markdown(
            "- AUC ≈ 0,57 a 5 días para los flotantes. Es señal modesta, no oráculo.\n"
            "- ARS está dominado por política local; los drivers globales explican <5%.\n"
            "- Si FRED no responde, se usa snapshot del proyecto como fallback."
        )
        st.markdown("---")
        if st.button("🔄 Forzar recarga de datos"):
            st.cache_data.clear()
            st.rerun()

    fx_df, drv_df = fetch_all_data()
    fred, fred_source = fetch_fred_drivers()

    if fx_df.empty:
        st.error("No pude bajar datos de Yahoo Finance. Probá refrescar la página en unos minutos.")
        st.stop()

    preds = predict_today(fx_df, drv_df, fred)

    cards_data = []
    for ccy in FX_COLS:
        if ccy not in fx_df.columns: continue
        cd = card_data(ccy, fx_df[ccy], preds.get(ccy))
        if cd: cards_data.append(cd)

    drivers_data = {}
    for d in ["DXY", "VIX", "DGS10", "BRENT", "COPPER", "GOLD"]:
        if d in drv_df.columns and not drv_df[d].dropna().empty:
            card = driver_card_data(d, drv_df[d])
            if card: drivers_data[d] = card

    episodes_data = detect_episodes_live(fx_df, drv_df, fred)

    last_date = fx_df.index.max().strftime("%Y-%m-%d")
    n_alarmas = sum(1 for c in cards_data if c["alarma_1d"] or c["alarma_5d"])
    if n_alarmas:
        msg = ", ".join(f"{c['ccy']}" for c in cards_data if c["alarma_1d"] or c["alarma_5d"])
        st.error(f"⚠️ Alarmas reactivas activas: {msg}")
    else:
        st.success(f"✅ Sin alarmas reactivas activas — datos al cierre del {last_date}")

    import streamlit.components.v1 as components
    html = render_html(cards_data, drivers_data, episodes_data)
    height = 2200 + len(episodes_data) * 50
    components.html(html, height=height, scrolling=True)

    fred_msg = {"live": "FRED en vivo", "snapshot": "FRED snapshot (fallback)", "none": "sin FRED"}.get(fred_source, "")
    st.caption(
        f"Datos: Yahoo Finance (en vivo, caché 1h) + {fred_msg}. "
        f"Episodios detectados sobre la marcha. "
        f"Última fecha: {last_date}."
    )


if __name__ == "__main__":
    main()
