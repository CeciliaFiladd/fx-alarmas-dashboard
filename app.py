"""
Sistema de alarmas FX — Streamlit app.

Cada visita: baja datos de Yahoo Finance en vivo, computa features,
predice con los modelos LightGBM persistidos, y renderiza el dashboard.
"""
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, timezone
import math

import numpy as np
import pandas as pd
import requests
import streamlit as st
import lightgbm as lgb

st.set_page_config(
    page_title="Sistema de alarmas FX",
    page_icon="📊",
    layout="wide",
)

# ---------------- Constantes ----------------

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
SNAPSHOTS = ROOT / "snapshots"

FX_COLS = ["USDARS", "USDBRL", "USDCLP", "USDCOP", "USDINR"]

# Tickers Yahoo
YAHOO_TICKERS = {
    "USDARS": "USDARS=X",
    "USDBRL": "USDBRL=X",
    "USDCLP": "USDCLP=X",
    "USDCOP": "USDCOP=X",
    "USDINR": "USDINR=X",
    "DXY":    "DX-Y.NYB",
    "VIX":    "^VIX",
    "DGS10":  "^TNX",
    "BRENT":  "BZ=F",
    "WTI":    "CL=F",
    "COPPER": "HG=F",
    "GOLD":   "GC=F",
    "SOYBEANS": "ZS=F",
    "IRON":   "TIO=F",
}

PRINCIPAL_COMM = {
    "USDARS": "SOYBEANS",
    "USDBRL": "IRON",
    "USDCLP": "COPPER",
    "USDCOP": "BRENT",
    "USDINR": "BRENT",
}

GLOBAL_DRIVERS = ["DXY", "VIX", "DGS10", "DFF", "T10Y2Y"]
DRIVERS_WITH_LEVEL = ["VIX", "DGS10", "DFF", "T10Y2Y"]

THRESHOLDS_1D = {"USDARS": 0.05, "USDBRL": 0.02, "USDCLP": 0.015, "USDCOP": 0.015, "USDINR": 0.008}
THRESHOLDS_5D = {"USDARS": 0.10, "USDBRL": 0.05, "USDCLP": 0.04, "USDCOP": 0.04, "USDINR": 0.02}

EXT_THR = {
    ("USDARS","5d"):0.05, ("USDARS","20d"):0.10,
    ("USDBRL","5d"):0.025, ("USDBRL","20d"):0.05,
    ("USDCLP","5d"):0.020, ("USDCLP","20d"):0.04,
    ("USDCOP","5d"):0.022, ("USDCOP","20d"):0.045,
    ("USDINR","5d"):0.012, ("USDINR","20d"):0.025,
}

DRIVER_AFFECTS = {
    "DXY":   "Afecta a TODOS los flotantes. Si sube, las EM se debilitan.",
    "VIX":   "Miedo global. BRL e INR son los más sensibles.",
    "DGS10": "Yield US 10Y. Sube → fuga de EM.",
    "BRENT": "Petróleo. Afecta a USDCOP y USDINR.",
    "COPPER":"Cobre. Driver casi exclusivo del USDCLP.",
    "GOLD":  "Oro. Refugio de valor en risk-off.",
}

# ---------------- Fetch Yahoo ----------------

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_yahoo_one(ticker: str, days: int = 400) -> pd.Series:
    """Baja `days` días de cierre desde Yahoo. Devuelve serie."""
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
        if not res:
            return pd.Series(dtype=float)
        ts = res[0].get("timestamp", []) or []
        closes = res[0].get("indicators", {}).get("quote", [{}])[0].get("close", []) or []
        if not ts:
            return pd.Series(dtype=float)
        idx = pd.to_datetime([pd.Timestamp(t, unit="s", tz="UTC") for t in ts]).tz_convert(None).normalize()
        s = pd.Series(closes, index=idx)
        s = s[~s.index.duplicated(keep="last")]
        return s.dropna()
    except Exception:
        return pd.Series(dtype=float)


@st.cache_data(ttl=3600, show_spinner="Bajando datos en vivo de Yahoo Finance...")
def fetch_all_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve (fx_df, drv_df). drv_df puede no tener DFF / T10Y2Y, completar con snapshot."""
    series = {}
    for name, ticker in YAHOO_TICKERS.items():
        s = fetch_yahoo_one(ticker)
        if not s.empty:
            series[name] = s
    if not series:
        return pd.DataFrame(), pd.DataFrame()

    # Construir índice de business days unificado
    all_idx = sorted(set().union(*[s.index for s in series.values()]))
    bdays = pd.bdate_range(min(all_idx), max(all_idx))

    # Reindex+ffill cada serie individualmente
    aligned = {n: s.reindex(bdays, method="ffill").ffill() for n, s in series.items()}

    fx_df = pd.DataFrame({k: aligned[k] for k in FX_COLS if k in aligned})

    drv_cols = ["DXY", "VIX", "DGS10", "BRENT", "WTI", "COPPER", "GOLD", "SOYBEANS", "IRON"]
    drv_df = pd.DataFrame({k: aligned[k] for k in drv_cols if k in aligned})

    # Yahoo ^TNX es el yield * 100, ej. 4.34 → fine; pero a veces viene como 43.4
    if "DGS10" in drv_df.columns and drv_df["DGS10"].median() > 30:
        drv_df["DGS10"] = drv_df["DGS10"] / 10

    return fx_df, drv_df


@st.cache_data(ttl=86400, show_spinner=False)
def load_snapshot_fred() -> pd.DataFrame:
    """Carga el snapshot estático de FRED (DFF, T10Y2Y)."""
    path = SNAPSHOTS / "fred_recent.json"
    if not path.exists():
        return pd.DataFrame()
    j = json.loads(path.read_text())
    out = {}
    for col, days in j.items():
        s = pd.Series({pd.Timestamp(d): v for d, v in days.items()})
        out[col] = s.sort_index()
    return pd.DataFrame(out)


# ---------------- Features (mismo pipeline que scripts/08) ----------------

def safe_log(s):
    return np.log(s.where(s > 0))


def build_self(fx):
    p = fx
    lp = safe_log(p)
    ret1 = lp.diff()
    out = pd.DataFrame(index=p.index)
    out["ret_1d"] = ret1
    out["ret_5d"] = lp.diff(5)
    out["ret_20d"] = lp.diff(20)
    out["vol_20d"] = ret1.rolling(20).std()
    out["vol_60d"] = ret1.rolling(60).std()
    sma20 = p.rolling(20).mean()
    sma60 = p.rolling(60).mean()
    std60 = p.rolling(60).std()
    out["dist_sma20"] = (p - sma20) / sma20
    out["zscore_60d"] = (p - sma60) / std60
    return out


def build_driver(s, name, with_level):
    out = pd.DataFrame(index=s.index)
    if with_level:
        out[f"{name}_lvl"] = s
    lp = safe_log(s)
    out[f"{name}_ret_5d"] = lp.diff(5)
    out[f"{name}_ret_20d"] = lp.diff(20)
    ret1 = s.diff() if with_level else lp.diff()
    out[f"{name}_vol_20d"] = ret1.rolling(20).std()
    return out


def build_features_for(ccy: str, fx_df: pd.DataFrame, drv_df: pd.DataFrame, fred: pd.DataFrame) -> pd.DataFrame:
    feats = [build_self(fx_df[ccy])]
    # alinear FRED al índice de fx
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
            clf_path = MODELS / f"gbm_{ccy}_{h}.txt"
            reg_path = MODELS / f"gbm_reg_{ccy}_{h}.txt"
            if clf_path.exists():
                out[(ccy, h, "clf")] = lgb.Booster(model_file=str(clf_path))
            if reg_path.exists():
                out[(ccy, h, "reg")] = lgb.Booster(model_file=str(reg_path))
    return out, feat_cols


def predict_today(fx_df, drv_df, fred):
    models, feat_cols_by_ccy = load_models()
    out = {}
    for ccy in FX_COLS:
        if ccy not in fx_df.columns or fx_df[ccy].dropna().empty:
            continue
        feat = build_features_for(ccy, fx_df, drv_df, fred)
        # garantizar que están todas las cols esperadas (en el mismo orden que el modelo)
        expected = feat_cols_by_ccy.get(ccy, list(feat.columns))
        for col in expected:
            if col not in feat.columns:
                feat[col] = np.nan
        feat = feat[expected]
        last_idx = feat.dropna().index.max() if not feat.dropna().empty else feat.index.max()
        x_now = feat.loc[[last_idx]].fillna(0)

        preds = {}
        for h in ["5d", "20d"]:
            clf = models.get((ccy, h, "clf"))
            reg = models.get((ccy, h, "reg"))
            prob = float(clf.predict(x_now)[0]) if clf else None
            ret_e = float(reg.predict(x_now)[0]) if reg else None
            preds[h] = {"prob": prob, "ret": ret_e}
        out[ccy] = preds
    return out


# ---------------- Render helpers ----------------

def card_html(ccy, fx_series, pred, react):
    s = fx_series.dropna()
    if s.empty:
        return f"<div class='card'>{ccy}: sin datos</div>"
    px = float(s.iloc[-1])
    ret_1d = float(np.log(s.iloc[-1]/s.iloc[-2])*100) if len(s) >= 2 else None
    ret_5d = float(np.log(s.iloc[-1]/s.iloc[-6])*100) if len(s) >= 6 else None
    ret_20d = float(np.log(s.iloc[-1]/s.iloc[-21])*100) if len(s) >= 21 else None
    sp = s.iloc[-60:]
    vals = sp.tolist()
    dates = [d.strftime("%Y-%m-%d") for d in sp.index]

    p5 = pred.get("5d", {}) if pred else {}
    p20 = pred.get("20d", {}) if pred else {}
    base5 = EXT_THR[(ccy, "5d")] * 100  # umbral, no base rate
    base20 = EXT_THR[(ccy, "20d")] * 100

    alarm_1d = abs(ret_1d) > THRESHOLDS_1D[ccy] * 100 if ret_1d else False
    alarm_5d = abs(ret_5d) > THRESHOLDS_5D[ccy] * 100 if ret_5d else False

    return {
        "ccy": ccy,
        "px": px,
        "ret_1d": ret_1d, "ret_5d": ret_5d, "ret_20d": ret_20d,
        "spark": vals, "spark_dates": dates,
        "spark_min": min(vals), "spark_max": max(vals),
        "spark_min_date": dates[vals.index(min(vals))],
        "spark_max_date": dates[vals.index(max(vals))],
        "spark_first": vals[0], "spark_first_date": dates[0],
        "spark_last": vals[-1], "spark_last_date": dates[-1],
        "alarma_1d": alarm_1d, "alarma_5d": alarm_5d,
        "pred_5d": {"prob": (p5.get("prob") or 0) * 100, "ret": (p5.get("ret") or 0) * 100, "umbral": base5},
        "pred_20d": {"prob": (p20.get("prob") or 0) * 100, "ret": (p20.get("ret") or 0) * 100, "umbral": base20},
    }


def render_html(cards_data, drivers_data, last_date):
    """Genera el bloque HTML del dashboard (cards + drivers)."""
    cards_json = json.dumps(cards_data)
    drivers_json = json.dumps(drivers_data)
    affects_json = json.dumps(DRIVER_AFFECTS)

    return f"""
<style>
.fx-wrapper {{font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif;}}
.fx-wrapper .card-grid{{display:grid;grid-template-columns:repeat(auto-fit, minmax(310px,1fr));gap:14px;}}
.fx-wrapper .card{{background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:16px;box-shadow:0 1px 2px rgba(0,0,0,0.04);}}
.fx-wrapper .ccy-name{{font-size:16px;font-weight:600;margin-bottom:6px;display:flex;justify-content:space-between;align-items:center;color:#0f172a;}}
.fx-wrapper .px{{font-size:22px;font-weight:600;color:#0f172a;}}
.fx-wrapper .row{{display:flex;justify-content:space-between;padding:3px 0;font-size:13px;color:#64748b;}}
.fx-wrapper .row span:nth-child(2){{color:#0f172a;font-weight:500;font-variant-numeric:tabular-nums;}}
.fx-wrapper .ret-up{{color:#dc2626;}}
.fx-wrapper .ret-dn{{color:#16a34a;}}
.fx-wrapper .alarm-on{{background:#dc2626;color:#fff;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600;}}
.fx-wrapper .alarm-off{{color:#94a3b8;font-size:11px;}}
.fx-wrapper .prob-bar{{height:6px;background:#e2e8f0;border-radius:3px;overflow:hidden;margin:4px 0 8px;}}
.fx-wrapper .prob-bar-fill{{height:100%;background:#3b82f6;}}
.fx-wrapper .pred-block{{background:#f8fafc;border:1px solid #e2e8f0;border-radius:5px;padding:8px 10px;margin:8px 0;font-size:12px;}}
.fx-wrapper .pred-block .ph{{font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:0.4px;margin-bottom:4px;}}
.fx-wrapper .pred-block .pmain{{font-size:13px;font-weight:500;margin-bottom:3px;color:#0f172a;}}
.fx-wrapper .pred-block .psub{{font-size:11px;color:#64748b;}}
.fx-wrapper .dir-up{{color:#dc2626;font-weight:600;}}
.fx-wrapper .dir-dn{{color:#16a34a;font-weight:600;}}
.fx-wrapper .dir-flat{{color:#64748b;}}
.fx-wrapper .driver-tile{{display:inline-block;background:#fff;border:1px solid #e2e8f0;padding:10px 14px;border-radius:6px;margin-right:8px;margin-bottom:8px;min-width:170px;max-width:240px;vertical-align:top;}}
.fx-wrapper .driver-name{{font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:0.5px;}}
.fx-wrapper .driver-val{{font-size:18px;font-weight:600;margin:4px 0;color:#0f172a;}}
.fx-wrapper .driver-chg{{font-size:12px;}}
.fx-wrapper .driver-affects{{font-size:10px;color:#64748b;margin-top:6px;line-height:1.4;}}
.fx-wrapper .spark-wrap{{position:relative;margin:8px 0 6px 0;}}
.fx-wrapper .spark-svg{{display:block;width:100%;height:90px;}}
.fx-wrapper .spark-meta{{display:flex;justify-content:space-between;font-size:10px;color:#94a3b8;margin-top:2px;}}
.fx-wrapper .spark-extremes{{display:flex;justify-content:space-between;font-size:11px;margin-top:4px;color:#475569;}}
.fx-wrapper .spark-extremes b{{font-variant-numeric:tabular-nums;}}
.fx-wrapper .spark-tooltip{{position:absolute;pointer-events:none;background:#0f172a;color:#fff;padding:6px 9px;border-radius:5px;font-size:11px;line-height:1.4;white-space:nowrap;display:none;z-index:10;transform:translate(-50%,-110%);box-shadow:0 4px 12px rgba(0,0,0,0.15);}}
</style>
<div class="fx-wrapper">
<div class="card-grid" id="fx-cards"></div>
<div style="margin-top:24px;border-bottom:1px solid #e2e8f0;padding-bottom:8px;font-weight:600;font-size:16px;">Drivers globales</div>
<div id="fx-drivers" style="margin-top:10px;"></div>
</div>
<script>
const CARDS = {cards_json};
const DRIVERS = {drivers_json};
const AFFECTS = {affects_json};

const fmtRet = (v) => v == null ? '—' : (v > 0 ? '+' : '') + v.toFixed(2) + '%';
const cls = (v) => v == null ? '' : (v > 0 ? 'ret-up' : 'ret-dn');
function dirSpan(ret) {{
  if (ret == null || Math.abs(ret) < 0.05) return '<span class="dir-flat">→ neutro</span>';
  if (ret > 0) return '<span class="dir-up">↑ USD sube</span>';
  return '<span class="dir-dn">↓ USD baja</span>';
}}
function fmtDate(s){{const m=["ene","feb","mar","abr","may","jun","jul","ago","sep","oct","nov","dic"];const [y,mn,d]=s.split("-");return parseInt(d,10)+"-"+m[parseInt(mn,10)-1];}}
function fmtVal(v, ref){{if (ref > 1000) return v.toFixed(2);if (ref > 100) return v.toFixed(2);if (ref > 10) return v.toFixed(3);return v.toFixed(4);}}

function renderSparkline(c) {{
  const W = 320, H = 90, padX = 4, padTop = 12, padBot = 18;
  const vals = c.spark, min = c.spark_min, max = c.spark_max;
  const range = (max - min) || 1;
  const xFor = i => padX + (i / (vals.length - 1)) * (W - padX*2);
  const yFor = v => padTop + (1 - (v - min) / range) * (H - padTop - padBot);
  const pts = vals.map((v, i) => xFor(i).toFixed(1)+","+yFor(v).toFixed(1)).join(' ');
  const lastUp = vals[vals.length-1] >= vals[0];
  const lineColor = lastUp ? '#dc2626' : '#16a34a';
  const fillColor = lastUp ? 'rgba(220,38,38,0.08)' : 'rgba(22,163,74,0.08)';
  const minIdx = vals.indexOf(min), maxIdx = vals.indexOf(max);
  const xN = xFor(vals.length - 1), yMin = yFor(min), yMax = yFor(max);
  const areaPath = "M "+xFor(0)+","+(H-padBot)+" L "+pts.split(' ').join(' L ')+" L "+xN+","+(H-padBot)+" Z";

  let svg = '<svg class="spark-svg" viewBox="0 0 '+W+' '+H+'" preserveAspectRatio="none">';
  svg += '<path d="'+areaPath+'" fill="'+fillColor+'" stroke="none"/>';
  svg += '<line x1="'+padX+'" y1="'+yMax+'" x2="'+(W-padX)+'" y2="'+yMax+'" stroke="#cbd5e1" stroke-dasharray="2,3" stroke-width="0.7"/>';
  svg += '<line x1="'+padX+'" y1="'+yMin+'" x2="'+(W-padX)+'" y2="'+yMin+'" stroke="#cbd5e1" stroke-dasharray="2,3" stroke-width="0.7"/>';
  svg += '<polyline fill="none" stroke="'+lineColor+'" stroke-width="1.6" points="'+pts+'"/>';
  svg += '<circle cx="'+xFor(maxIdx)+'" cy="'+yMax+'" r="2.5" fill="#1e293b"/>';
  svg += '<circle cx="'+xFor(minIdx)+'" cy="'+yMin+'" r="2.5" fill="#1e293b"/>';
  svg += '<circle cx="'+xN+'" cy="'+yFor(vals[vals.length-1])+'" r="3.5" fill="'+lineColor+'" stroke="#fff" stroke-width="1.5"/>';
  svg += '<line class="hover-line" x1="0" y1="'+padTop+'" x2="0" y2="'+(H-padBot)+'" stroke="#94a3b8" stroke-width="1" stroke-dasharray="2,2" style="display:none"/>';
  svg += '<circle class="hover-dot" cx="0" cy="0" r="3" fill="'+lineColor+'" stroke="#fff" stroke-width="1.5" style="display:none"/>';
  svg += '</svg>';
  return svg;
}}

function attachInteraction(svg, c) {{
  const wrap = svg.parentElement;
  const tooltip = wrap.querySelector('.spark-tooltip');
  const hoverLine = svg.querySelector('.hover-line');
  const hoverDot = svg.querySelector('.hover-dot');
  const W = 320, H = 90, padX = 4, padTop = 12, padBot = 18;
  const vals = c.spark, dates = c.spark_dates;
  const min = c.spark_min, max = c.spark_max, range = (max - min) || 1;
  const xFor = i => padX + (i / (vals.length - 1)) * (W - padX*2);
  const yFor = v => padTop + (1 - (v - min) / range) * (H - padTop - padBot);
  svg.addEventListener('mousemove', e => {{
    const rect = svg.getBoundingClientRect();
    const xPx = e.clientX - rect.left;
    const xViewBox = (xPx / rect.width) * W;
    let idx = Math.round((xViewBox - padX) / (W - padX*2) * (vals.length - 1));
    idx = Math.max(0, Math.min(vals.length - 1, idx));
    const cx = xFor(idx), cy = yFor(vals[idx]);
    hoverLine.setAttribute('x1', cx); hoverLine.setAttribute('x2', cx);
    hoverLine.style.display = '';
    hoverDot.setAttribute('cx', cx); hoverDot.setAttribute('cy', cy);
    hoverDot.style.display = '';
    const wrapRect = wrap.getBoundingClientRect();
    tooltip.style.display = 'block';
    tooltip.innerHTML = '<b>'+fmtDate(dates[idx])+'</b><br>'+fmtVal(vals[idx], vals[idx]);
    tooltip.style.left = (e.clientX - wrapRect.left) + 'px';
    tooltip.style.top = ((cy / H) * rect.height + 4) + 'px';
  }});
  svg.addEventListener('mouseleave', () => {{
    hoverLine.style.display = 'none';
    hoverDot.style.display = 'none';
    tooltip.style.display = 'none';
  }});
}}

const cardsDiv = document.getElementById('fx-cards');
for (const c of CARDS) {{
  const p5 = c.pred_5d, p20 = c.pred_20d;
  const ratio5 = p5.prob > 0 && p5.umbral > 0 ? (p5.prob / p5.umbral).toFixed(2) : '—';
  cardsDiv.insertAdjacentHTML('beforeend', `
    <div class="card">
      <div class="ccy-name">${{c.ccy}}
        ${{(c.alarma_1d || c.alarma_5d) ? '<span class="alarm-on">ALARMA</span>' : '<span class="alarm-off">ok</span>'}}
      </div>
      <div class="px">${{c.px.toFixed(c.px > 100 ? 2 : 4)}}</div>
      <div class="spark-wrap" data-ccy="${{c.ccy}}">
        ${{renderSparkline(c)}}
        <div class="spark-tooltip"></div>
        <div class="spark-meta">
          <span>${{fmtDate(c.spark_first_date)}}</span>
          <span>últimos 60 días hábiles</span>
          <span>${{fmtDate(c.spark_last_date)}}</span>
        </div>
        <div class="spark-extremes">
          <span>Máx: <b>${{fmtVal(c.spark_max, c.px)}}</b> <span style="color:#94a3b8">(${{fmtDate(c.spark_max_date)}})</span></span>
          <span>Mín: <b>${{fmtVal(c.spark_min, c.px)}}</b> <span style="color:#94a3b8">(${{fmtDate(c.spark_min_date)}})</span></span>
        </div>
      </div>
      <div class="row"><span>Hoy (1d)</span><span class="${{cls(c.ret_1d)}}">${{fmtRet(c.ret_1d)}}</span></div>
      <div class="row"><span>Esta semana (5d)</span><span class="${{cls(c.ret_5d)}}">${{fmtRet(c.ret_5d)}}</span></div>
      <div class="row"><span>Este mes (20d)</span><span class="${{cls(c.ret_20d)}}">${{fmtRet(c.ret_20d)}}</span></div>
      <div class="pred-block">
        <div class="ph">Próximos 5 días</div>
        <div class="pmain">${{dirSpan(p5.ret)}} · esperado ${{fmtRet(p5.ret)}}</div>
        <div class="psub">P(movimiento extremo): ${{p5.prob.toFixed(1)}}% · umbral ${{p5.umbral}}%</div>
        <div class="prob-bar"><div class="prob-bar-fill" style="width:${{Math.min(100, p5.prob * 2.5)}}%"></div></div>
      </div>
      <div class="pred-block">
        <div class="ph">Próximos 20 días</div>
        <div class="pmain">${{dirSpan(p20.ret)}} · esperado ${{fmtRet(p20.ret)}}</div>
        <div class="psub">P(movimiento extremo): ${{p20.prob.toFixed(1)}}% · umbral ${{p20.umbral}}%</div>
        <div class="prob-bar"><div class="prob-bar-fill" style="width:${{Math.min(100, p20.prob * 2.5)}}%"></div></div>
      </div>
    </div>
  `);
}}
document.querySelectorAll('.spark-wrap').forEach(wrap => {{
  const ccy = wrap.dataset.ccy;
  const c = CARDS.find(x => x.ccy === ccy);
  attachInteraction(wrap.querySelector('.spark-svg'), c);
}});

const drvDiv = document.getElementById('fx-drivers');
for (const [name, d] of Object.entries(DRIVERS)) {{
  const c = d.chg_5d_pct == null ? '' : (d.chg_5d_pct > 0 ? 'ret-up' : 'ret-dn');
  drvDiv.insertAdjacentHTML('beforeend', `
    <div class="driver-tile">
      <div class="driver-name">${{name}}</div>
      <div class="driver-val">${{d.lvl.toFixed(d.lvl > 100 ? 2 : 3)}}</div>
      <div class="driver-chg ${{c}}">${{d.chg_5d_pct == null ? '' : (d.chg_5d_pct > 0 ? '+' : '') + d.chg_5d_pct.toFixed(2) + '% · 5d'}}</div>
      <div class="driver-affects">${{AFFECTS[name] || ''}}</div>
    </div>
  `);
}}
</script>
"""


# ---------------- Main ----------------

def main():
    st.title("Sistema de alarmas FX")

    # Side panel con info
    with st.sidebar:
        st.markdown("### Acerca")
        st.markdown(
            "Dashboard predictivo de tipo de cambio: USD vs ARS / BRL / CLP / COP / INR.\n\n"
            "Cada visita baja datos en vivo de Yahoo Finance, computa features y predice con "
            "un modelo LightGBM entrenado sobre 14 ventanas de validación walk-forward."
        )
        st.markdown("### Limitaciones")
        st.markdown(
            "- AUC ≈ 0,57 a 5 días para los flotantes. Es señal modesta, no oráculo.\n"
            "- ARS está dominado por política local; los drivers globales explican <5%.\n"
            "- Datos FRED (DFF, T10Y2Y) vienen de un snapshot del proyecto, no en vivo."
        )
        st.markdown("---")
        if st.button("🔄 Forzar recarga de datos"):
            st.cache_data.clear()
            st.rerun()

    # Help boxes
    with st.expander("**¿Cómo leer este dashboard?**", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                "**Sparkline interactivo**\n\n"
                "Pasá el mouse sobre el gráfico para ver fecha y valor de cada día. "
                "Las líneas punteadas marcan máx/mín de los últimos 60 días."
            )
        with col2:
            st.markdown(
                "**Probabilidad de extremo**\n\n"
                "Probabilidad estimada de que la moneda se mueva más que el umbral en los próximos 5 (o 20) días. "
                "Comparar con el umbral de cada moneda."
            )
        with col3:
            st.markdown(
                "**Dirección esperada**\n\n"
                "🔴 ↑ USD sube = la moneda local se debilita (mejor haber comprado dólares antes).\n"
                "🟢 ↓ USD baja = la moneda local se fortalece (mejor para vender dólares)."
            )

    with st.expander("**¿Qué mueve a cada moneda?**"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**USDARS**\n\nPolítica local domina: cepos, devaluaciones, elecciones. Drivers globales explican <5%. El peso solo se devalúa.")
            st.markdown("**USDBRL**\n\nMezcla 60-40: drivers globales (DXY, VIX) + política local. Sensible al risk-off.")
        with c2:
            st.markdown("**USDCLP**\n\nEl más limpio: cobre + DXY. Cuando el cobre cae, el peso chileno se debilita.")
            st.markdown("**USDCOP**\n\nPetróleo + risk-off. Sensible al Brent.")
        with c3:
            st.markdown("**USDINR**\n\nManaged float: el RBI suaviza shocks. Solo eventos globales muy fuertes pasan.")
            st.markdown("**Drivers globales**\n\nDXY → todos. VIX → BRL/INR. Cobre → CLP. Brent → COP/INR.")

    # Fetch data
    fx_df, drv_df = fetch_all_data()
    fred = load_snapshot_fred()

    if fx_df.empty:
        st.error("No pude bajar datos de Yahoo Finance. Probá refrescar la página en unos minutos.")
        st.stop()

    # Predictions
    preds = predict_today(fx_df, drv_df, fred)

    # Build cards data
    cards_data = []
    for ccy in FX_COLS:
        if ccy not in fx_df.columns:
            continue
        cd = card_html(ccy, fx_df[ccy], preds.get(ccy), None)
        cards_data.append(cd)

    # Drivers panel
    drivers_data = {}
    drv_show = ["DXY", "VIX", "DGS10", "BRENT", "COPPER", "GOLD"]
    for d in drv_show:
        if d in drv_df.columns and not drv_df[d].dropna().empty:
            s = drv_df[d].dropna()
            v_now = float(s.iloc[-1])
            v_5d_ago = float(s.iloc[-6]) if len(s) >= 6 else None
            chg5 = ((v_now / v_5d_ago - 1) * 100) if v_5d_ago else None
            drivers_data[d] = {"lvl": v_now, "chg_5d_pct": chg5}

    # Status banner
    last_date = fx_df.index.max().strftime("%Y-%m-%d")
    n_alarmas = sum(1 for c in cards_data if c["alarma_1d"] or c["alarma_5d"])
    if n_alarmas:
        msg = ", ".join(f"{c['ccy']}" for c in cards_data if c["alarma_1d"] or c["alarma_5d"])
        st.error(f"⚠️ Alarmas reactivas activas: {msg}")
    else:
        st.success(f"✅ Sin alarmas reactivas activas — datos al cierre del {last_date}")

    # Render dashboard HTML
    import streamlit.components.v1 as components
    html = render_html(cards_data, drivers_data, last_date)
    components.html(html, height=1400, scrolling=True)

    st.caption(
        f"Datos: Yahoo Finance (en vivo, caché 1 hora) + snapshot de FRED. "
        f"Última fecha disponible: {last_date}. "
        f"Para forzar recarga, click en 🔄 del panel lateral."
    )


if __name__ == "__main__":
    main()
