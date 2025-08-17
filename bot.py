#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Discord Signal Bot v3.3 – Today uses EOM-anchored momentum
- Postet nur an US-Handelstagen (QQQ-Check).
- 'Stand heute': ΣMOM = Summe aus (Latest / EOM(1,3,6,9M-Anchor) - 1), sortiert nach ΣMOM.
- Gate HEUTE auf diese Top-3 (Preis > 10M-SMA & 20d-Vol < 30%).
- EOM-Block unverändert (ΣMomentum 1/3/6/9M ohne Überlappung).
"""

from __future__ import annotations

import os
import requests
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timezone

import pandas as pd
import numpy as np
import yfinance as yf

# ================== Config ==================
TICKERS        = ["BTC-USD", "QQQ", "GLD", "DBO", "EEM", "FEZ", "IEF"]
START          = "2024-01-01"
SMA_DAYS       = 150
TENM_SMA_DAYS  = 210
VOL_WINDOW     = 20
VOL_THR        = 0.30
TOP_N          = 3

DEBUG           = os.getenv("DEBUG", "0") == "1"
ANNUALIZER_MODE = "uniform_252"
MIXED_ANNUALIZER: Dict[str, int] = {"BTC-USD": 365}

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
ALWAYS_SEND         = os.getenv("ALWAYS_SEND", "0") == "1"  # optionaler Override

# ================== Data helpers ==================
def _tz_naive(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_convert(None)
    return df

def last_completed_month_label(ts: Optional[pd.Timestamp] = None) -> pd.Timestamp:
    ts = (pd.Timestamp.utcnow() if ts is None else pd.Timestamp(ts)).tz_localize(None)
    return (ts - pd.offsets.MonthEnd(1)).normalize()

def fetch_unadjusted_close(tickers: List[str], start: str) -> pd.DataFrame:
    if len(tickers) == 1:
        data = yf.download(tickers[0], start=start, progress=False, auto_adjust=False)
        close = data["Close"].to_frame(name=tickers[0])
    else:
        data  = yf.download(tickers, start=start, group_by="ticker", progress=False, auto_adjust=False)
        frames = []
        for t in tickers:
            df = data[t] if isinstance(data.columns, pd.MultiIndex) else data
            frames.append(df["Close"].rename(t))
        close = pd.concat(frames, axis=1)
    close = _tz_naive(close.sort_index())
    return close.reindex(columns=tickers)

def monthly_last_ffilled(df: pd.DataFrame) -> pd.DataFrame:
    return df.ffill().resample("ME").last()

def rolling_sma_per_column(df: pd.DataFrame, window: int) -> pd.DataFrame:
    out = {}
    for col in df.columns:
        s = df[col].astype(float)
        out[col] = s.dropna().rolling(window, min_periods=window).mean().reindex(s.index)
    return pd.DataFrame(out, index=df.index)

def realized_vol_ann_from_prices_per_column(
    prices: pd.DataFrame, window: int, annualizer_mode: str, mixed_map: Dict[str, int]
) -> pd.DataFrame:
    out = {}
    for col in prices.columns:
        s = prices[col].dropna().astype(float)
        r = s.pct_change(fill_method=None)
        ann = 252 if annualizer_mode == "uniform_252" else 365 if annualizer_mode == "uniform_365" else mixed_map.get(col, 252)
        out[col] = (r.rolling(window, min_periods=window).std() * np.sqrt(ann)).reindex(prices.index)
    return pd.DataFrame(out, index=prices.index)

# ---- Momentum ohne Überlappung (1/3/6/9M kumuliert) ----
def sum_momentum_1_3_6_9m(rets_m: pd.DataFrame, as_of_eom: pd.Timestamp) -> pd.Series:
    prices_rel = (1.0 + rets_m).cumprod()
    if as_of_eom not in prices_rel.index:
        as_of_eom = prices_rel.index.max()
    pos = prices_rel.index.get_loc(as_of_eom)

    def cumret(series: pd.Series, m: int) -> float:
        if pos - m < 0: return np.nan
        return series.iloc[pos] / series.iloc[pos - m] - 1.0

    out = {}
    for col in prices_rel.columns:
        s = prices_rel[col].dropna()
        out[col] = np.nansum([cumret(s, 1), cumret(s, 3), cumret(s, 6), cumret(s, 9)])
    return pd.Series(out).sort_values(ascending=False)

def evaluate_gate(price: float, v20: float, s10m: float) -> Tuple[bool, str]:
    if any(pd.isna(x) for x in (price, v20, s10m)):
        return False, "NaN input"
    ok = (price > s10m) and (v20 < VOL_THR)
    return ok, "OK" if ok else "FAIL"

# ================== Formatting helpers ==================
def short_ticker(t: str) -> str:
    return t.replace("-USD", "")

def fmt_price(v) -> str:
    return "NaN" if pd.isna(v) else f"{v:,.2f}"

def fmt_pct(v) -> str:
    return "NaN" if pd.isna(v) else f"{v*100:.2f}%"

def table_schema(headers: List[str], rows: List[List[str]]) -> str:
    widths = [len(h) for h in headers]
    for r in rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len("" if cell is None else str(cell)))
    head_line = "  ".join(headers[i].ljust(widths[i]) for i in range(len(headers)))
    underline = "  ".join("-" * widths[i] for i in range(len(headers)))
    body = "\n".join(
        "  ".join(("" if r[i] is None else str(r[i])).ljust(widths[i]) for i in range(len(headers)))
        for r in rows
    )
    return "\n".join([head_line, underline, body]) if rows else "\n".join([head_line, underline])

def sep_line() -> str:
    return "-" * 33

# ================== Dataclasses ==================
@dataclass
class GateRow:
    ticker: str
    date: Optional[str]
    price: Optional[str]
    vol20: str
    gate: str

# ================== Trading day guard ==================
def is_us_trading_day() -> bool:
    """True nur wenn heute ein US-Handelstag ist (Mo–Fr & QQQ hat heutigen Daily-Bar)."""
    if ALWAYS_SEND:
        return True
    now_utc = datetime.now(timezone.utc)
    if now_utc.weekday() >= 5:  # 5=Sa, 6=So
        return False
    try:
        q = yf.download("QQQ", period="7d", interval="1d", auto_adjust=False, progress=False)
        if q.empty:
            return False
        last_bar_date = q.index[-1].tz_convert(None).date()
        return last_bar_date == now_utc.date()
    except Exception:
        # Fail-safe: lieber nichts posten als an Feiertagen zu spammen
        return False

# ================== Helpers for "Today" EOM-anchored momentum ==================
def eom_anchor_dates_for_today(as_of_eom: pd.Timestamp) -> Dict[int, pd.Timestamp]:
    """
    EOM-Anker relativ zum letzten abgeschlossenen Monat (as_of_eom):
      1M -> as_of_eom
      3M -> as_of_eom - MonthEnd(2)
      6M -> as_of_eom - MonthEnd(5)
      9M -> as_of_eom - MonthEnd(8)
    """
    anchors = {1: as_of_eom}
    anchors[3] = (as_of_eom - pd.offsets.MonthEnd(2)).normalize()
    anchors[6] = (as_of_eom - pd.offsets.MonthEnd(5)).normalize()
    anchors[9] = (as_of_eom - pd.offsets.MonthEnd(8)).normalize()
    return anchors

# ================== Strategy core ==================
def compute_state() -> Dict[str, object]:
    prices_d = fetch_unadjusted_close(TICKERS, start=START)
    sma150_d = rolling_sma_per_column(prices_d, SMA_DAYS)
    sma10_d  = rolling_sma_per_column(prices_d, TENM_SMA_DAYS)
    vol20_d  = realized_vol_ann_from_prices_per_column(prices_d, VOL_WINDOW, ANNUALIZER_MODE, MIXED_ANNUALIZER)

    prices_m = monthly_last_ffilled(prices_d)
    rets_m   = prices_m.pct_change(fill_method=None)
    sma150_m = rolling_sma_per_column(prices_d, SMA_DAYS).ffill().resample("ME").last()
    sma10_m  = rolling_sma_per_column(prices_d, TENM_SMA_DAYS).ffill().resample("ME").last()
    vol20_m  = realized_vol_ann_from_prices_per_column(prices_d, VOL_WINDOW, ANNUALIZER_MODE, MIXED_ANNUALIZER).ffill().resample("ME").last()

    as_of = last_completed_month_label()
    sum_mom = sum_momentum_1_3_6_9m(rets_m, as_of)  # EOM-basiert, wie gehabt (für Monatsblock)

    # ===== Monatsblock =====
    mask = prices_m.loc[as_of] > sma150_m.loc[as_of]
    top_official = [t for t in sum_mom.index if bool(mask.get(t, False))][:TOP_N]

    gate_month: List[GateRow] = []
    if top_official:
        checks = []
        for t in top_official:
            p, s10, v20 = prices_m.loc[as_of, t], sma10_m.loc[as_of, t], vol20_m.loc[as_of, t]
            ok, _ = evaluate_gate(p, v20, s10)
            checks.append(ok)
            gate_month.append(GateRow(short_ticker(t), None, fmt_price(p), fmt_pct(v20), "PASS" if ok else "FAIL"))
        allow_leverage_official = all(checks)
    else:
        allow_leverage_official = False
    lev_official = "3x" if allow_leverage_official else "1x"

    # ===== HEUTE (EOM-anchors) =====
    # Letzter Handelstag je Ticker + ΔSMA
    latest_rows = []
    for t in TICKERS:
        s = prices_d[t].dropna()
        if len(s) == 0:
            continue
        d = s.index[-1]
        latest_rows.append({"Ticker": t, "Date": d, "Close": s.iloc[-1], "SMA": sma150_d.loc[d, t]})
    latest = pd.DataFrame(latest_rows).set_index("Ticker")
    latest["ΔSMA150_%"] = (latest["Close"] / latest["SMA"] - 1.0) * 100.0

    # Kandidaten: über täglichem SMA150
    over_list = latest.index[latest["Close"] > latest["SMA"]].tolist()

    # ΣMOM_today: Latest vs EOM-Anker (1/3/6/9M), dann Summe
    anchors = eom_anchor_dates_for_today(as_of)
    today_calc = []
    for t in over_list:
        if t not in prices_m.columns:
            continue
        latest_px = latest.loc[t, "Close"]
        r_vals = []
        for n in (1, 3, 6, 9):
            eom = anchors[n]
            ref_px = prices_m.loc[eom, t] if (eom in prices_m.index) else np.nan
            r = (latest_px / ref_px - 1.0) if (pd.notna(latest_px) and pd.notna(ref_px) and ref_px != 0) else np.nan
            r_vals.append(r)
        sigma_today = float(np.nansum(r_vals))
        today_calc.append((t, sigma_today))

    # Sortierung heute NUR nach ΣMOM_today (absteigend)
    today_calc.sort(key=lambda x: x[1], reverse=True)
    ranked_today_syms = [t for t, _ in today_calc]

    # Ausgabe-Liste für „Stand heute (über SMA150)“
    today_ranking = [{
        "ticker": short_ticker(t),
        "sumom": f"{(sigma*100.0):.2f}%",
        "dsma":  f"{latest.loc[t,'ΔSMA150_%']:.2f}%"
    } for t, sigma in today_calc]

    # Gate HEUTE nur auf Top-3 dieser Sortierung
    gate_today: List[GateRow] = []
    if ranked_today_syms:
        checks = []
        for t in ranked_today_syms[:TOP_N]:
            d = latest.loc[t, "Date"]
            p, s10, v20 = prices_d.loc[d, t], sma10_d.loc[d, t], vol20_d.loc[d, t]
            ok, _ = evaluate_gate(p, v20, s10)
            checks.append(ok)
            gate_today.append(GateRow(short_ticker(t), str(pd.to_datetime(d).date()), None, fmt_pct(v20), "PASS" if ok else "FAIL"))
        allow_leverage_today = all(checks)
    else:
        allow_leverage_today = False
    lev_today = "3x" if allow_leverage_today else "1x"

    return {
        "as_of": as_of.date(),
        "top_official": [short_ticker(t) for t in top_official],
        "lev_official": lev_official,
        "gate_month": gate_month,
        "ranking_today": today_ranking,
        "lev_today": lev_today,
        "gate_today": gate_today,
    }

# ---------- Build message ----------
def build_message(state: Dict[str, object]) -> str:
    as_of = state["as_of"]
    lev_off = state["lev_official"]
    lev_today = state["lev_today"]
    gm: List[GateRow] = state["gate_month"]  # type: ignore
    rank = state["ranking_today"]            # type: ignore
    gt: List[GateRow] = state["gate_today"]  # type: ignore

    parts: List[str] = []
    parts.append(f"Letzter Monat: {as_of}")
    parts.append(sep_line())
    parts.append("Top 3")
    parts.append(sep_line())

    headers_m = ["Ticker", "Preis", "20d-Vol", "Gate"]
    rows_m = [[r.ticker, r.price or "", r.vol20, r.gate] for r in gm]
    parts.append(table_schema(headers_m, rows_m))
    parts.append("")
    parts.append(f"Leverage: {lev_off}")
    parts.append(sep_line())
    parts.append("")
    parts.append("Stand heute (über SMA150):")
    parts.append(sep_line())

    headers_r = ["Ticker", "ΣMom", "ΔSMA"]
    rows_r = [[r["ticker"], r["sumom"], r["dsma"]] for r in rank]
    parts.append(table_schema(headers_r, rows_r))
    parts.append("")
    parts.append(sep_line())
    parts.append("Top-3")
    parts.append(sep_line())

    headers_t = ["Ticker", "Date", "20d-Vol", "Gate"]
    rows_t = [[r.ticker, r.date or "", r.vol20, r.gate] for r in gt]
    parts.append(table_schema(headers_t, rows_t))
    parts.append("")
    parts.append(f"Leverage: {lev_today}")

    return "\n".join(parts).rstrip() + "\n"

# ================== Discord ==================
def send_codeblock(content: str, webhook_url: str):
    if not webhook_url:
        raise ValueError("DISCORD_WEBHOOK_URL ist nicht gesetzt.")
    payload = {"content": f"```{content}```"}
    r = requests.post(webhook_url, json=payload, timeout=30)
    r.raise_for_status()

# ================== Main ==================
def main():
    # ---- Guard: nur an US-Handelstagen senden ----
    if not is_us_trading_day():
        print("Non-trading day in US markets – skipping Discord post.")
        return

    state = compute_state()
    text = build_message(state)
    send_codeblock(text, DISCORD_WEBHOOK_URL)

if __name__ == "__main__":
    main()
