#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Discord Signal Bot – basiert auf ULT-GTAA v2.5 (Pretty View only)
- Läuft lokal oder per GitHub Actions.
- Holt Marktdaten (yfinance), erzeugt die Pretty-View-Tabellen
  und postet sie als Codeblock(e) in einen Discord-Channel via Webhook.
- DEBUG-Ausgabe per ENV steuerbar: DEBUG=1 zeigt in Gate-Tabellen zusätzlich 10M-SMA, obs20, Reason.
"""

from __future__ import annotations

import os
import time
from io import StringIO
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Dict, Tuple

# ================== Config (Strategie & Ausgabe) ==================
TICKERS        = ["BTC-USD", "QQQ", "GLD", "USO", "EEM", "FEZ", "IEF"]
START          = "2024-01-01"
SMA_DAYS       = 150
TENM_SMA_DAYS  = 210           # ~10 Monate à 21 Handelstage
VOL_WINDOW     = 20
VOL_THR        = 0.30          # 30% annualisiert
TOP_N          = 3

# Pretty View: immer an
# DEBUG via ENV (falls nicht gesetzt: False)
DEBUG          = os.getenv("DEBUG", "0") == "1"

# Annualizer für Vol: "uniform_252", "uniform_365" oder "mixed"
ANNUALIZER_MODE = "uniform_252"
MIXED_ANNUALIZER: Dict[str, int] = {"BTC-USD": 365}  # nur bei "mixed" aktiv

# Discord Webhook aus ENV
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "").strip()

# ================== Data Helpers ==================
def _tz_naive(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_convert(None)
    return df

def last_completed_month_label(ts: pd.Timestamp | None = None) -> pd.Timestamp:
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
        if annualizer_mode == "uniform_252":
            ann = 252
        elif annualizer_mode == "uniform_365":
            ann = 365
        else:
            ann = mixed_map.get(col, 252)
        out[col] = (r.rolling(window, min_periods=window).std() * np.sqrt(ann)).reindex(prices.index)
    return pd.DataFrame(out, index=prices.index)

def sum_momentum_1_3_6_9m(rets_m: pd.DataFrame, as_of_eom: pd.Timestamp) -> pd.Series:
    if as_of_eom not in rets_m.index:
        as_of_eom = rets_m.index.dropna().max()
    pos = rets_m.index.get_loc(as_of_eom)
    if pos < 8:
        take = [1, min(3, pos+1), min(6, pos+1), min(9, pos+1)]
        sums = {}
        for col in rets_m.columns:
            s = rets_m[col].iloc[:pos+1].dropna()
            parts = [
                s.iloc[-take[0]:].sum() if len(s) >= take[0] else np.nan,
                s.iloc[-take[1]:].sum() if len(s) >= take[1] else np.nan,
                s.iloc[-take[2]:].sum() if len(s) >= take[2] else np.nan,
                s.iloc[-take[3]:].sum() if len(s) >= take[3] else np.nan,
            ]
            sums[col] = np.nansum(parts)
        return pd.Series(sums).sort_values(ascending=False)
    sums = {}
    for col in rets_m.columns:
        s = rets_m[col]
        sums[col] = np.nansum([
            s.iloc[pos-0:pos+1].sum(),
            s.iloc[pos-2:pos+1].sum(),
            s.iloc[pos-5:pos+1].sum(),
            s.iloc[pos-8:pos+1].sum(),
        ])
    return pd.Series(sums).sort_values(ascending=False)

def compute_obs20(prices: pd.Series, d: pd.Timestamp, window: int = VOL_WINDOW) -> int:
    s = prices.dropna().loc[:d]
    r = s.pct_change(fill_method=None).dropna()
    return int(r.iloc[-window:].shape[0])

def evaluate_gate(price: float, v20: float, s10m: float) -> Tuple[bool, str]:
    if any(pd.isna(x) for x in (price, v20, s10m)):
        return False, "NaN input"
    reasons = []
    if not (price > s10m):
        reasons.append("Preis<=10M-SMA")
    if not (v20 < VOL_THR):
        reasons.append(f"Vol>={VOL_THR:.0%}")
    ok = (len(reasons) == 0)
    return ok, ("OK" if ok else ";".join(reasons))

# ================== Formatting (Pretty View) ==================
def short_ticker(t: str) -> str:
    return t.replace("-USD", "")  # Anzeige-only (BTC-USD -> BTC)

def fmt_price(v) -> str:
    return "NaN" if pd.isna(v) else f"{v:,.2f}"

def fmt_pct(v) -> str:
    return "NaN" if pd.isna(v) else f"{v*100:.2f}%"

def make_pretty_table(headers: List[str], rows: List[List[str]], min_widths: Dict[int, int] | None = None) -> str:
    """
    Links­bündige, bündige ASCII-Tabelle (Spaces). Spaltenbreite = max(Header, Zellen, Mindestbreite).
    """
    min_widths = min_widths or {}
    cols = len(headers)
    widths = [len(str(headers[i])) for i in range(cols)]
    for r in rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(str(cell)))
    for i, wmin in min_widths.items():
        if i < cols:
            widths[i] = max(widths[i], wmin)
    head = "  ".join(str(headers[i]).ljust(widths[i]) for i in range(cols))
    sep  = "  ".join("-" * widths[i] for i in range(cols))
    body = "\n".join("  ".join(str(r[i]).ljust(widths[i]) for i in range(cols)) for r in rows)
    return "\n".join([head, sep, body]) if rows else "\n".join([head, sep])

# ================== Core (Strategie) ==================
def run_strategy(debug: bool = DEBUG) -> str:
    """Erzeugt den kompletten Pretty-View-Output als String."""
    # Tages-/Monatsreihen
    prices_d = fetch_unadjusted_close(TICKERS, start=START)
    sma150_d = rolling_sma_per_column(prices_d, SMA_DAYS)
    sma10_d  = rolling_sma_per_column(prices_d, TENM_SMA_DAYS)
    vol20_d  = realized_vol_ann_from_prices_per_column(prices_d, VOL_WINDOW, ANNUALIZER_MODE, MIXED_ANNUALIZER)

    prices_m = monthly_last_ffilled(prices_d)
    rets_m   = prices_m.pct_change(fill_method=None)
    sma150_m = rolling_sma_per_column(prices_d, SMA_DAYS).ffill().resample("ME").last()
    sma10_m  = rolling_sma_per_column(prices_d, TENM_SMA_DAYS).ffill().resample("ME").last()
    vol20_m  = realized_vol_ann_from_prices_per_column(prices_d, VOL_WINDOW, ANNUALIZER_MODE, MIXED_ANNUALIZER).ffill().resample("ME").last()

    # ===== Monatsblock =====
    as_of = last_completed_month_label()
    sum_mom = sum_momentum_1_3_6_9m(rets_m, as_of)
    mask = prices_m.loc[as_of] > sma150_m.loc[as_of]
    top_official = [t for t in sum_mom.index if bool(mask.get(t, False))][:TOP_N]

    allow_leverage_official = False
    gate_rows_official: List[List[str]] = []
    if top_official:
        checks = []
        for t in top_official:
            p, s10, v20 = prices_m.loc[as_of, t], sma10_m.loc[as_of, t], vol20_m.loc[as_of, t]
            ok, reason = evaluate_gate(p, v20, s10)
            checks.append(ok)
            d_m = prices_d.index[prices_d.index <= as_of].max()
            obs20 = compute_obs20(prices_d[t], d_m, VOL_WINDOW)
            if debug:
                gate_rows_official.append([short_ticker(t), fmt_price(p), fmt_pct(v20),
                                           "PASS" if ok else "FAIL", fmt_price(s10), str(obs20), reason])
            else:
                gate_rows_official.append([short_ticker(t), fmt_price(p), fmt_pct(v20), "PASS" if ok else "FAIL"])
        allow_leverage_official = all(checks)
    lev_official = "3x" if allow_leverage_official else "1x"

    out = []
    out.append("=== Letzter abgeschlossener Monat ===")
    out.append(f"As-of: {as_of.date()}")
    out.append("")
    out.append("Top-3:")
    if top_official:
        w = f"{100.0/len(top_official):.1f}%"
        rows_top = [[short_ticker(t), w] for t in top_official]
        out.append(make_pretty_table(["Ticker","Gewicht"], rows_top, {0:5, 1:7}))
    else:
        out.append("—")
    out.append("")
    out.append(f"Leverage-Empfehlung: {lev_official}")
    out.append("")

    headers_a = ["Ticker","Preis","20d-Vol","Gate"] if not debug else ["Ticker","Preis","20d-Vol","Gate","10M-SMA","obs20","Reason"]
    out.append(make_pretty_table(headers_a, gate_rows_official, {0:5,1:10,2:7,3:4}))
    if gate_rows_official:
        passed = sum(1 for r in gate_rows_official if r[3] == "PASS")
        out.append(f"\nGate-Summary: {passed}/{len(gate_rows_official)} PASS")
    out.append("")

    # ===== Heute =====
    latest_rows = []
    for t in TICKERS:
        s = prices_d[t].dropna()
        if len(s) == 0:
            continue
        d = s.index[-1]
        latest_rows.append({"Ticker": t, "Date": d, "Close": s.iloc[-1], "SMA": sma150_d.loc[d, t]})
    latest = pd.DataFrame(latest_rows).set_index("Ticker")
    latest["ΔSMA150_%"] = (latest["Close"] / latest["SMA"] - 1.0) * 100.0
    over_list = latest.index[latest["Close"] > latest["SMA"]].tolist()
    ranked_today = [t for t in sum_mom.index if t in over_list]

    out.append("=== Stand heute ===")
    out.append("")
    if len(ranked_today) == 0:
        out.append("Über SMA150: —")
        out.append("")
    else:
        rows_rank = []
        for t in ranked_today:
            sum_pct = sum_mom.loc[t] * 100.0 if t in sum_mom.index else float("nan")
            d_pct   = latest.loc[t, "ΔSMA150_%"]
            rows_rank.append([short_ticker(t), f"{sum_pct:.2f}%", f"{d_pct:.2f}%"])
        out.append(make_pretty_table(["Ticker","ΣMom","ΔSMA"], rows_rank, {0:5,1:7,2:7}))
        out.append("")

    # Gate heute (nur Top-N)
    allow_leverage_today = False
    gate_rows_today: List[List[str]] = []
    if ranked_today:
        checks = []
        for t in ranked_today[:TOP_N]:
            d = latest.loc[t, "Date"]
            p, s10, v20 = prices_d.loc[d, t], sma10_d.loc[d, t], vol20_d.loc[d, t]
            ok, reason = evaluate_gate(p, v20, s10)
            checks.append(ok)
            obs20 = compute_obs20(prices_d[t], d, VOL_WINDOW)
            if debug:
                gate_rows_today.append([short_ticker(t), str(pd.to_datetime(d).date()), fmt_price(p), fmt_pct(v20),
                                        "PASS" if ok else "FAIL", fmt_price(s10), str(obs20), reason])
            else:
                gate_rows_today.append([short_ticker(t), str(pd.to_datetime(d).date()), fmt_price(p), fmt_pct(v20),
                                        "PASS" if ok else "FAIL"])
        allow_leverage_today = all(checks)
    lev_today = "3x" if allow_leverage_today else "1x"

    out.append(f"Leverage-Empfehlung: {lev_today}")
    out.append("")
    headers_b = ["Ticker","Date","Preis","20d-Vol","Gate"] if not debug else ["Ticker","Date","Preis","20d-Vol","Gate","10M-SMA","obs20","Reason"]
    out.append(make_pretty_table(headers_b, gate_rows_today, {0:5,1:10,2:10,3:7,4:4}))
    if gate_rows_today:
        passed = sum(1 for r in gate_rows_today if r[4] == "PASS")
        out.append(f"\nGate-Summary: {passed}/{len(gate_rows_today)} PASS")

    return "\n".join(out).rstrip() + "\n"

# ================== Discord ==================
def send_to_discord(content: str, webhook_url: str, pause_sec: float = 0.8):
    """
    Schickt den Text als Codeblock an Discord.
    Discord hat ein 2000-Zeichen-Limit pro Nachricht -> in Blöcke splitten.
    """
    if not webhook_url:
        raise ValueError("DISCORD_WEBHOOK_URL ist nicht gesetzt.")
    max_len = 1900  # Sicherheitsmarge (wegen ``` und evtl. Prefix)
    chunks = []
    buf = ""
    for line in content.splitlines(True):  # True: behalte \n
        if len(buf) + len(line) > max_len:
            chunks.append(buf)
            buf = ""
        buf += line
    if buf:
        chunks.append(buf)

    for idx, chunk in enumerate(chunks, 1):
        payload = {"content": f"```{chunk}```"}
        r = requests.post(webhook_url, json=payload, timeout=30)
        try:
            r.raise_for_status()
        except requests.HTTPError as e:
            # Fehlertext mit ausgeben
            raise RuntimeError(f"Discord-Webhook fehlgeschlagen (Teil {idx}/{len(chunks)}): {r.text}") from e
        # leicht rate-limiten
        if idx < len(chunks):
            time.sleep(pause_sec)

# ================== Main ==================
def main():
    out = run_strategy(debug=DEBUG)
    send_to_discord(out, DISCORD_WEBHOOK_URL)

if __name__ == "__main__":
    main()
