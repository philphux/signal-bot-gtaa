# GTAA Momentum Discord Bot

A Python bot that posts **Global Tactical Asset Allocation (GTAA)** signals into a Discord channel.  
The strategy combines multi-horizon momentum (1M, 3M, 6M, 9M), a long-term trend filter (SMA150), and a volatility gate — both in **monthly mode (EOM)** and as a **daily update (“Today”) using EOM anchors**.

---

## 🚀 Features

- **Monthly signals (EOM-based):**
  - Ranking via **ΣMOM** = sum of 1M/3M/6M/9M momentum (non-overlapping).
  - **SMA150 filter** at month-end (only assets above SMA150 are eligible).
  - Select the **Top 3** assets.
  - **Gate (monthly):** Price > 10-month SMA and annualized 20-day volatility < 30%.
  - **Leverage:** **3×** if all three pass, otherwise **1×**.

- **Daily update (“Today”):**
  - Uses **EOM anchors**: latest daily close vs. last EOM for 1M/3M/6M/9M → **ΣMOM_today**.
  - Considers only assets **above daily SMA150**.
  - Shows ranking by **ΣMOM_today** and **ΔSMA** (distance to SMA150).
  - **Gate (today):** applied to the **Top 3** of today’s ranking with daily 10M-SMA and 20d-vol.  
  - **Leverage:** **3×** if all three pass, otherwise **1×**.

- **US trading day filter (new & robust):**
  - Posts **only on official NYSE trading days** via `exchange-calendars` (calendar **XNYS**).
  - **Fallback:** If the calendar is unavailable, checks **QQQ intraday (1m)** to verify activity today (US/Eastern).
  - **Override:** `ALWAYS_SEND=1` forces posting (e.g. for testing).

- **Discord output:**  
  - Clean, monospaced **code blocks** with fixed-width tables.

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/gtaa-discord-bot.git
cd gtaa-discord-bot
pip install -r requirements.txt
```

### Requirements (`requirements.txt`)
- `pandas`
- `numpy`
- `yfinance`
- `requests`
- **new:** `exchange-calendars>=4.5`

> Note: The bot requires **Python ≥ 3.9** (for `zoneinfo`).

---

## ⚙️ Configuration

Set the following environment variables:

- `DISCORD_WEBHOOK_URL` – your Discord webhook URL (**required**).
- `DEBUG` – `1` for verbose logging (optional).
- `ALWAYS_SEND` – `1` to override the trading-day check (optional).

**Example:**
```bash
export DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/..."
export DEBUG=1
```

---

## ▶️ Usage

**Run manually:**
```bash
python bot.py
```

**Cron (local/server):**
```bash
0 12 * * * /usr/bin/python3 /path/to/bot.py
```

**GitHub Actions (recommended):**  
If running as a GitHub Action, schedule **after US market close** (e.g. ~22:10 UTC ≈ 18:10 ET), to ensure stable EOD data.  
Store secrets/vars (Webhook, DEBUG/ALWAYS_SEND) under **Settings → Secrets and variables**.

Minimal `schedule` example:
```yaml
on:
  schedule:
    - cron: "10 22 * * 1-5"  # 22:10 UTC, Mon–Fri
```

---

## 📊 Default Assets

- **Equities/ETFs:** `QQQ`, `EEM`, `FEZ`  
- **Commodities:** `GLD`, `DBO`  
- **Bonds:** `IEF`  
- **Crypto:** `BTC-USD`

> Adjust in `bot.py` (`TICKERS`) if needed.

---

## 📖 Strategy Details

- Momentum horizons use **fixed periods** (1M/3M/6M/9M).  
- Monthly logic and filters at **end-of-month**; daily logic uses **latest vs. EOM anchors**.  
- Gate thresholds: **Price > 10M-SMA** and **20d-vol < 30%**.  
- **Leverage rule:** Top 3 **all PASS → 3×**, otherwise **1×**.

---

## ⚠️ Disclaimer

This project is for **educational purposes only** and does **not constitute investment advice**. Use at your own risk.
