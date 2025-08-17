# GTAA Momentum Discord Bot

A Python bot that posts **Global Tactical Asset Allocation (GTAA)** momentum signals to a Discord channel.  
The strategy combines multi-horizon momentum (1M, 3M, 6M, 9M), a long-term trend filter (SMA150), and a volatility-based gate.

---

## 🚀 Features

- **Monthly signals (EOM-based):**
  - Rank assets by **ΣMOM** = sum of 1M, 3M, 6M, 9M end-of-month momentum (non-overlapping).
  - **SMA150 filter** at month end (only assets above SMA150 are eligible).
  - Select the **Top-3** assets.
  - **Gate (monthly):** Price > 10-month SMA and 20-day annualized volatility < 30%.
  - Leverage: **3x** if all three pass, otherwise **1x**.

- **Daily update (“Today”):**
  - Uses **EOM anchors** for momentum: Latest daily close vs. last EOM for 1M/3M/6M/9M; ΣMOM_today = sum of those four.
  - Considers only assets **above daily SMA150** (each on its own most recent trading day).
  - Sorts by **ΣMOM_today** and shows **ΔSMA** (distance to SMA150).
  - **Gate (today):** applied to the **Top-3** from today’s ranking using daily 10M-SMA and 20d-vol at each asset’s latest date.
  - Leverage: **3x** if all three pass, otherwise **1x**.

- **US trading day filter:**
  - Posts **only on US trading days** (checks QQQ for a daily bar today).
  - Prevents weekend/holiday posts (BTC trades daily, but equities do not).

- **Discord output:**
  - Clean, fixed-width tables posted as code blocks.

---

## 📦 Installation

Clone the repository and install requirements:

~~~bash
git clone https://github.com/yourusername/gtaa-discord-bot.git
cd gtaa-discord-bot
pip install -r requirements.txt
~~~

**Dependencies**
- `pandas`
- `numpy`
- `yfinance`
- `requests`

---

## ⚙️ Configuration

Set the following environment variables before running:

- `DISCORD_WEBHOOK_URL` – your Discord webhook URL (**required**).
- `DEBUG` – set `1` to enable debug logging (optional).
- `ALWAYS_SEND` – set `1` to force messages even on weekends/holidays (optional).

**Example:**

~~~bash
export DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/..."
export DEBUG=1
~~~

---

## ▶️ Usage

Run manually:

~~~bash
python bot.py
~~~

Or schedule daily execution with `cron`:

~~~bash
0 12 * * * /usr/bin/python3 /path/to/bot.py
~~~

> On GitHub Actions, set the environment variables as **Repository Secrets/Variables**.

---

## 📊 Assets

Default tickers (adjust in `bot.py` if needed):

- **Equities/ETFs:** `QQQ`, `EEM`, `FEZ`  
- **Commodities:** `GLD`, `DBO`  
- **Bonds:** `IEF`  
- **Crypto:** `BTC-USD`

---

## 📖 Strategy Notes

- Momentum horizons are measured as price changes over **fixed periods** (1M, 3M, 6M, 9M).  
- Monthly momentum and filters use **end-of-month** prices; today’s momentum uses **latest vs. EOM anchors**.  
- Gate thresholds: **Price > 10M-SMA** and **20d-vol < 30%**.  
- Final leverage is **Top-3 all PASS → 3x**, otherwise **1x**.

---

## ⚠️ Disclaimer

This project is for **educational purposes only**.  
It is **not financial advice**. Use at your own risk.
