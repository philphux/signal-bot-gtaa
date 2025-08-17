# GTAA Momentum Discord Bot

A Python bot that posts **Global Tactical Asset Allocation (GTAA)** momentum signals to a Discord channel.  
The strategy is based on multi-horizon momentum (1M, 3M, 6M, 9M), a long-term trend filter (SMA150), and volatility-based gating.

---

## 🚀 Features

- **Monthly signals** (based on end-of-month prices):
  - Rank assets by ΣMOM = sum of 1M, 3M, 6M, and 9M momentum.
  - Apply a **SMA150 filter** (only assets above SMA150 are considered).
  - Select the **Top-3 assets**.
  - Gate check: price above 10M SMA and 20-day annualized volatility < 30%.
  - If all three assets pass → use **3x leverage**, otherwise **1x**.

- **Daily update ("Today")**:
  - Uses **end-of-month anchors** for momentum calculations (latest vs. last EOM for 1M, 3M, 6M, 9M).
  - Only considers assets above SMA150.
  - Sorts by ΣMOM_today and displays ΔSMA.
  - Gate check on the **Top-3** symbols.
  - Final leverage decision (3x or 1x).

- **US trading day filter**:
  - Posts only on US trading days (checks QQQ daily bar).
  - Prevents unnecessary weekend/holiday posts (BTC trades daily but equities do not).

- **Discord integration**:
  - Nicely formatted tables are posted as code blocks.
  - Output example:

    ```
    Last Month: 2025-07-31
    ---------------------------------
    Top 3
    ---------------------------------
    Ticker  Price       20d-Vol  Gate
    ------  ----------  -------  ----
    BTC     115,758.20  16.45%   PASS
    GLD     302.96      13.21%   PASS
    FEZ     57.87       17.22%   PASS

    Leverage: 3x
    ---------------------------------

    Today (above SMA150):
    ---------------------------------
    Ticker  ΣMom     ΔSMA
    ------  -------  ------
    BTC     74.23%   15.83%
    QQQ     40.35%   12.59%
    GLD     45.34%   5.86%

    ---------------------------------
    Top-3
    ---------------------------------
    Ticker  Date        20d-Vol  Gate
    ------  ----------  -------  ----
    BTC     2025-08-14  26.14%   PASS
    QQQ     2025-08-14  15.37%   PASS
    GLD     2025-08-14  17.84%   PASS

    Leverage: 3x
    ```

---

## 📦 Installation

Clone the repository and install requirements:

```bash
git clone https://github.com/yourusername/gtaa-discord-bot.git
cd gtaa-discord-bot
pip install -r requirements.txt
