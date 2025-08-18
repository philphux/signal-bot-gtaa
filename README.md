# GTAA Momentum Discord Bot

Ein Python-Bot, der **Global Tactical Asset Allocation (GTAA)**-Signale in einen Discord-Channel postet.  
Die Strategie kombiniert Multi-Horizont-Momentum (1M, 3M, 6M, 9M), einen Langfrist-Trendfilter (SMA150) und ein Volatilitäts-Gate — sowohl im **Monatsmodus (EOM)** als auch als **tägliches Update („Today“) auf EOM-Ankern**.

---

## 🚀 Features

- **Monatliche Signale (EOM-basiert):**
  - Ranking per **ΣMOM** = Summe der 1M/3M/6M/9M-Momenten (nicht überlappend).
  - **SMA150-Filter** zum Monatsultimo (nur Assets > SMA150 sind zulässig).
  - Auswahl der **Top-3**.
  - **Gate (monatlich):** Preis > 10-Monats-SMA und annualisierte 20-Tage-Volatilität < 30 %.
  - **Leverage:** **3×**, wenn alle drei **PASS**, sonst **1×**.

- **Tägliches Update („Today“):**
  - Nutzt **EOM-Anker**: letzter Tages-Close vs. letzter EOM für 1M/3M/6M/9M → **ΣMOM_today**.
  - Berücksichtigt nur Assets **über täglichem SMA150**.
  - Zeigt Ranking nach **ΣMOM_today** sowie **ΔSMA** (Abstand zum SMA150).
  - **Gate (heute):** auf die **Top-3** des heutigen Rankings mit täglichem 10M-SMA und 20d-Vol.  
  - **Leverage:** **3×**, wenn alle drei **PASS**, sonst **1×**.

- **US-Handelstags-Filter (neu, robust):**
  - Postet **nur an offiziellen NYSE-Handelstagen** via `exchange-calendars` (Kalender **XNYS**).
  - **Fallback:** Wenn der Kalender nicht verfügbar ist, wird per **QQQ-Intraday (1m)** geprüft, ob heute (US/Eastern) Marktaktivität vorliegt.
  - **Override:** `ALWAYS_SEND=1` erzwingt Posts (z. B. für Tests).

- **Discord-Ausgabe:**  
  - Saubere, monospaced **Code-Blocks** mit fixbreiten Tabellen.

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
- **neu:** `exchange-calendars>=4.5`

> Hinweis: Die Bot-Datei erwartet **Python ≥ 3.9** (wegen `zoneinfo`).

---

## ⚙️ Konfiguration

Setze folgende Umgebungsvariablen:

- `DISCORD_WEBHOOK_URL` – deine Discord Webhook-URL (**erforderlich**).
- `DEBUG` – `1` für ausführliches Logging (optional).
- `ALWAYS_SEND` – `1`, um den Handelstags-Check zu überschreiben (optional).

**Beispiel:**
```bash
export DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/..."
export DEBUG=1
```

---

## ▶️ Nutzung

**Manuell starten:**
```bash
python bot.py
```

**Cron (lokal/Server):**
```bash
0 12 * * * /usr/bin/python3 /path/to/bot.py
```

**GitHub Actions (empfohlen):**  
Wenn der Bot via Action läuft, wähle eine Uhrzeit **nach US-Börsenschluss** (z. B. ~22:10 UTC ≈ 18:10 ET), damit Tagesdaten stabil sind.  
Hinterlege Secrets/Vars (Webhook, DEBUG/ALWAYS_SEND) unter **Settings → Secrets and variables**.

Minimaler `schedule`-Beispielauszug:
```yaml
on:
  schedule:
    - cron: "10 22 * * 1-5"  # 22:10 UTC, Mo–Fr
```

---

## 📊 Standard-Assets

- **Equities/ETFs:** `QQQ`, `EEM`, `FEZ`  
- **Commodities:** `GLD`, `DBO`  
- **Bonds:** `IEF`  
- **Crypto:** `BTC-USD`

> Passe die Liste bei Bedarf in `bot.py` an (`TICKERS`).

---

## 📖 Strategie-Details

- Momentum-Horizonte auf **fixe Perioden** (1M/3M/6M/9M).  
- Monatslogik und Filter auf **End-of-Month**; Tageslogik nutzt **Latest vs. EOM-Anker**.  
- Gate-Schwellen: **Preis > 10M-SMA** und **20d-Vol < 30 %**.  
- **Leverage-Regel:** Top-3 **alle PASS → 3×**, sonst **1×**.

---

## ⚠️ Disclaimer

Dieses Projekt dient **rein zu Bildungszwecken** und stellt **keine Anlageberatung** dar. Nutzung auf eigenes Risiko.
