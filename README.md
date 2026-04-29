# Advanced-Business-Analytics
This repository contains the final project for the Advanced Business Analytics course, focusing on the theme "Intelligent Methods for Resilience." Our project addresses resilience in the Danish electricity system by developing a consumer-oriented electricity price forecasting and recommendation system for the DK1 price area (Western Denmark).

## Project Objective

We design and evaluate a data-driven forecasting and decision-support system that:

- **Forecasts electricity prices** up to 120 hours ahead for the DK1 price area.
- **Communicates uncertainty** in price predictions leveraging residual variance from the developed model training
- **Generates consumer-friendly recommendations** such as:
  - *"If you can wait 3 days to charge your EV, the electricity prices are likely to drop significantly"*
  - *"The next 48 hours are expected to be unusually expensive. It is recommended to preload by doing laundry, etc. sooner rather than later"*

By translating raw price forecasts into recommendations, the system helps consumers make better consumption decisions, indirectly balancing supply and demand and hopefully reducing the amount of hours the grid experiences high stress loads.

## Repository Structure

```
Advanced-Business-Analytics/
├── .claude/
├── .devcontainer/
├── .streamlit/
├── requirements.txt               # Python dependencies
├── data/                          # Local data (CSV files, parquet, etc. Some git-ignored)
├── src/
│   ├── data/
│   │   ├──fetch_consumption_data.py          # Data acquisition (API / CSV)
│   │   ├── fetch_consumption_data.py          # Data acquisition (API / CSV)
│   │   ├── fetch_day_ahead_price_data.py          # Data acquisition (API / CSV)
│   │   ├── fetch_weather_actuals_data.py          # Data acquisition (API / CSV)
│   │   ├── fetch_weather_forecast_data.py          # Data acquisition (API / CSV)
│   │
│   ├── analysis/ # This folder contains the scripts with all necessary functions for the project_handin.ipynb
│   │   ├── energy_congestion_analysis.py     # Looks at grid stress using consumption data
│   │   ├── forecast_analysis.py              # Looks at forecast energy price predictions
│   │   ├── impact_analysis.py                # Looks at the impact of recommendations via. simulation results
│   │
├── gitignore
├── project_handin.ipynb           # Technical report. Contains markdown and compressed code blocks drawing on multiple python files. It is technical but self-explanatory. It is divided into project relevant sections. It has an introduction and a conclusion. 
├── gitignore
├── gitignore
├── gitignore
├── gitignore
```

## Approach

### Data Acquisition & Processing
Hourly DK1 spot prices are fetched from the **Energi Data Service API**
(`https://api.energidataservice.dk/dataset/Elspotprices`).
Data can also be loaded from a pre-downloaded CSV file.

### Feature Engineering
Each hourly observation is enriched with:
- **Calendar features**: hour, day-of-week, month (raw + cyclical sine/cosine encoding), Danish public holidays.
- **Lagged prices**: 1 h, 2 h, 3 h, 6 h, 12 h, 24 h, 48 h, 72 h, 168 h look-back.
- **Rolling statistics**: 6 h / 12 h / 24 h / 48 h / 168 h mean, std, min, max.
- **Trend features**: 1 h / 24 h price difference, 24 h / 168 h percentage change.

### Forecasting Models
**LightGBM with quantile regression** is used to produce three simultaneous outputs per horizon:

| Column  | Meaning                                              |
|---------|------------------------------------------------------|
| `q0.10` | Lower bound — prices are unlikely to fall below this |
| `q0.50` | Median forecast (point estimate)                     |
| `q0.90` | Upper bound — prices are unlikely to exceed this     |

A separate model is trained for each combination of `(horizon, quantile)`, covering 13 horizons from 1 h to 168 h (7 days).

### Decision Support Layer
The recommendation engine classifies each forecast hour as **cheap**, **normal**, or **expensive** relative to the recent price distribution and generates plain-language guidance:
- "Best saving opportunity" message when waiting could save ≥ 50 DKK/MWh.
- Warning messages for upcoming expensive windows.
- Encouragement messages for cheap near-term windows.

### Evaluation
Models are evaluated on a held-out 15 % test split using:
- Point-forecast metrics: MAE, RMSE, MAPE, sMAPE.
- Probabilistic metrics: pinball / quantile loss, prediction interval coverage, interval width.

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline (fetch API → train → evaluate → recommend)

#import datasets from Energinet: from main folder run:

```bash
python src/data/fetch_day_ahead_price_data.py --start 2021-01-01 --end 2026-02-02 --price-area DK1 --csv data/day_ahead_prices_dk1_raw.csv

python src/data/fetch_supply_forecast_data.py --start 2021-01-01 --end 2026-02-02 --price-area DK1 --csv data/supply_forecasts_dk1_raw.csv

python src/data/fetch_consumption_data.py --start 2021-01-01 --end 2026-02-02 --price-area DK1 --csv data/consumption_dk1_raw.csv

```

```bash
python main.py --mode full --start 2021-01-01
```

### 3. Train from a pre-downloaded CSV

```bash
python main.py --mode train --csv data/spot_prices_DK1.csv
```

### 4. Generate recommendations from a saved model

```bash
python main.py --mode predict --model models/forecaster.joblib --csv data/spot_prices_DK1.csv
```

### 5. Run tests

```bash
python -m pytest tests/ -v
```

## CLI Reference

| Argument  | Description                          | Default             |
|-----------|--------------------------------------|---------------------|
| `--mode`  | `train`, `predict`, or `full`        | `full`              |
| `--csv`   | Path to pre-downloaded CSV file      | —                   |
| `--start` | API fetch start date (`YYYY-MM-DD`)  | 3 years ago         |
| `--end`   | API fetch end date (`YYYY-MM-DD`)    | today               |
| `--model` | Path to save/load the model          | `models/forecaster.joblib` |

## Impact

By extending electricity price visibility from 24 hours to 7 days, this project demonstrates how machine learning and probabilistic forecasting can support smarter consumer behaviour and improved grid resilience. When many users shift consumption away from predicted high-price periods, the electricity system becomes more stable, efficient, and sustainable.
