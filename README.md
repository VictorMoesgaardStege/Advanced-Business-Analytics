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
├── .claude/                       # Claude configuration files
├── .devcontainer/                 # Development container configuration
├── .streamlit/                    # Streamlit app configuration
├── data/                          # Local datasets (CSV, Parquet, etc.; some files are git-ignored)
├── src/
│   ├── analysis/                  # Analysis scripts used by project_handin.ipynb
│   │   ├── energy_congestion_analysis.py  # Analyzes grid stress using electricity consumption data
│   │   ├── forecast_analysis.py           # Evaluates electricity price forecast performance
│   │   └── impact_analysis.py             # Assesses recommendation impact using simulation results
│   ├── data/                      # Data acquisition scripts and processing
│   │   ├── data_collection.py             # Collects all API fetching for all raw data sources
│   │   ├── data_processing.py             # Builds dataset for XG boost modelling
│   │   ├── fetch_consumption_data.py      # Fetches electricity consumption data
│   │   ├── fetch_day_ahead_price_data.py  # Fetches day-ahead electricity price data
│   │   ├── fetch_weather_actuals_data.py  # Fetches historical weather observations
│   │   └── fetch_weather_forecast_data.py # Fetches weather forecast data
│   └── models/                    # Forecasting and simulation models
│       ├── forecast_model.py      # Builds XGBoost forecasts and uncertainty estimates
│       └── simulation_model.py    # Runs heuristic simulations of recommendation impact
├── .gitignore                     # Files and folders excluded from version control
├── dashboard.py                   # Streamlit dashboard for price forecasting and recommendations
├── project_handin.ipynb           # Main technical report with narrative, analysis, and code outputs
└── requirements.txt               # Python dependencies
```

## Approach

### Data Acquisition & Processing

Fetched from the **Energi Data Service API**
(`https://api.energidataservice.dk/dataset/Elspotprices`).

Fetched from **Open-Meteo Historical Weather API**
(`https://archive-api.open-meteo.com/v1/archive`)

Fetched from **Open-Meteo Previous Runs API**
(`https://previous-runs-api.open-meteo.com/v1/forecast`)



### Data Processing
//TO DO

### Forecasting Model

//TO DO

### Decision Support Layer

//TO DO
The recommendation engine classifies each forecast hour as **cheap**, **normal**, or **expensive** relative to the recent price distribution and generates plain-language guidance:
- "Best saving opportunity" message when waiting could save ≥ 50 DKK/MWh.
- Warning messages for upcoming expensive windows.
- Encouragement messages for cheap near-term windows.

### Evaluation
Models are evaluated on a held-out 15 % test split using:
- Point-forecast metrics: MAE, RMSE, MAPE, sMAPE.
- Probabilistic metrics: pinball / quantile loss, prediction interval coverage, interval width.

### Impact Simulation Model
//TO DO

## Quick Start


python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
brew install libomp


//TO DO

### 1. Install dependencies

```bash
pip install -r requirements.txt
```


###
python3 -m pip install -r requirements.txt
python3 -m pip install lime
python3 src/data/data_processing.py
python3 src/models/XG_Boost_full_Res.py
python3 src/analysis/explainable_ai.py

### 2. Run the project projecthandin.ipynb





## Impact

By extending electricity price visibility from 24 hours to 120 hours, this project demonstrates how machine learning and forecasting can support smarter consumer behaviour and improved grid resilience. When many users shift consumption away from predicted high-price periods, the electricity system becomes more stable, efficient, and sustainable.
