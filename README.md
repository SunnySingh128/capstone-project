# Subscription Revenue Forecasting Service

**Author**: Sunny Singh ([@SunnySingh128](https://github.com/SunnySingh128))  
**Repository**: [https://github.com/SunnySingh128/capstone-project](https://github.com/SunnySingh128/capstone-project)

A reproducible, containerized time-series forecasting service for subscription revenue across multiple countries. It ingests invoice data, builds daily/monthly country-level series and an `all` aggregate, compares forecasting models, versions trained artifacts, exposes a Flask API, logs predictions, and includes lightweight performance and data drift monitoring.

> **Currency and accounting note:** The sample data uses a single synthetic reporting currency. If source countries bill in different currencies, convert amounts to a shared reporting currency before summing the `all` series. This implementation treats each row's price/mapped revenue field as a signed revenue amount; adjust for quantity, tax, discounts, refunds, invoice status, and revenue-recognition rules as needed.

---

## Features

- **Multi-Format Ingestion**: Ingests CSV, JSON, and JSONL formats with automated schema validation and daily/monthly aggregation.
- **Complete Time Series Generation**: Builds per-country and `all` total aggregate time series; missing periods inside the observed global date range are filled with zero.
- **Exploratory Data Analysis (EDA)**: Produces baseline comparison, country trajectories, revenue distributions, and seasonal trend plots saved to `eda/figures/`.
- **Model Comparison & Selection**: Evaluates Naive last-value baseline, Random Forest, Gradient Boosting, and Linear Regression on a chronological holdout using RMSE and MAE. The best machine learning model is selected by holdout RMSE (tie-broken by MAE) and refit on all historical data.
- **Artifact Versioning & Registry**: Timestamped model artifacts saved with a `latest.json` pointer for each country.
- **Production Flask API**: Fully featured REST endpoints: `GET /health`, `POST /train`, `GET /predict`, `POST /predict`, `GET /logs`, and `POST /monitor`.
- **Audit Logging**: CSV audit logs for model training (`training.csv`) and inference (`predictions.csv`).
- **Performance & Drift Monitoring**: Model performance evaluation against holdout thresholds and standardized mean-shift drift detection ($Z_{\text{shift}}$).
- **Isolated Testing Suite**: 100% of unit and integration tests use temporary directories without altering production models or logs.

---

## Repository Layout

```text
capstone-project/
├── app.py                 # Flask REST API endpoints
├── data_ingestion.py      # CLI ingestion pipeline (CSV, JSON, JSONL)
├── model.py               # Feature engineering, training, versioning, recursive prediction
├── logger.py              # CSV audit logging (training and prediction events)
├── monitoring.py          # Performance metrics & standardized mean-shift drift monitoring
├── run_tests.py           # Single test runner for all unit/integration tests
├── requirements.txt       # Production dependencies
├── Dockerfile             # Multi-stage containerization with Python 3.12
├── README.md              # Comprehensive project documentation
├── data/
│   ├── raw/invoices.csv   # Synthetic sample invoice dataset (2024–2027)
│   └── processed/         # Generated processed time series (revenue_timeseries.csv)
├── models/
│   ├── production/        # Production model artifacts & latest.json pointer
│   └── test/              # Isolated test model artifacts
├── logs/
│   ├── production/        # Production audit logs (training.csv, predictions.csv)
│   └── test/              # Isolated test logs
├── eda/
│   ├── eda.py             # EDA visual chart generator
│   └── figures/           # Generated visual figures (PNG)
└── tests/
    ├── __init__.py
    ├── test_data_ingestion.py
    ├── test_model.py
    ├── test_logger.py
    ├── test_monitoring.py
    └── test_api.py
```

---

## Requirements

- Python 3.10+ is recommended (Python 3.12 is used in Docker).
- Dependencies: `pandas`, `numpy`, `scikit-learn`, `flask`, `matplotlib`, `joblib`.

---

## Install and Run Locally

### 1. Setup Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate       # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Ingest Sample Invoice Data

Build monthly aggregated series from raw invoices:
```bash
python data_ingestion.py --input-dir data/raw --output-dir data/processed --frequency monthly
```

To build daily series instead:
```bash
python data_ingestion.py --input-dir data/raw --output-dir data/processed --frequency daily
```

If raw column names differ from `date`, `country`, and `price`, provide column mappings:
```bash
python data_ingestion.py --date-column invoice_date --country-column billing_country --revenue-column amount
```

### 3. Generate Exploratory Plots (EDA)

```bash
python eda/eda.py --input data/processed/revenue_timeseries.csv --output-dir eda/figures
```
Generated charts saved in `eda/figures/`:
- `overall_revenue_trend.png`
- `country_revenue_comparison.png`
- `country_revenue_distribution.png`
- `monthly_seasonality.png`
- `baseline_visualization.png`

---

## Train Models

### Via Flask API

In one terminal, start the API:
```bash
python app.py
```

In another terminal, trigger model training:
```bash
curl -X POST http://localhost:5000/train \
  -H 'Content-Type: application/json' \
  -d '{}'
```

### Directly from Python

```bash
python -c "import pandas as pd; from model import train_models; print(train_models(pd.read_csv('data/processed/revenue_timeseries.csv')))"
```

The `/train` endpoint accepts optional `random_state`, `test_size`, `data_path`, and `test` fields. For example, `{"test": true}` routes model artifacts and training logs to test directories (`models/test/` and `logs/test/`).

---

## API Examples

### Forecast Future Revenue (`GET` or `POST` `/predict`)

Forecast one country for a date after the last date in its training history:
```bash
curl 'http://localhost:5000/predict?country=India&date=2028-01-01'
```

Forecast all countries combined:
```bash
curl 'http://localhost:5000/predict?country=all&date=2028-01-01'
```

POST is also supported:
```bash
curl -X POST http://localhost:5000/predict \
  -H 'Content-Type: application/json' \
  -d '{"country":"USA","date":"2028-01-01"}'
```

### View Audit Logs (`GET /logs`)

```bash
curl 'http://localhost:5000/logs?kind=all&limit=100'
curl 'http://localhost:5000/logs?kind=predictions&limit=20'
curl 'http://localhost:5000/logs?kind=training&limit=20'
```

### Performance & Drift Monitoring (`POST /monitor`)

Performance monitoring using the model's holdout metrics against an acceptable threshold:
```bash
curl -X POST http://localhost:5000/monitor \
  -H 'Content-Type: application/json' \
  -d '{"country":"all","threshold":3000}'
```

Standardized mean-shift drift check ($Z_{\text{shift}} = \frac{|\mu_{\text{current}} - \mu_{\text{reference}}|}{\sigma_{\text{reference}}}$):
```bash
curl -X POST http://localhost:5000/monitor \
  -H 'Content-Type: application/json' \
  -d '{"country":"all","reference_values":[100,110,105,98],"current_values":[250,260,255]}'
```

### Health Check (`GET /health`)

```bash
curl -f http://localhost:5000/health
```

---

## Run Tests

Run the complete isolated test suite:
```bash
python run_tests.py
```

The tests use temporary model and log directories. They do not depend on existing production artifacts or write to production logs.

---

## Docker

### Build the Image
```bash
docker build -t subscription-revenue-forecast:latest .
```

### Run the API Container
Mount persistent local data, model, and log folders:
```bash
docker run --rm -p 5000:5000 \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/logs:/app/logs" \
  subscription-revenue-forecast:latest
```

### Run Ingestion and Training inside Running Container
```bash
# Ingest data inside running container
docker exec "$(docker ps -q --filter ancestor=subscription-revenue-forecast:latest | head -n 1)" \
  python data_ingestion.py --input-dir data/raw --output-dir data/processed --frequency monthly

# Train models inside container
docker exec "$(docker ps -q --filter ancestor=subscription-revenue-forecast:latest | head -n 1)" \
  python -c "import pandas as pd; from model import train_models; train_models(pd.read_csv('data/processed/revenue_timeseries.csv'))"
```

### Smoke Tests
```bash
curl -f http://localhost:5000/health
curl -f 'http://localhost:5000/predict?country=all&date=2028-01-01'
```

### Run Test Suite in Docker
```bash
docker run --rm subscription-revenue-forecast:latest python run_tests.py
```

---

## Model Comparison Summary

The training pipeline evaluates a **Naive last-observation baseline**, **Random Forest**, **Gradient Boosting**, and **Linear Regression** using chronological holdout validation and reports RMSE and MAE.

1. **Chronological Holdout Split**: The last 20% of historical periods are held out as an out-of-sample test window.
2. **Naive Baseline**: Projects the previous period's value forward ($y_{t} = y_{t-1}$).
3. **Candidate ML Models**: Fits Linear Regression, Random Forest, and Gradient Boosting on train periods using lag features ($t-1, t-2, t-3, t-6$), rolling mean ($3$-period), and calendar indicators (month, year, quarter, linear time index).
4. **Selection Logic**: The best ML model is selected strictly by holdout RMSE (with MAE as a tie-breaker). The Naive baseline is preserved for benchmark comparison and is never chosen as the selected ML model.
5. **Full History Refit**: The winning ML candidate is refit on all available historical data (train + holdout test) so it possesses the most up-to-date trajectory for future multi-step forecasting.
6. **Artifact Storage**: The refit model, feature set, metrics, and latest.json pointer are written to `models/production/`.

---

## Final Checklist and Proof Commands

| Requirement | Verification Command / Action |
| :--- | :--- |
| **1. Data ingestion** | `python data_ingestion.py --input-dir data/raw --output-dir data/processed --frequency monthly` |
| **2. EDA** | `python eda/eda.py --input data/processed/revenue_timeseries.csv --output-dir eda/figures` |
| **3. Model comparison** | `curl -X POST localhost:5000/train -H 'Content-Type: application/json' -d '{}'` and inspect returned RMSE/MAE |
| **4. Baseline visualization** | See `eda/figures/baseline_visualization.png`; holdout model comparison returned in artifacts |
| **5. Versioned model module** | `python -c "from model import load_model; print(load_model('all')['version'])"` after training |
| **6. Flask endpoints** | `curl -f localhost:5000/health`; `curl 'localhost:5000/predict?country=all&date=2028-01-01'`; `curl localhost:5000/logs` |
| **7. CSV logging** | `cat logs/production/training.csv logs/production/predictions.csv` after training/prediction |
| **8. Monitoring** | `curl -X POST localhost:5000/monitor -H 'Content-Type: application/json' -d '{"country":"all","threshold":3000}'` |
| **9. Unit tests** | `python run_tests.py` |
| **10. Test isolation** | `python run_tests.py`; tests create temporary directories and `/train` with `{"test": true}` uses `models/test`, `logs/test` |
| **11. Single test runner** | `python run_tests.py` |
| **12. Docker** | `docker build -t subscription-revenue-forecast:latest .` then run container and `curl /health` |

---

## Author
**Sunny Singh**  
GitHub: [SunnySingh128](https://github.com/SunnySingh128)  
Email: [sunnysingh14092005@gmail.com](mailto:sunnysingh14092005@gmail.com)
