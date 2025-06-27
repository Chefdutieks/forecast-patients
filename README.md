# Forecast Patients

This repository groups a set of scripts used to predict the number of patients in a clinic or hospital environment. The project retrieves historical data from a production database, processes it to create training sets, and trains several machine‑learning models. Predictions can then be exposed through a small API or written back to Google Sheets.

## Why this project?

Planning how many doctors are required at any given time is difficult. By forecasting patient arrivals, the operations team can better adjust staffing levels and thus reduce waiting times. Automating these forecasts also allows the team to monitor trends week after week.

## Intended audience

The codebase is aimed at data analysts and operational managers who need reproducible scripts to build the models and generate predictions. Familiarity with Python and basic command‑line usage is assumed.

## Expected impact

Accurate forecasts help clinics allocate medical staff at the right time and anticipate peaks in activity. Better anticipation of patient numbers ultimately improves service quality and optimises resource utilisation.

## Repository structure

Below is a short description of each Python module.

| File | Description |
|------|-------------|
|`TrainModels.py`|Training pipeline that tunes and saves regression models such as LightGBM. It performs grid search with time-series cross validation and archives the generated models. See lines 1‑8 for the module summary【F:TrainModels.py†L2-L8】.|
|`besoinsMG.py`|Exploratory script that analyses the ratio between available doctors and patients to estimate an ideal staffing level. Example calculations start at line 14【F:besoinsMG.py†L14-L20】.|
|`featuresAnalysis.py`|Utilities to inspect feature quality. It computes correlations and VIF scores after preprocessing the data retrieved from `get_train_data` (lines 5‑11)【F:featuresAnalysis.py†L5-L11】.|
|`main.py`|Example scheduler for a weekly task that reads Google Sheets, trains the model and writes results back. The header comment explains the goal (lines 3‑8)【F:main.py†L3-L8】.|
|`api.py`|Flask API exposing a `/predict` endpoint. It validates input, loads a saved model and returns predictions in JSON (lines 1‑10)【F:api.py†L1-L10】.|
|`data_processing.py`|Collection of helpers to read/write Google Sheets, generate future dates and preprocess them so they match the training format. Generation of future data is shown in lines 112‑142【F:data_processing.py†L112-L142】.|
|`get_train_data.py`|Connects to the production database using credentials from environment variables and returns the cleaned training dataframe. The connection configuration starts at line 21【F:get_train_data.py†L21-L28】.|
|`predict.py`|Uses trained models to evaluate predictions on recent history and to forecast future weeks. Metrics are compiled as illustrated around line 30 onwards【F:predict.py†L30-L54】.|

## Running the project

1. **Configure environment variables** – Database and API credentials are loaded from `.env`. At a minimum you need the Azure database settings (`AZURE_DB_SERVER`, `AZURE_DB_NAME`, etc.) and a service account for Google Sheets.
2. **Retrieve training data** – Run `python get_train_data.py` to fetch and preprocess historical patient data.
3. **Train models** – Execute `python TrainModels.py` to perform cross‑validation and save the best models inside `models/current`.
4. **Generate forecasts** – Use `python predict.py` to create CSV files with future predictions.
5. **Serve predictions** – Launch the API locally with `python api.py` and send authenticated GET requests to `/predict`.

The scripts rely on external libraries such as `pandas`, `scikit-learn`, `xgboost` and `lightgbm`. Install the project requirements to reproduce the results.

