# Essential Food Item Inflation Forecaster

An end-to-end data science project for forecasting essential food prices and detecting sudden inflation shocks using historical retail-price data from the Government of India.

## Overview

Sudden increases in the prices of essential commodities such as tomatoes, onions, potatoes, pulses, and edible oils can significantly affect household budgets. Existing government price-monitoring platforms primarily provide historical and current price information but do not provide predictive forecasting or consumer-oriented inflation alerts.

This project aims to build an **Essential Food Item Inflation Forecaster** that predicts future retail prices and identifies potential inflation shocks using historical price trends and seasonal patterns.

## Problem Statement

The system focuses on forecasting retail prices for essential food commodities and identifying significant week-on-week price increases.

An **inflation shock** is defined as a retail price increase of **20% or more** compared with the previous observation.

The system is designed to support:

* Short-term price forecasting
* Early detection of significant price shocks
* Explainable predictions
* Consumer-oriented food budget planning

## Dataset

The project uses real-world retail-price data published by the **Price Monitoring Division, Department of Consumer Affairs, Government of India**, through `fcainfoweb.nic.in`.

The dataset covers Maharashtra and includes historical observations for essential commodities across 2024–2026.

The core commodity basket includes items such as:

* Rice
* Wheat and Atta
* Pulses
* Sugar
* Milk
* Edible oils
* Tea
* Salt
* Potato
* Onion
* Tomato

## Data Pipeline

```text
Government Retail Price Data
            ↓
      Data Profiling
            ↓
   Data Cleaning & Validation
            ↓
     Feature Engineering
            ↓
     Forecasting Models
            ↓
   Inflation Shock Detection
            ↓
 Explainability & Evaluation
            ↓
 Consumer-Facing Prediction
```

## Data Preparation

The preprocessing pipeline includes:

* Missing-value analysis and treatment
* Duplicate detection
* Data-type standardization
* Outlier identification
* Temporal feature extraction
* Lag features
* Rolling statistics
* Week-on-week price variation
* Inflation-shock labeling
* Schema and data-quality validation

## Feature Engineering

Important features include:

* Year, month, quarter and week
* Cyclical seasonal features
* Previous-period prices
* Rolling price averages
* Week-on-week percentage change
* Inflation-shock indicators

These features capture both **short-term price momentum and seasonal behaviour**.

## Technology Stack

| Category         | Technologies             |
| ---------------- | ------------------------ |
| Programming      | Python                   |
| Data Processing  | Pandas, NumPy, PyJanitor |
| Data Profiling   | YData Profiling          |
| Data Validation  | Great Expectations       |
| Version Control  | Git, DVC                 |
| Visualization    | Matplotlib               |
| Machine Learning | Scikit-learn             |
| Deep Learning    | PyTorch                  |
| Explainability   | SHAP                     |
| Development      | Jupyter Notebook         |

## Project Goals

The forecasting system targets:

* **≤10% MAPE** for 7-day forecasts
* **≤15% MAPE** for 30-day forecasts
* **≥85% recall** for inflation-shock detection
* **≥90% post-cleaning data coverage**

## Reproducibility

Git is used for source-code version control, while **DVC** is used to version datasets and maintain reproducible data pipelines.

```bash
dvc add <dataset>
dvc push
```

## Future Scope

The project can be extended with:

* Multi-city and multi-state forecasting
* Additional economic and weather indicators
* Advanced time-series and transformer models
* Probabilistic forecasting and uncertainty intervals
* SHAP-based explanations for price movements
* Consumer-facing budget recommendations
* Deployment through a REST API and interactive dashboard

## License

This project is intended for academic and research purposes.
