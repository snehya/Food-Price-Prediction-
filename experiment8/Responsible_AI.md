
# Responsible AI Report

## 1. Purpose

This project develops a machine learning model for forecasting Tomato retail prices using historical food-price data from Maharashtra.

The dashboard is intended as an experimental decision-support tool. Predictions should not be treated as guaranteed future prices.

## 2. Dataset and Scope

- **Geographic scope:** Maharashtra
- **Target variable:** Tomato retail price
- **Dataset size:** 85 observations
- **Data period:** 2024–2026
- **Model:** Tuned Random Forest Regressor
- **Test RMSE:** 2.7064

The project uses aggregated food-price information and does not require personal user information for prediction.

## 3. Privacy

The dataset used in this project does not contain personally identifiable information.

The dashboard does not request names, addresses, phone numbers, or other personal identifiers.

Future deployments should continue to avoid collecting unnecessary personal information.

## 4. Fairness

This project does not contain protected demographic attributes such as gender, caste, religion, or income.

Therefore, demographic fairness cannot be directly evaluated from the current dataset.

Any grouping used for model auditing should not automatically be interpreted as a protected demographic group.

The current project is limited to Maharashtra, which also limits the ability to evaluate geographic fairness across different states or regions.

## 5. Explainability

SHAP is used to explain the Random Forest model.

The dashboard displays global SHAP feature importance to show which features have the greatest overall influence on model predictions.

The most influential feature in the current analysis is `Tomato_Lag1`.

SHAP describes model behavior and feature contribution. It does not establish that a feature causes the predicted price.

## 6. Model Performance

The baseline Random Forest achieved a test RMSE of 2.8525.

After hyperparameter tuning, the Random Forest achieved a test RMSE of 2.7064.

This represents an approximately 5.12% reduction in test RMSE.

These results are based on the available historical dataset and should not be assumed to represent future real-world performance.

## 7. Data Limitations

The dataset is relatively small, with 85 observations.

The current project focuses only on Maharashtra and one target commodity.

Food prices can also be affected by factors that are not represented in the current model, including supply conditions, weather, transportation, seasonal events, and market conditions.

Therefore, the model may not generalize well to different regions or future market conditions.

## 8. Model Limitations

The model uses historical price and calendar-based features.

Its predictions are dependent on the quality and distribution of the available historical data.

A lower test RMSE does not guarantee accurate future predictions.

The dashboard should therefore be treated as an experimental forecasting system rather than an autonomous decision-making system.

## 9. Human Oversight

Predictions should be reviewed by a human before being used for important decisions.

The system should support analysis rather than automatically making decisions about pricing, procurement, or policy.

Users should consider external market information alongside model predictions.

## 10. Drift Monitoring

The dashboard includes a simple indicative drift check comparing average Tomato prices between the training and test periods.

This is not a production-grade monitoring system.

A real deployment should monitor changes in feature distributions, prediction errors, data quality, and model performance over time.

## 11. Responsible Deployment

Before production deployment, the system should be evaluated using:

- Larger datasets
- Multiple regions
- Additional commodities
- More relevant economic and environmental features
- Continuous performance monitoring
- Periodic model validation and retraining

Any production deployment should also document the intended users, decision context, model limitations, and escalation procedures.

## 12. Future Improvements

Potential future improvements include:

1. Expanding the dataset to additional states and cities.
2. Including weather and supply-related variables.
3. Testing additional forecasting models.
4. Adding stronger statistical drift detection.
5. Evaluating fairness across appropriate geographic or socioeconomic groups when such data is available.
6. Adding continuous model-performance monitoring.
7. Periodically retraining the model with new data.

## 13. Conclusion

This project demonstrates a small end-to-end responsible ML workflow for food-price forecasting.

The system combines machine learning, explainability, performance evaluation, and basic drift analysis while explicitly documenting the limitations of the dataset and model.
