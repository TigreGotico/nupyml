# sine_forecast

**Goal:** forecast the next 24 steps of a trended, seasonal, noisy series.

- **Kind:** forecast — `solve(y_history, horizon) -> y_future`
- **Data:** a synthetic series = level + linear trend + period-12 seasonality + noise (240 points; the last 24 are held out).
- **Metric:** RMSE (**lower is better**).
- **QA floor:** 6.0 (loose — even the naive "repeat last value" reference clears it). A strong seasonal forecaster stays under ~1.5.

The series has both a trend and a season, so a method that models both (Holt-Winters, RMSE ~1.2) crushes one that models neither (naive-last, ~5.3). Plain ARIMA without seasonal terms lands in between — a reminder that the model has to match the structure.
