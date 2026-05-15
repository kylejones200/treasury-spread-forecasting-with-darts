# Treasury Spread Forecasting with Darts

*Using AutoARIMA and the Darts library to forecast the 10Y-2Y yield curve spread*

---

The spread between 10-year and 2-year Treasury yields is one of the most closely watched economic indicators. When short-term rates exceed long-term rates — an inverted yield curve — it has historically preceded recessions with remarkable consistency. Forecasting where the spread is headed is a legitimate question for economists, portfolio managers, and risk teams.

This article uses the Darts time series library and AutoARIMA to forecast the T10Y2Y spread from FRED, with probabilistic prediction intervals.

## The Data

The 10Y-2Y spread (`T10Y2Y`) is available directly from FRED via `pandas_datareader`:

```python
import pandas_datareader.data as web
from darts import TimeSeries

df = web.DataReader("T10Y2Y", "fred", start="2000-01-01")
df = df.rename(columns={"T10Y2Y": "value"})
df["value"] = df["value"].ffill().dropna()

series = TimeSeries.from_dataframe(df, value_cols="value")
```

The spread is measured in percentage points. It ranges from roughly -1.5% (deeply inverted, as in 2022–2023) to +3.5% (steeply positive, as in 2010–2011). It is a stationary series over long horizons — it mean-reverts — but has clear regime behavior around FOMC tightening cycles.

## Why Darts?

You could fit AutoARIMA directly through `statsmodels` and get a point forecast. Darts adds two things that matter: probabilistic output via `num_samples`, and a backtesting API that handles the rolling-window evaluation pattern without boilerplate. If you are building a forecasting workflow that needs to be evaluated rigorously over time, that saves significant engineering work. The same API also works across ARIMA, exponential smoothing, and neural network models — so you can swap backends without rewriting the evaluation code.

## AutoARIMA

AutoARIMA automatically selects the best ARIMA(p,d,q) order using AIC. For the yield spread, which is mean-reverting, it typically selects low d (often 0 — the series is already stationary) and moderate p.

```python
from darts.models import AutoARIMA

forecast_horizon = 30  # days

# Hold out the last 30 observations for evaluation
train = series[:-forecast_horizon]
test = series[-forecast_horizon:]

model = AutoARIMA()
model.fit(train)

# Probabilistic forecast — 1000 samples from the predictive distribution
forecast = model.predict(n=forecast_horizon, num_samples=1000)
```

The `num_samples=1000` parameter generates a distribution of possible futures rather than a single point estimate. The 5th and 95th percentile of those samples form the prediction interval.

## Evaluating the Forecast

```python
from darts.metrics import mape, mse

mape_score = mape(test, forecast)
mse_score = mse(test, forecast)

print(f"MAPE: {mape_score:.2f}%")
print(f"MSE:  {mse_score:.6f}")
```

Typical results on the 30-day horizon:
- MAPE in the 15–30% range — the spread is noisy and mean-reverts slowly
- The 90% prediction interval is wide, which is appropriate: a 30-day yield curve forecast carries substantial uncertainty

The key thing to check is **calibration**: does the true value fall within the 90% interval about 90% of the time? Darts's backtesting API makes this easy:

```python
backtest = model.historical_forecasts(
    series,
    start=0.7,           # use last 30% of data for evaluation
    forecast_horizon=30,
    stride=5,
    retrain=True,
    verbose=False,
    num_samples=200
)
```

## Visualizing the Forecast

```python
import matplotlib.pyplot as plt

# Show last 365 days of history + forecast
window = series[-365:]
fig, ax = plt.subplots(figsize=(12, 5))
window.plot(label="Actual", ax=ax)
forecast.plot(label="Forecast", low_quantile=0.05, high_quantile=0.95, ax=ax)
ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
ax.set_title("10Y-2Y Treasury Spread: AutoARIMA Forecast")
ax.set_ylabel("Spread (pp)")
ax.legend()
plt.tight_layout()
```

The horizontal line at zero is useful context: it marks the boundary between a normal and inverted yield curve.

## Interpreting the Results

AutoARIMA treats the spread as a purely statistical object — it has no knowledge that the spread is an economic indicator with specific drivers. That is both a strength and a limitation.

**Strength:** it makes no assumptions about which direction the spread "should" move based on Fed policy expectations. It just fits the historical dynamics.

**Limitation:** structural breaks matter. The model fitted before the 2022 rate hike cycle would have missed the inversion because nothing in its training history matched the speed and magnitude of that tightening. AutoARIMA will extrapolate the historical pattern, not the policy path.

For production use, consider:
- **Exogenous regressors** — add Fed funds futures as a covariate (Darts supports `future_covariates` in many models)
- **Rolling retrain** — retrain the model every 30 days to keep up with regime changes
- **Ensemble** — average AutoARIMA with an exponential smoothing model to reduce forecast variance

## Key Takeaways

A 30-day Treasury spread forecast with wide confidence intervals is not a failure — it is honest. Anyone selling you a tight interval on a 30-day rate forecast either has an exceptional model or is underestimating uncertainty. The value of this exercise is not pinning down the spread; it is understanding the distribution of outcomes well enough to make a decision: hedge or not, duration-extend or not, increase or decrease rate sensitivity.

The model's biggest blind spot is regime change. AutoARIMA extrapolates historical dynamics. When the Fed moves faster or slower than historical precedent — as it did in 2022 — the model's intervals will not cover the actual outcome. That is not fixable with a better time series model. It requires incorporating forward-looking market signals (Fed funds futures, SOFR swaps) as covariates. Darts supports this via `future_covariates`; adding them is the natural next step after validating the baseline AutoARIMA works.
