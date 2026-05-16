# Treasury Spread Forecasting with Darts

*AutoARIMA on the 10Y-2Y yield curve spread*

---

The spread between 10-year and 2-year Treasury yields is one of the most closely watched economic indicators. When short-term rates exceed long-term rates, it has historically preceded recessions with remarkable consistency. Where the spread goes next is a legitimate question for economists, portfolio managers, and risk teams.

## The Data

The 10Y-2Y spread (`T10Y2Y`) is available from FRED via `pandas_datareader`:

```python
import pandas_datareader.data as web
from darts import TimeSeries

df = web.DataReader("T10Y2Y", "fred", start="2000-01-01")
df = df.rename(columns={"T10Y2Y": "value"})
df["value"] = df["value"].ffill().dropna()

series = TimeSeries.from_dataframe(df, value_cols="value")
```

The spread runs from roughly -1.5 percentage points (deeply inverted, as in 2022-23) to +3.5 (steeply positive, as in 2010-11). Over long horizons it is stationary and mean-reverting, but around FOMC tightening cycles it shows clear regime behavior.

## Why Darts?

You could fit AutoARIMA directly through `statsmodels` and get a point forecast. Darts adds two things worth having: probabilistic output via `num_samples`, and a backtesting API that handles the rolling-window evaluation pattern without boilerplate. The same API works across ARIMA, exponential smoothing, and neural network models, so you can swap backends without rewriting the evaluation layer.

## AutoARIMA

AutoARIMA selects the best ARIMA(p,d,q) order by AIC. For the yield spread, which is stationary, it typically picks d = 0 and moderate p.

```python
from darts.models import AutoARIMA

forecast_horizon = 30  # days

train = series[:-forecast_horizon]
test = series[-forecast_horizon:]

model = AutoARIMA()
model.fit(train)

forecast = model.predict(n=forecast_horizon, num_samples=1000)
```

`num_samples=1000` generates a distribution of futures. The 5th and 95th percentiles of that distribution form the prediction interval.

## Evaluation

```python
from darts.metrics import mape, mse

mape_score = mape(test, forecast)
mse_score = mse(test, forecast)

print(f"MAPE: {mape_score:.2f}%")
print(f"MSE:  {mse_score:.6f}")
```

On a 30-day horizon, expect MAPE in the 15-30% range. The 90% prediction interval will be wide. That is not a failure of the model. It is an accurate reflection of how noisy the spread is at short horizons.

The key check is calibration: does the actual value fall within the 90% interval roughly 90% of the time? The `historical_forecasts` method runs this test without writing a loop by hand:

```python
backtest = model.historical_forecasts(
    series,
    start=0.7,
    forecast_horizon=30,
    stride=5,
    retrain=True,
    verbose=False,
    num_samples=200
)
```

## Plotting

```python
import matplotlib.pyplot as plt

window = series[-365:]
fig, ax = plt.subplots(figsize=(12, 5))
window.plot(label="Actual", ax=ax)
forecast.plot(label="Forecast", low_quantile=0.05, high_quantile=0.95, ax=ax)
ax.axhline(0, linestyle='--', linewidth=0.8, alpha=0.6)
ax.set_ylabel("Spread (pp)")
ax.legend()
plt.tight_layout()
```

The zero line marks the boundary between a normal and inverted yield curve. Worth keeping in the chart.

## What the Model Cannot Do

AutoARIMA treats the spread as a statistical object. It has no knowledge of Fed policy, inflation expectations, or the shape of the forward curve. That is both a strength and a limitation.

The strength: no assumptions about which direction the spread should move. It fits what actually happened.

The limitation: regime change breaks it. A model trained before 2022 would not have seen a tightening cycle of that speed and magnitude, so its intervals would not have covered the inversion. That is not fixable with a better time series algorithm. It requires forward-looking covariates, Fed funds futures or SOFR swaps, fed through the `future_covariates` argument that Darts supports in many models.

---

A wide confidence interval on a 30-day rate forecast is honest. Anyone offering you a tight one is either sitting on an exceptional model or underestimating uncertainty. The point of this exercise is not to pin down the spread. It is to understand the distribution of outcomes well enough to decide whether to hedge, extend duration, or adjust rate sensitivity.
