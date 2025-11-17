import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ruptures as rpt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

os.chdir("/Users/cherylyao/Desktop/neoval-project")
df = pd.read_csv("data/df_factor_trends.csv")
df["month_date"] = pd.to_datetime(df["month_date"])

df_sub = df[(df["month_date"] >= "2000-01") & (df["month_date"] <= "2020-12")]
y_sub = df_sub["mining"].values.reshape(-1,1)
                                                                                                    
algo = rpt.Binseg(model = "rbf").fit(y_sub)
breakpoints = algo.predict(n_bkps = 3)[:-1]

for start, end in zip([0] + breakpoints, breakpoints + [len(df_sub)]):
    start_date = df_sub.iloc[start]["month_date"].strftime("%Y-%m")
    end_date = df_sub.iloc[end-1]["month_date"].strftime("%Y-%m")
    print(f"Segment: {start_date} to {end_date}")

# Plot the full mining factor series
plt.figure(figsize=(10,4))
plt.plot(df["month_date"], df["mining"], color = "black", label = "Mining factor")

# Overlay fitted linear trends for each segment
for start, end in zip([0] + breakpoints, breakpoints + [len(y_sub)]):
    segment_x = np.arange(start, end)
    segment_y = y_sub[start:end].ravel()
    coeffs = np.polyfit(segment_x, segment_y, 1)  
    fitted = np.polyval(coeffs, segment_x)
    plt.plot(df.loc[(df["month_date"] >= "2000-01") & (df["month_date"] <= "2020-12"), "month_date"].iloc[start:end],
             fitted, color="red", linewidth=2)

# Add vertical dashed lines at breakpoints
dates_sub = df.loc[(df["month_date"] >= "2000-01") & (df["month_date"] <= "2020-12"), "month_date"]
for bp in breakpoints:
    plt.axvline(x=dates_sub.iloc[bp], color="blue", linestyle="--")

plt.title("Mining Time Series with Breakpoints")
plt.xlabel("Date")
plt.ylabel("Mining factor (Perth - Sydney Spread)")
plt.legend()
plt.savefig("mining_trend_breakpoints.pdf", format = "pdf", bbox_inches = "tight")
plt.show()

# %%
# Include dummy variables for the breakperiod
df["boom_dummy"] = (
    (df["month_date"] >= "2005-06") & 
    (df["month_date"] <= "2015-05")
).astype(int)

df["decline_dummy"] = (
    (df["month_date"] >= "2015-06") & 
    (df["month_date"] <= "2016-08")
).astype(int)

df["recovery_dummy"] = (
    (df["month_date"] >= "2016-09") & 
    (df["month_date"] <= "2020-12")
).astype(int)

# %%
# Stationary test 
y = df["mining"].dropna()

adf_raw = adfuller(y)
print("ADF Test on RAW factor")
print(f"ADF statistic: {adf_raw[0]:.4f}")
print(f"p-value: {adf_raw[1]:.4f}")
print("----------------------------------")

y_diff = y.diff().dropna()

adf_diff = adfuller(y_diff)
print("ADF Test on First")
print(f"ADF statistic: {adf_diff[0]:.4f}")
print(f"p-value: {adf_diff[1]:.4f}")
print("----------------------------------")

regimes = [
    ("2000-01", "2005-05"),
    ("2005-06", "2015-05"),
    ("2015-06", "2016-08"),
    ("2016-09", "2020-12")
]

print("ADF Test on Each Regime (local stationarity)")

for i, (start, end) in enumerate(regimes, 1):
    reg_series = df.loc[(df["month_date"] >= start) &
                        (df["month_date"] <= end), "mining"].dropna()
    adf_reg = adfuller(reg_series)
    print(f"Regime {i}: {start} to {end}")
    print(f"   ADF statistic: {adf_reg[0]:.4f}")
    print(f"   p-value:       {adf_reg[1]:.4f}")
    print("----------------------------------")


# ARIMAX with the exogenous regressors 
y = df["mining"]
X = df[["boom_dummy", "decline_dummy", "recovery_dummy"]]

model = SARIMAX(
    df["mining"], 
    exog = X,
    order = (2,0,1),
    enforce_stationarity=True,
    enforce_invertibility=True
)
res = model.fit(method="powell", disp = False)
print(res.summary())

# %%
# 10 year forecast 
n_forecast = 120  
last_date = df["month_date"].iloc[-1]

forecast_index = pd.date_range(
    start = last_date + pd.DateOffset(months=1),
    periods = n_forecast,
    freq="MS"
)

X_future = pd.DataFrame(
    np.zeros((n_forecast, 3)),
    columns=["boom_dummy", "decline_dummy", "recovery_dummy"]
)

pred = res.get_forecast(steps=n_forecast, exog=X_future)
pred_mean = pred.predicted_mean
conf_int = pred.conf_int()
sigma_10y = pred.se_mean.iloc[-1]
x95 = np.exp(1.96 * sigma_10y)

print("10-year ahead standard deviation:", round(sigma_10y, 3))
print("Multiplicative 95 percent band:", round(x95, 2))

# %%
plt.figure(figsize = (10,4))

# Observed
plt.plot(df["month_date"], df["mining"], 
         label = "Observed", linewidth = 1.75)

# Forecast
plt.plot(forecast_index, pred_mean, 
         label = "Forecast", linewidth = 1.75)

# Confidence interval
plt.fill_between(
    forecast_index,
    conf_int.iloc[:, 0],
    conf_int.iloc[:, 1],
    alpha = 0.20,
    label = "95% interval"
)

plt.axhline(0, lw = 0.8, ls = "--", alpha = 0.5)

plt.xlabel("Month")
plt.ylabel("Level (log)")
plt.title("a) Mining")
plt.legend(loc = "upper left", frameon = False)
plt.tight_layout()
plt.savefig("mining_break_funnel.pdf", format = "pdf", bbox_inches = "tight")
plt.show()


