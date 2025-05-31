# category_buffer_time_forecast_categorywide.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import sys
import os

# ――――――――― 0. CONFIGURATION ―――――――――
FILE_PATH = '/Users/siddhant/Desktop/Beedie Hackathon/EVERYTHING.xlsx'  # ← Update to your local path
FORECAST_MONTHS = 6    # how many months ahead to forecast
OUTPUT_DIR = '.'       # directory to save PNGs

# The exact Category NAME values in your sheet:
LOW_TOLERANCE_CATEGORIES = [
    'Distribution Transf',
    'Switchgear',
    'Wire And Cable',
    'Aux Elec Equip'
]

# ――――――――― 1. LOAD DATA ―――――――――
try:
    df_inv     = pd.read_excel(FILE_PATH, sheet_name='Vendors Inventory')
    df_vendors = pd.read_excel(FILE_PATH, sheet_name='BCH All Vendor Data')
    df_tariff  = pd.read_excel(FILE_PATH, sheet_name='Tariffs Rates by Country')
except FileNotFoundError:
    print(f"\nERROR: File not found at '{FILE_PATH}'. Please verify the path.\n")
    sys.exit(1)
except Exception as e:
    print("\nERROR: Failed to read Excel file:\n", str(e))
    sys.exit(1)

# ――――――――― 2. VERIFY COLUMNS & COMPUTE BUFFER TIME ―――――――――
required_cols = [
    'START DATE',
    'Average Lead time (days)',
    'Days of supply (current)',
    'Category NAME',
    'Country Exporting',
    'VendorNumber'
]
missing_cols = [c for c in required_cols if c not in df_inv.columns]
if missing_cols:
    print("\nERROR: Missing columns in 'Vendors Inventory':", missing_cols)
    print("Available columns:", df_inv.columns.tolist())
    sys.exit(1)

# Compute raw buffer, clip at zero
df_inv['Raw_Buffer'] = df_inv['Days of supply (current)'] - df_inv['Average Lead time (days)']
df_inv['Buffer Time (days)'] = df_inv['Raw_Buffer'].clip(lower=0)

# Drop any rows where Buffer Time is not finite
mask_bad = ~np.isfinite(df_inv['Buffer Time (days)'])
if mask_bad.any():
    df_inv = df_inv[~mask_bad].copy()

# ――――――――― 3. MAP VENDOR PERFORMANCE → SCORE (1–5) ―――――――――
perf_map = {'Excellent': 5, 'Good': 4, 'Average': 3, 'Developing': 2, 'Poor': 1}
df_vendors['Vendor_Score'] = df_vendors['Vendor performance'].map(perf_map).fillna(3)

# ――――――――― 4. FORECASTING PER CATEGORY (AGGREGATED ACROSS ALL COUNTRIES) ―――――――――
for category in LOW_TOLERANCE_CATEGORIES:
    # 4.1 Filter inventory to this category
    df_cat = df_inv[df_inv['Category NAME'] == category].copy()
    if df_cat.empty:
        print(f"No data for category '{category}'. Skipping.")
        continue

    # 4.2 Create monthly Buffer Time series (across all exporting countries)
    df_cat['month'] = pd.to_datetime(df_cat['START DATE']).dt.to_period('M').dt.to_timestamp()
    monthly_df = (
        df_cat
        .groupby('month')['Buffer Time (days)']
        .mean()
        .reset_index()
    )

    # Convert to a 1D Series indexed by month with frequency MS (month start)
    monthly_series = monthly_df.set_index('month')['Buffer Time (days)']
    monthly_series = monthly_series.asfreq('MS')  # fill any missing months with NaN

    # Drop NaNs to produce a clean numeric series
    monthly_clean = monthly_series.dropna()
    n_points = len(monthly_clean)

    if n_points < 2:
        print(f"Not enough data points ({n_points}) for forecasting '{category}'. Skipping.")
        continue

    # 4.3 Choose model: if ≥ 24 months of data, use Holt-Winters additive seasonal; else simple exponential
    if n_points >= 24:
        model = ExponentialSmoothing(
            monthly_clean,
            trend='add',
            seasonal='add',
            seasonal_periods=12
        )
    else:
        model = ExponentialSmoothing(
            monthly_clean,
            trend=None,
            seasonal=None
        )

    # 4.4 Fit model
    try:
        fit = model.fit(optimized=True)
    except Exception as e:
        print(f"Error fitting model for '{category}': {e}")
        continue

    # 4.5 Forecast next FORECAST_MONTHS
    last_date = monthly_clean.index.max()
    future_index = pd.date_range(
        start=last_date + pd.offsets.MonthBegin(1),
        periods=FORECAST_MONTHS,
        freq='MS'
    )
    forecast_values = fit.forecast(FORECAST_MONTHS)
    forecast = pd.Series(forecast_values, index=future_index, name='Forecast')

    # 4.6 Plot historical + forecast
    plt.figure(figsize=(10,5))
    plt.plot(monthly_series.index, monthly_series, label='Historical', marker='o')
    plt.plot(forecast.index, forecast.values, label='Forecast', marker='o', linestyle='--')
    plt.title(f"{category} – Buffer Time Forecast (Next {FORECAST_MONTHS} Months)")
    plt.xlabel('Date')
    plt.ylabel('Buffer Time (days)')
    plt.legend()
    plt.tight_layout()

    # 4.7 Save plot
    safe_cat = category.replace(' ', '_')
    filename = os.path.join(OUTPUT_DIR, f"{safe_cat}_BufferForecast.png")
    plt.savefig(filename, dpi=200)
    plt.close()

    # 4.8 Print forecast values in console
    print(f"\nCategory: {category}")
    print("Next 6-Month Forecast of Buffer Time (days):")
    print(forecast.round(1).to_string())

print("\nForecasting complete. PNGs saved to:", os.path.abspath(OUTPUT_DIR))
