import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings('ignore')

# ——————————————————————————————————————————
# UPDATE THIS PATH to where EVERYTHING.xlsx is on your machine
# ——————————————————————————————————————————
file_path = '/Users/siddhant/Desktop/Beedie Hackathon/EVERYTHING.xlsx'

# Load sheets
df_inventory = pd.read_excel(file_path, sheet_name='Vendors Inventory')
df_vendors   = pd.read_excel(file_path, sheet_name='BCH All Vendor Data')
df_tariffs   = pd.read_excel(file_path, sheet_name='Tariffs Rates by Country')

# Compute Buffer Time
df_inventory['Raw_Buffer'] = df_inventory['Days of supply (current)'] - df_inventory['Average Lead time (days)']
df_inventory['Buffer Time (days)'] = df_inventory['Raw_Buffer'].clip(lower=0)

# Map Vendor performance → numeric score
perf_map = {'Excellent':5, 'Good':4, 'Average':3, 'Developing':2, 'Poor':1}
df_vendors['Vendor_Score'] = df_vendors['Vendor performance'].map(perf_map).fillna(3)

# Latest tariff per exporting country (for imports into Canada)
df_tariffs_can = df_tariffs[df_tariffs['Country Importing'] == 'Canada'].copy()
df_tariffs_can['Year'] = df_tariffs_can['Year'].astype(int)
df_latest_tariff = (
    df_tariffs_can.groupby(['Country Exporting','Year'], as_index=False)['Avg. Duty (%)']
    .mean().sort_values(['Country Exporting','Year'], ascending=[True, False])
    .drop_duplicates(subset=['Country Exporting'], keep='first')
    .rename(columns={'Country Exporting':'Country','Avg. Duty (%)':'Current_Tariff_%'})
    [['Country','Current_Tariff_%']]
)

# Define categories and their candidate countries
LOW_TOLERANCE_CATEGORIES = ['Distribution Transf','Switchgear','Wire And Cable','Aux Elec Equip']
CANDIDATE_COUNTRIES = {
    'Distribution Transf': ['South Korea', 'USA', 'Canada'],
    'Switchgear':         ['Germany', 'USA', 'Canada'],
    'Wire And Cable':     ['USA', 'Canada'],
    'Aux Elec Equip':     ['Germany', 'USA', 'Canada']
}

# 1. Build a monthly aggregated dataset of (category, country, month) → avg_buffer, avg_vendor_score, tariff_pct
records = []
for cat in LOW_TOLERANCE_CATEGORIES:
    for country in CANDIDATE_COUNTRIES[cat]:
        df_sub = df_inventory[
            (df_inventory['Category NAME'] == cat) & 
            (df_inventory['Country Exporting'] == country)
        ].copy()
        if df_sub.empty:
            continue
        # Merge vendor score
        df_sub = df_sub.merge(df_vendors[['VendorNumber','Vendor_Score']], on='VendorNumber', how='left')
        # Find tariff %
        trow = df_latest_tariff[df_latest_tariff['Country'] == country]
        tariff_pct = float(trow['Current_Tariff_%']) if not trow.empty else 0.0
        # Create month column
        df_sub['month'] = pd.to_datetime(df_sub['START DATE']).dt.to_period('M').dt.to_timestamp()
        grouped = df_sub.groupby('month').agg(
            avg_buffer=('Buffer Time (days)','mean'),
            avg_vendor_score=('Vendor_Score','mean')
        ).reset_index().sort_values('month')
        for _, row in grouped.iterrows():
            records.append({
                'category': cat,
                'country':  country,
                'month':    row['month'],
                'avg_buffer':       row['avg_buffer'],
                'avg_vendor_score': row['avg_vendor_score'],
                'tariff_pct':       tariff_pct
            })

df_monthly = pd.DataFrame(records)
df_monthly.dropna(subset=['avg_buffer','avg_vendor_score'], inplace=True)

# 2. Create features: time_index, year, month_num, and one-hot for category & country
df_monthly['year'] = df_monthly['month'].dt.year
df_monthly['month_num'] = df_monthly['month'].dt.month
df_monthly['time_index'] = (df_monthly['year'] - df_monthly['year'].min()) * 12 + (df_monthly['month_num'] - 1)

# One-hot encoding (use sparse_output=False instead of sparse=False)
ohe = OneHotEncoder(sparse_output=False)
encoded = ohe.fit_transform(df_monthly[['category','country']])
feature_cols = ohe.get_feature_names_out(['category','country'])
df_encoded = pd.DataFrame(encoded, columns=feature_cols, index=df_monthly.index)

df_features = pd.concat([
    df_monthly[['time_index','avg_vendor_score','tariff_pct']],
    df_encoded
], axis=1)
y = df_monthly['avg_buffer']

# 3. Split into train/test: last 6 months as test
max_time = df_features['time_index'].max()
test_threshold = max_time - 5  # last 6 points inclusive
train_mask = df_features['time_index'] < test_threshold
test_mask  = ~train_mask

X_train, y_train = df_features[train_mask], y[train_mask]
X_test,  y_test  = df_features[test_mask],  y[test_mask]

# 4. Train a Random Forest regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate on test
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Test RMSE on average buffer: {rmse:.2f} days\n")

# Plot actual vs predicted on test data
plt.figure(figsize=(10,4))
plt.plot(df_monthly.loc[test_mask,'month'], y_test, label='Actual', marker='o')
plt.plot(df_monthly.loc[test_mask,'month'], y_pred, label='Predicted', marker='o')
plt.title('RandomForest: Avg Buffer Prediction (Test Period)')
plt.xlabel('Month')
plt.ylabel('Avg Buffer Time (days)')
plt.legend()
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# 6. Forecast next 6 months for each category-country using the most recent vendor score and tariff
future_records = []
future_months = 6
recent_time = df_monthly['time_index'].max()

for cat in LOW_TOLERANCE_CATEGORIES:
    for country in CANDIDATE_COUNTRIES[cat]:
        subset = df_monthly[(df_monthly['category']==cat) & (df_monthly['country']==country)]
        if subset.empty:
            continue
        latest_month = subset['month'].max()
        latest_vendor_score = subset['avg_vendor_score'].iloc[-1]
        latest_tariff = subset['tariff_pct'].iloc[-1]
        for i in range(1, future_months+1):
            future_date = latest_month + pd.DateOffset(months=i)
            future_time_idx = recent_time + i
            base_row = {
                'time_index': future_time_idx,
                'avg_vendor_score': latest_vendor_score,
                'tariff_pct': latest_tariff
            }
            cat_ctry_encoded = pd.DataFrame(ohe.transform([[cat, country]]), columns=feature_cols)
            combined = pd.concat([pd.DataFrame([base_row]), cat_ctry_encoded.reset_index(drop=True)], axis=1)
            future_records.append({
                'category': cat,
                'country':  country,
                'month':    future_date,
                'features': combined
            })

# Build DataFrame for future features & predictions
future_dfs = []
for rec in future_records:
    df_feat = rec['features']
    df_feat['category'] = rec['category']
    df_feat['country']  = rec['country']
    df_feat['month']    = rec['month']
    future_dfs.append(df_feat)

df_future = pd.concat(future_dfs, ignore_index=True)
X_future = df_future.drop(['category','country','month'], axis=1)
df_future['predicted_buffer'] = model.predict(X_future)

# Display forecast
print("Forecasted Avg Buffer Time (days) for Next 6 Months:\n")
print(df_future[['category','country','month','predicted_buffer']])

# Plot forecast for each category-country
for cat in LOW_TOLERANCE_CATEGORIES:
    for country in CANDIDATE_COUNTRIES[cat]:
        df_plot = df_future[(df_future['category']==cat) & (df_future['country']==country)]
        if df_plot.empty:
            continue
        plt.figure(figsize=(8,3))
        plt.plot(df_plot['month'], df_plot['predicted_buffer'], marker='o')
        plt.title(f"{cat} – {country} Forecasted Buffer (Next 6 M)")
        plt.xlabel('Month')
        plt.ylabel('Predicted Buffer Time (days)')
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.show()
