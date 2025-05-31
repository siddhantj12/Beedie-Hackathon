# category_buffer_time_comparison_final_USA.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

# ——————————————————————————————————————————
# 0. CONFIGURATION: adjust to your local path
# ——————————————————————————————————————————
FILE_PATH = '/Users/siddhant/Desktop/Beedie Hackathon/EVERYTHING.xlsx'
FUTURE_US_TARIFF = 25.0  # assume 25% tariff on USA → Canada

# Exact “Category NAME” values from your file
LOW_TOLERANCE_CATEGORIES = [
    'Distribution Transf',
    'Switchgear',
    'Wire And Cable',
    'Aux Elec Equip'
]

# Candidate countries for each category (using "USA" instead of full name)
CANDIDATE_ALTERNATIVES = {
    'Distribution Transf': ['South Korea', 'USA', 'Canada'],
    'Switchgear':         ['Germany',     'USA', 'Canada'],
    'Wire And Cable':     ['USA', 'Canada'],
    'Aux Elec Equip':     ['Germany',     'USA', 'Canada']
}

US_COUNTRY_NAME = 'USA'
CANADA_NAME    = 'Canada'


# ——————————————————————————————————————————
# 1. LOAD DATA
# ——————————————————————————————————————————
try:
    df_inventory = pd.read_excel(FILE_PATH, sheet_name='Vendors Inventory')
    df_vendors   = pd.read_excel(FILE_PATH, sheet_name='BCH All Vendor Data')
    df_tariffs   = pd.read_excel(FILE_PATH, sheet_name='Tariffs Rates by Country')
except FileNotFoundError:
    print(f"\nERROR: File not found at '{FILE_PATH}'. Please verify the path.\n")
    sys.exit(1)
except Exception as e:
    print("\nERROR: Failed to read the Excel file:\n", str(e))
    sys.exit(1)


# ——————————————————————————————————————————
# 2. VERIFY REQUIRED COLUMNS & COMPUTE BUFFER TIME
# ——————————————————————————————————————————
required_cols = [
    'Days of supply (current)',
    'Average Lead time (days)',
    'Category NAME',
    'Country Exporting',
    'VendorNumber'
]
missing_cols = [c for c in required_cols if c not in df_inventory.columns]
if missing_cols:
    print("\nERROR: Missing columns in 'Vendors Inventory':", missing_cols)
    print("Available columns are:\n", df_inventory.columns.tolist())
    sys.exit(1)

# Compute raw buffer: Days of supply – Lead time
df_inventory['Raw_Buffer'] = (
    df_inventory['Days of supply (current)'] 
    - df_inventory['Average Lead time (days)']
)

# Clip negative buffers to zero
df_inventory['Buffer Time (days)'] = df_inventory['Raw_Buffer'].clip(lower=0)

# Drop any rows where Buffer Time is not finite
bad_buffer = ~np.isfinite(df_inventory['Buffer Time (days)'])
if bad_buffer.any():
    print(f"\n⚠️  Warning: Dropping {bad_buffer.sum()} rows with invalid Buffer Time.\n")
    df_inventory = df_inventory[~bad_buffer].copy()


# ——————————————————————————————————————————
# 3. PREPARE VENDOR SCORE
# ——————————————————————————————————————————
perf_map = {
    'Excellent': 5,
    'Good':      4,
    'Average':   3,
    'Developing':2,
    'Poor':      1
}
df_vendors['Vendor_Score'] = df_vendors['Vendor performance'].map(perf_map).fillna(3)


# ——————————————————————————————————————————
# 4. BUILD “MOST RECENT TARIFF BY COUNTRY” TABLE
# ——————————————————————————————————————————
# Filter to only rows where Canada is importing
df_tariffs_canada = df_tariffs[df_tariffs['Country Importing'] == CANADA_NAME].copy()
df_tariffs_canada['Year'] = df_tariffs_canada['Year'].astype(int)

# For each exporting country, pick the highest Year’s Avg. Duty (%)
df_latest_tariff = (
    df_tariffs_canada
    .groupby(['Country Exporting','Year'], as_index=False)['Avg. Duty (%)']
    .mean()  # average across HS codes in that year
    .sort_values(['Country Exporting','Year'], ascending=[True, False])
    .drop_duplicates(subset=['Country Exporting'], keep='first')
    .rename(columns={'Country Exporting':'Country','Avg. Duty (%)':'Current_Tariff_%'})
    .loc[:, ['Country','Current_Tariff_%']]
)
# Any exporting country not in this → tariff = 0 (via fillna later)


# ——————————————————————————————————————————
# 5. PROCESS EACH CATEGORY (USING COMPUTED BUFFER TIME)
# ——————————————————————————————————————————
results = []
plots_data = {}

for category in LOW_TOLERANCE_CATEGORIES:
    # 5.1. Filter inventory for this category
    df_cat = df_inventory[df_inventory['Category NAME'] == category].copy()
    if df_cat.empty:
        print(f"⚠️  Warning: No inventory data found for category '{category}'. Skipping.\n")
        continue

    # 5.2. Merge vendor score
    df_cat = df_cat.merge(
        df_vendors[['VendorNumber','Vendor_Score']],
        on='VendorNumber',
        how='left'
    )

    # 5.3. Group by exporting country to compute:
    #      - Avg_Buffer_Time   = mean of df_cat['Buffer Time (days)']
    #      - Avg_Vendor_Score  = mean of df_cat['Vendor_Score']
    df_grouped = (
        df_cat
        .groupby('Country Exporting', as_index=False)
        .agg(
            Avg_Buffer_Time = ('Buffer Time (days)', 'mean'),
            Avg_Vendor_Score = ('Vendor_Score',       'mean'),
            Num_Vendors      = ('VendorNumber',      'nunique'),
            Num_Shipments    = ('Buffer Time (days)', 'count')
        )
        .rename(columns={'Country Exporting':'Country'})
    )

    # 5.4. Merge in Current_Tariff_% for each country
    df_grouped = df_grouped.merge(
        df_latest_tariff,
        on='Country',
        how='left'
    )
    df_grouped['Current_Tariff_%'] = df_grouped['Current_Tariff_%'].fillna(0.0)

    # 5.5. Override USA tariff → FUTURE_US_TARIFF
    df_grouped.loc[
        df_grouped['Country'] == US_COUNTRY_NAME,
        'Current_Tariff_%'
    ] = FUTURE_US_TARIFF

    # 5.6. Compute Effective_Buffer_Index = Avg_Buffer_Time × (1 + Tariff%/100)
    df_grouped['Effective_Buffer_Index'] = (
        df_grouped['Avg_Buffer_Time'] * (1 + df_grouped['Current_Tariff_%'] / 100.0)
    )

    # 5.7. Keep only candidate countries for this category
    candidates = CANDIDATE_ALTERNATIVES.get(category, [])
    df_grouped = df_grouped[df_grouped['Country'].isin(candidates)].copy()
    if df_grouped.empty:
        print(f"⚠️  Warning: No candidate countries found for '{category}'. Skipping.\n")
        continue

    # 5.8. Rank by Effective_Buffer_Index (ascending), tiebreak on Avg_Vendor_Score (descending)
    df_grouped['Rank_Buffer'] = df_grouped['Effective_Buffer_Index'].rank(method='dense', ascending=True)
    df_grouped['Rank_Vendor'] = df_grouped['Avg_Vendor_Score'].rank(method='dense', ascending=False)
    df_grouped['Combined_Rank'] = df_grouped['Rank_Buffer'] + df_grouped['Rank_Vendor'] * 0.01

    df_grouped = df_grouped.sort_values(
        by=['Combined_Rank','Avg_Vendor_Score','Avg_Buffer_Time'],
        ascending=[True, False, True]
    ).reset_index(drop=True)

    recommended = df_grouped.loc[0, 'Country']

    results.append({
        'Category': category,
        'Recommended_Country': recommended,
        'Table': df_grouped.copy()
    })
    plots_data[category] = df_grouped.copy()


# ——————————————————————————————————————————
# 6. OUTPUT RESULTS & SAVE PLOTS
# ——————————————————————————————————————————
for entry in results:
    cat = entry['Category']
    rec = entry['Recommended_Country']
    df_out = entry['Table']

    print(f"\n\n=== Category: {cat} ===")
    print(f"Recommended Sourcing Country → {rec}\n")
    print(
        df_out[['Country','Avg_Buffer_Time','Avg_Vendor_Score','Current_Tariff_%','Effective_Buffer_Index']]
        .round({
            'Avg_Buffer_Time':        1,
            'Avg_Vendor_Score':       2,
            'Current_Tariff_%':       1,
            'Effective_Buffer_Index': 1
        })
        .to_string(index=False)
    )

    # A) Avg Buffer Time bar chart
    plt.figure(figsize=(6,4))
    x = df_out['Country']
    y = df_out['Avg_Buffer_Time']
    bars = plt.bar(x, y, color='tab:blue', alpha=0.7)
    for i, v in enumerate(y):
        plt.text(i, v + (max(y)*0.02), f"{v:.1f}", ha='center', va='bottom', fontsize=9)
    plt.ylabel('Avg Buffer Time (days)')
    plt.title(f"{cat} – Avg Buffer Time by Country")
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(f"{cat.replace(' ','_')}_AvgBufferTime.png")
    plt.close()

    # B) Avg Vendor Score bar chart
    plt.figure(figsize=(6,4))
    y2 = df_out['Avg_Vendor_Score']
    bars = plt.bar(x, y2, color='tab:green', alpha=0.7)
    for i, v in enumerate(y2):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center', va='bottom', fontsize=9)
    plt.ylabel('Avg Vendor Score (1–5)')
    plt.ylim(0,5.5)
    plt.title(f"{cat} – Avg Vendor Score by Country")
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(f"{cat.replace(' ','_')}_AvgVendorScore.png")
    plt.close()

    # C) Current Tariff bar chart
    plt.figure(figsize=(6,4))
    y3 = df_out['Current_Tariff_%']
    bars = plt.bar(x, y3, color='tab:red', alpha=0.7)
    for i, v in enumerate(y3):
        offset = (max(y3)*0.02) if max(y3)>0 else 0.5
        plt.text(i, v + offset, f"{v:.1f}%", ha='center', va='bottom', fontsize=9)
    plt.ylabel('Tariff Rate (%)')
    plt.title(f"{cat} – Current Tariff (%) by Country")
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(f"{cat.replace(' ','_')}_CurrentTariff.png")
    plt.close()

    # D) Effective Buffer Index bar chart
    plt.figure(figsize=(6,4))
    y4 = df_out['Effective_Buffer_Index']
    bars = plt.bar(x, y4, color='tab:purple', alpha=0.7)
    for i, v in enumerate(y4):
        plt.text(i, v + (max(y4)*0.02), f"{v:.1f}", ha='center', va='bottom', fontsize=9)
    plt.ylabel('Effective Buffer Index')
    plt.title(f"{cat} – Effective Buffer Index by Country")
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(f"{cat.replace(' ','_')}_EffBufferIndex.png")
    plt.close()


# ——————————————————————————————————————————
# 7. FINAL SUMMARY
# ——————————————————————————————————————————
print("\n\n#############################")
print("     FINAL RECOMMENDATIONS  ")
print("#############################")
for entry in results:
    print(f"- {entry['Category']}  →  {entry['Recommended_Country']}")
