import pandas as pd

# ——————————————————————————————————————————
# 1. CONFIGURATION
# ——————————————————————————————————————————
FILE_PATH = '/Users/siddhant/Desktop/Beedie Hackathon/EVERYTHING.xlsx'

# Sheet names in your Excel file:
VENDOR_SHEET    = 'BCH All Vendor Data'
INVENTORY_SHEET = 'Vendors Inventory'

# How many top countries to show
TOP_N = 10

# Mapping categorical performance to numeric scores:
#   “Excellent” → 5, “Good” → 4, “Average” → 3, “Developing” → 2, “Poor” → 1
PERFORMANCE_MAP = {
    'Excellent': 5,
    'Good':      4,
    'Average':   3,
    'Developing':2,
    'Poor':      1
}


# ——————————————————————————————————————————
# 2. LOAD SHEETS AND DISPLAY BASIC INFO
# ——————————————————————————————————————————
# 2.1 Read “All Vendor Data” (contains Vendor performance)
df_vendor = pd.read_excel(FILE_PATH, sheet_name=VENDOR_SHEET)

# 2.2 Read “Vendors Inventory” (contains Country Exporting and VendorNumber)
df_inventory = pd.read_excel(FILE_PATH, sheet_name=INVENTORY_SHEET)

# Sanity check: ensure required columns exist
for col in ['VendorNumber', 'Vendor performance']:
    if col not in df_vendor.columns:
        raise KeyError(f"'{col}' not found in sheet '{VENDOR_SHEET}'")

for col in ['VendorNumber', 'Country Exporting']:
    if col not in df_inventory.columns:
        raise KeyError(f"'{col}' not found in sheet '{INVENTORY_SHEET}'")



# ——————————————————————————————————————————
# 3. PREPARE AND MERGE DATA
# ——————————————————————————————————————————
# 3.1 Keep only necessary columns from each sheet
df_vendor_sub = df_vendor[['VendorNumber', 'Vendor performance']].copy()
df_inventory_sub = df_inventory[['VendorNumber', 'Country Exporting']].copy()

# 3.2 Merge on VendorNumber (inner join → only vendors present in both sheets)
df_merged = pd.merge(
    df_vendor_sub,
    df_inventory_sub,
    on='VendorNumber',
    how='inner'
)

# 3.3 Drop any rows with missing or unrecognized performance labels
df_merged = df_merged.dropna(subset=['Vendor performance', 'Country Exporting'])
df_merged = df_merged[df_merged['Vendor performance'].isin(PERFORMANCE_MAP.keys())]


# ——————————————————————————————————————————
# 4. MAP CATEGORIES TO NUMERIC SCORES
# ——————————————————————————————————————————
df_merged['Performance Score'] = df_merged['Vendor performance'].map(PERFORMANCE_MAP)

# If there are any vendor performances not in PERFORMANCE_MAP, they will become NaN.
# We’ve already filtered out unrecognized labels above.


# ——————————————————————————————————————————
# 5. AGGREGATE BY COUNTRY
# ——————————————————————————————————————————
# 5.1 Compute average performance score and vendor count per exporting country
df_country_perf = (
    df_merged
    .groupby('Country Exporting', as_index=False)
    .agg(
        Avg_Performance=('Performance Score', 'mean'),
        Vendor_Count=('VendorNumber', 'nunique')
    )
)

# 5.2 Sort by Avg_Performance descending (higher = better)
df_country_perf = df_country_perf.sort_values(by='Avg_Performance', ascending=False)


# ——————————————————————————————————————————
# 6. DISPLAY RESULTS
# ——————————————————————————————————————————
print("\n" + "="*60)
print("VENDOR PERFORMANCE BY EXPORTING COUNTRY")
print("="*60 + "\n")

print(f"Top {TOP_N} exporting countries (highest average vendor performance):\n")
print(
    df_country_perf
    .head(TOP_N)
    .assign(
        Avg_Performance=lambda df: df['Avg_Performance'].round(2)
    )
    .to_string(index=False)
)

print("\n" + "-"*60 + "\n")

print(f"Bottom {TOP_N} exporting countries (lowest average vendor performance):\n")
print(
    df_country_perf
    .tail(TOP_N)
    .sort_values(by='Avg_Performance', ascending=True)
    .assign(
        Avg_Performance=lambda df: df['Avg_Performance'].round(2)
    )
    .to_string(index=False)
)

print("\n" + "="*60 + "\n")

# Save the full country‐level performance table to CSV
OUT_CSV = 'country_vendor_performance_summary.csv'
df_country_perf.to_csv(OUT_CSV, index=False)
print(f"Full exporting‐country vendor performance saved to: {OUT_CSV}\n")
