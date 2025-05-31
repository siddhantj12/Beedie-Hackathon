import pandas as pd
import matplotlib.pyplot as plt

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
df_vendor_sub    = df_vendor[['VendorNumber', 'Vendor performance']].copy()
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


# ——————————————————————————————————————————
# 5. AGGREGATE BY COUNTRY
# ——————————————————————————————————————————
# 5.1 Compute average performance score and vendor count per exporting country
df_country_perf = (
    df_merged
    .groupby('Country Exporting', as_index=False)
    .agg(
        Avg_Performance=('Performance Score', 'mean'),
        Vendor_Count   =('VendorNumber', 'nunique')
    )
)

# 5.2 Round and sort by Avg_Performance descending (higher = better)
df_country_perf['Avg_Performance'] = df_country_perf['Avg_Performance'].round(2)
df_country_perf_sorted = df_country_perf.sort_values(
    by='Avg_Performance',
    ascending=False
)


# ——————————————————————————————————————————
# 6. DISPLAY RESULTS
# ——————————————————————————————————————————
print("\n" + "="*60)
print("VENDOR PERFORMANCE BY EXPORTING COUNTRY")
print("="*60 + "\n")

print(f"Top {TOP_N} exporting countries (highest average vendor performance):\n")
print(
    df_country_perf_sorted
    .head(TOP_N)
    .to_string(index=False)
)

print("\n" + "-"*60 + "\n")

print(f"Bottom {TOP_N} exporting countries (lowest average vendor performance):\n")
print(
    df_country_perf_sorted
    .tail(TOP_N)
    .sort_values(by='Avg_Performance', ascending=True)
    .to_string(index=False)
)

print("\n" + "="*60 + "\n")

# Save the full country‐level performance table to CSV
OUT_CSV = 'country_vendor_performance_summary.csv'
df_country_perf_sorted.to_csv(OUT_CSV, index=False)
print(f"Full exporting‐country vendor performance saved to: {OUT_CSV}\n")


# ——————————————————————————————————————————
# 7. VISUALIZE RESULTS
# ——————————————————————————————————————————

# 7.1 VISUALIZE TOP N COUNTRIES (VERTICAL BAR CHART)
df_top = df_country_perf_sorted.head(TOP_N)

plt.figure(figsize=(10, 6))
bars = plt.bar(
    df_top['Country Exporting'],
    df_top['Avg_Performance'],
    color='skyblue',
    edgecolor='black',
    linewidth=0.8
)

# Annotate each bar with its value
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.05,
        f"{height:.2f}",
        ha='center',
        va='bottom',
        fontsize=9
    )

plt.title(f"Top {TOP_N} Exporting Countries by Avg. Vendor Performance", fontsize=14, pad=12)
plt.xlabel("Exporting Country", fontsize=12)
plt.ylabel("Avg. Vendor Performance Score", fontsize=12)
plt.ylim(0, 5.5)
plt.xticks(rotation=35, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


# 7.2 VISUALIZE ALL COUNTRIES (HORIZONTAL BAR CHART)
plt.figure(figsize=(12, 8))
bars_h = plt.barh(
    df_country_perf_sorted['Country Exporting'],
    df_country_perf_sorted['Avg_Performance'],
    color='lightgreen',
    edgecolor='black',
    linewidth=0.8
)

# Annotate each bar on the right side
for bar in bars_h:
    width = bar.get_width()
    plt.text(
        width + 0.03,
        bar.get_y() + bar.get_height() / 2,
        f"{width:.2f}",
        ha='left',
        va='center',
        fontsize=8
    )

plt.title("Average Vendor Performance Score by Exporting Country", fontsize=14, pad=12)
plt.xlabel("Avg. Vendor Performance Score", fontsize=12)
plt.ylabel("Exporting Country", fontsize=12)
plt.xlim(0, 5.5)
plt.xticks(fontsize=10)
plt.yticks(fontsize=9)
plt.gca().invert_yaxis()  # Highest performers at the top
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
