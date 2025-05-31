import pandas as pd
import matplotlib.pyplot as plt

# ————————————————————————————————
# 1. CONFIGURATION
# ————————————————————————————————
FILE_PATH      = '/Users/siddhant/Desktop/Beedie Hackathon/EVERYTHING.xlsx'
SUPPLY_SHEET   = 'Supply chain risk & spend'
INVENTORY_SHEET = 'Vendors Inventory'

FOCUS_COUNTRIES = ['India', 'Japan']           # Countries to analyze
RISK_PRIORITY   = ['Low', 'Med', 'High']       # In descending order of “safer” risk tolerance
TOP_N           = 10                           # Number of categories to highlight if needed

# ————————————————————————————————
# 2. LOAD DATA
# ————————————————————————————————
df_supply    = pd.read_excel(FILE_PATH, sheet_name=SUPPLY_SHEET)
df_inventory = pd.read_excel(FILE_PATH, sheet_name=INVENTORY_SHEET)

# ————————————————————————————————
# 3. NORMALIZE CATEGORY NAMES (OPTIONAL BUT HELPFUL)
# ————————————————————————————————
# Strip whitespace and unify case in both DataFrames’ category columns
df_supply['CATEGORY_norm']       = df_supply['CATEGORY'].str.strip().str.lower()
df_supply['Risk_norm']           = df_supply['Risk Tolerance of the category'].str.strip().str.capitalize()

df_inventory['Category_NAME_norm'] = df_inventory['Category NAME'].str.strip().str.lower()
df_inventory['Country_norm']      = df_inventory['Country Exporting'].str.strip()

# ————————————————————————————————
# 4. DETERMINE TARGET RISK LEVEL FOR ANALYSIS
# ————————————————————————————————
# Find which risk level (Low, then Med, then High) yields any matching categories for both India & Japan
selected_risk_level = None

for risk in RISK_PRIORITY:
    risk_norm = risk.strip().lower()
    # Identify categories in supply sheet having that risk
    categories_at_risk = df_supply[df_supply['Risk_norm'] == risk]['CATEGORY_norm'].unique()
    # Filter inventory for those categories
    inv_lowrisk = df_inventory[df_inventory['Category_NAME_norm'].isin(categories_at_risk)]
    # Check if at least one of the focus countries appears
    supplied_countries = inv_lowrisk['Country_norm'].unique()
    if any(country in supplied_countries for country in FOCUS_COUNTRIES):
        selected_risk_level = risk
        break

if selected_risk_level is None:
    raise ValueError("None of the risk levels (Low/Med/High) have any categories supplied by India or Japan.")

print(f"\nSelected Risk Level for Analysis: {selected_risk_level}\n")

# ————————————————————————————————
# 5. FILTER FOR SELECTED RISK CATEGORIES
# ————————————————————————————————
risk_norm_selected = selected_risk_level.strip().lower()
lowrisk_categories_norm = df_supply[df_supply['Risk_norm'] == selected_risk_level]['CATEGORY_norm'].unique()

df_focus_inv = df_inventory[df_inventory['Category_NAME_norm'].isin(lowrisk_categories_norm)]
df_focus_inv = df_focus_inv[df_focus_inv['Country_norm'].isin(FOCUS_COUNTRIES)].copy()

# Attach the original CATEGORY and risk
df_focus_inv = pd.merge(
    df_focus_inv,
    df_supply[['CATEGORY_norm', 'CATEGORY', 'Annual spend', 'Risk_norm']],
    left_on='Category_NAME_norm',
    right_on='CATEGORY_norm',
    how='left'
)

# ————————————————————————————————
# 6. CALCULATE METRICS PER COUNTRY & CATEGORY
# ————————————————————————————————
# Group by Country_norm and CATEGORY (original name), aggregate:
#  - Average Lead Time (days)
#  - Total Spend (sum of Annual spend)
#  - Number of Unique Vendors
df_metrics = df_focus_inv.groupby(
    ['Country_norm', 'CATEGORY'], as_index=False
).agg(
    Avg_Lead_Time=('Average Lead time (days)', 'mean'),
    Total_Spend=('Annual spend', 'sum'),
    Num_Vendors=('VendorNumber', 'nunique')
)

# Round numeric metrics
df_metrics['Avg_Lead_Time'] = df_metrics['Avg_Lead_Time'].round(2)
df_metrics['Total_Spend']   = df_metrics['Total_Spend'].round(2)

# Separate dataframes for India and Japan
df_india = df_metrics[df_metrics['Country_norm'] == 'India']
df_japan = df_metrics[df_metrics['Country_norm'] == 'Japan']

# If any country has no data at this risk level, report and exit gracefully
if df_india.empty:
    print("No categories at risk level “Low” (or selected) are supplied by India.")
if df_japan.empty:
    print("No categories at risk level “Low” (or selected) are supplied by Japan.")

# ————————————————————————————————
# 7. VISUALIZATIONS FOR INDIA & JAPAN
# ————————————————————————————————
def plot_country_metrics(df_country, country_name):
    """
    Given df_country containing columns [ 'CATEGORY', 'Avg_Lead_Time', 'Total_Spend', 'Num_Vendors' ],
    produce:
     1. Bar chart of Avg Lead Time by Category
     2. Bar chart of Total Spend by Category
     3. Scatter plot: Lead Time vs Total Spend (bubble size by Num_Vendors)
    """
    if df_country.empty:
        return

    # 7.1 Bar Chart: Avg Lead Time by Category
    plt.figure(figsize=(10, 5))
    bars = plt.bar(
        df_country['CATEGORY'],
        df_country['Avg_Lead_Time'],
        color='skyblue',
        edgecolor='black',
        linewidth=0.8
    )
    plt.title(f"{country_name}: Avg Lead Time by Category ({selected_risk_level} Risk)", fontsize=14, pad=12)
    plt.xlabel("Category", fontsize=12)
    plt.ylabel("Avg Lead Time (days)", fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    max_lt = df_country['Avg_Lead_Time'].max()
    plt.ylim(0, max_lt + 5 if not pd.isna(max_lt) else 10)
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    for bar in bars:
        h = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,
            h + 0.3,
            f"{h:.2f}",
            ha='center',
            va='bottom',
            fontsize=9
        )
    plt.tight_layout()
    plt.show()

    # 7.2 Bar Chart: Total Spend by Category
    plt.figure(figsize=(10, 5))
    bars2 = plt.bar(
        df_country['CATEGORY'],
        df_country['Total_Spend'],
        color='lightgreen',
        edgecolor='black',
        linewidth=0.8
    )
    plt.title(f"{country_name}: Total Spend by Category ({selected_risk_level} Risk)", fontsize=14, pad=12)
    plt.xlabel("Category", fontsize=12)
    plt.ylabel("Total Annual Spend (USD)", fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    max_sp = df_country['Total_Spend'].max()
    plt.ylim(0, max_sp * 1.1 if not pd.isna(max_sp) else 10)
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    for bar in bars2:
        h = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,
            h + max_sp * 0.02 if not pd.isna(max_sp) else h + 1,
            f"{h:.2f}",
            ha='center',
            va='bottom',
            fontsize=9
        )
    plt.tight_layout()
    plt.show()

    # 7.3 Scatter Plot: Lead Time vs. Total Spend (bubble size by number of vendors)
    plt.figure(figsize=(8, 6))
    sizes = (df_country['Num_Vendors'] / df_country['Num_Vendors'].max()) * 300  # scale bubble sizes
    scatter = plt.scatter(
        df_country['Avg_Lead_Time'],
        df_country['Total_Spend'],
        s=sizes,
        c='coral',
        edgecolor='black',
        alpha=0.7
    )
    for _, row in df_country.iterrows():
        plt.text(
            row['Avg_Lead_Time'] + 0.2,
            row['Total_Spend'] + max(df_country['Total_Spend']) * 0.01,
            row['CATEGORY'],
            fontsize=9
        )
    plt.title(f"{country_name}: Lead Time vs Spend ({selected_risk_level} Risk)", fontsize=14, pad=12)
    plt.xlabel("Avg Lead Time (days)", fontsize=12)
    plt.ylabel("Total Annual Spend (USD)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()


# Plot for India
plot_country_metrics(df_india, "India")

# Plot for Japan
plot_country_metrics(df_japan, "Japan")

# ————————————————————————————————
# 8. COMBINED COMPARISON: INDIA vs. JAPAN
# ————————————————————————————————
# Merge India and Japan on CATEGORY to compare side by side
if not df_india.empty and not df_japan.empty:
    df_compare = pd.merge(
        df_india[['CATEGORY', 'Avg_Lead_Time', 'Total_Spend']],
        df_japan[['CATEGORY', 'Avg_Lead_Time', 'Total_Spend']],
        on='CATEGORY',
        suffixes=('_India', '_Japan'),
        how='inner'
    )
    if not df_compare.empty:
        # 8.1 Side-by-side Bar Chart: Avg Lead Time
        x = range(len(df_compare))
        width = 0.35
        plt.figure(figsize=(12, 6))
        plt.bar(
            [xi - width/2 for xi in x],
            df_compare['Avg_Lead_Time_India'],
            width,
            label='India',
            color='skyblue',
            edgecolor='black'
        )
        plt.bar(
            [xi + width/2 for xi in x],
            df_compare['Avg_Lead_Time_Japan'],
            width,
            label='Japan',
            color='lightcoral',
            edgecolor='black'
        )
        plt.xlabel("Category", fontsize=12)
        plt.ylabel("Avg Lead Time (days)", fontsize=12)
        plt.title(f"Comparison of Avg Lead Time by Category ({selected_risk_level} Risk)", fontsize=14, pad=12)
        plt.xticks(x, df_compare['CATEGORY'], rotation=45, ha='right', fontsize=10)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.show()

        # 8.2 Side-by-side Bar Chart: Total Spend
        plt.figure(figsize=(12, 6))
        plt.bar(
            [xi - width/2 for xi in x],
            df_compare['Total_Spend_India'],
            width,
            label='India',
            color='lightgreen',
            edgecolor='black'
        )
        plt.bar(
            [xi + width/2 for xi in x],
            df_compare['Total_Spend_Japan'],
            width,
            label='Japan',
            color='gold',
            edgecolor='black'
        )
        plt.xlabel("Category", fontsize=12)
        plt.ylabel("Total Annual Spend (USD)", fontsize=12)
        plt.title(f"Comparison of Total Spend by Category ({selected_risk_level} Risk)", fontsize=14, pad=12)
        plt.xticks(x, df_compare['CATEGORY'], rotation=45, ha='right', fontsize=10)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.show()
    else:
        print("No overlapping low-risk categories exist for both India and Japan.")

# ————————————————————————————————
# 9. OUTPUT DETAILED TABLES
# ————————————————————————————————
print("\n=== Detailed Metrics for India ===\n")
print(df_india.to_string(index=False))

print("\n=== Detailed Metrics for Japan ===\n")
print(df_japan.to_string(index=False))

# Save to CSV for further exploration
df_india.to_csv('India_lowrisk_metrics.csv', index=False)
df_japan.to_csv('Japan_lowrisk_metrics.csv', index=False)
