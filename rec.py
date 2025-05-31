import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ——————————————————————————————————————————
# 0. CONFIGURATION: Update FILE_PATH if EVERYTHING.xlsx is elsewhere
# ——————————————————————————————————————————
FILE_PATH             = '/Users/siddhant/Desktop/Beedie Hackathon/EVERYTHING.xlsx'

# Sheet names in EVERYTHING.xlsx (must match exactly)
SHEET_RISK_SPEND      = 'Supply chain risk & spend'
SHEET_IMPORT_VALUES   = 'Import Values'
SHEET_HS_MAPPING      = 'HS Code Definitions'
SHEET_VENDOR_DATA     = 'BCH All Vendor Data'
SHEET_INVENTORY       = 'Vendors Inventory'

# Tariff simulation parameters
TARIFF_SOURCE         = 'United States'
IMPORTER              = 'Canada'
TARIFF_RATES          = [0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5]  # 0%, 10%, 20%, 25%, 30%, 40%, 50%
TOP_LOW_N             = 5  # Number of top low-risk categories to focus on


# ——————————————————————————————————————————
# 1. LOAD & PREPARE “Supply chain risk & spend” DATA
# ——————————————————————————————————————————
df_risk = pd.read_excel(FILE_PATH, sheet_name=SHEET_RISK_SPEND)

# Ensure the expected columns exist
expected_cols_risk = {
    'PORTFOLIO',
    'CATEGORY',
    'Annual spend',
    'Risk Tolerance of the category'
}
missing_risk = expected_cols_risk - set(df_risk.columns)
if missing_risk:
    raise KeyError(f"Missing columns in '{SHEET_RISK_SPEND}': {missing_risk}")

# Keep only the necessary columns and drop any rows with missing values in those columns
df_risk = df_risk[
    ['PORTFOLIO', 'CATEGORY', 'Annual spend', 'Risk Tolerance of the category']
].dropna(subset=['PORTFOLIO', 'CATEGORY', 'Annual spend', 'Risk Tolerance of the category'])

# Normalize the “Risk Tolerance of the category” column
df_risk['Risk_norm'] = (
    df_risk['Risk Tolerance of the category']
    .str.strip()
    .str.capitalize()
    .replace({'Moderate': 'Medium'})
)

# Assign a numeric “vulnerability weight” to each risk level:
weight_map = {'Low': 3, 'Medium': 2, 'High': 1}
df_risk['Vuln_Weight'] = df_risk['Risk_norm'].map(weight_map)

# Compute a “Vulnerability Score” = Annual spend × Vuln_Weight
df_risk['Vulnerability Score'] = df_risk['Annual spend'] * df_risk['Vuln_Weight']


# ——————————————————————————————————————————
# 2. VISUALIZATION #1: RISK DISTRIBUTION BY PORTFOLIO
# ——————————————————————————————————————————
# 2.1 Count how many UNIQUE categories at each risk level per portfolio
count_by_portfolio = (
    df_risk
    .groupby(['PORTFOLIO', 'Risk_norm'])['CATEGORY']
    .nunique()
    .unstack(fill_value=0)
    .reindex(columns=['Low', 'Medium', 'High'], fill_value=0)
)

# 2.2 Total annual spend at each risk level per portfolio
spend_by_portfolio = (
    df_risk
    .groupby(['PORTFOLIO', 'Risk_norm'])['Annual spend']
    .sum()
    .unstack(fill_value=0)
    .reindex(columns=['Low', 'Medium', 'High'], fill_value=0)
)

# Plot (2.1): Number of categories by risk level within each portfolio
plt.figure(figsize=(12, 6))
count_by_portfolio.plot(
    kind='bar',
    stacked=True,
    color=['#2ca02c', '#ff7f0e', '#d62728'],  # Low=green, Medium=orange, High=red
    edgecolor='black',
    figsize=(12, 6)
)
plt.title('Number of Categories by Risk Level within Each Portfolio', fontsize=16)
plt.xlabel('Portfolio', fontsize=14)
plt.ylabel('Number of Categories', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.legend(title='Risk Level', fontsize=12)
plt.tight_layout()
plt.show()

# Plot (2.2): Total annual spend by risk level within each portfolio
plt.figure(figsize=(12, 6))
spend_by_portfolio.plot(
    kind='bar',
    stacked=True,
    color=['#98df8a', '#ffbb78', '#ff9896'],  # lighter versions of Low/Medium/High
    edgecolor='black',
    figsize=(12, 6)
)
plt.title('Total Annual Spend by Risk Level within Each Portfolio', fontsize=16)
plt.xlabel('Portfolio', fontsize=14)
plt.ylabel('Annual Spend (USD)', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.legend(title='Risk Level', fontsize=12)
plt.tight_layout()
plt.show()


# ——————————————————————————————————————————
# 3. TOP 10 VULNERABLE CATEGORIES (SPEND × VULN_WEIGHT)
# ——————————————————————————————————————————
df_vuln_sorted = df_risk.sort_values(by='Vulnerability Score', ascending=False)

# 3.1 Print the Top 10 to terminal
print("\nTop 10 Most Vulnerable Categories (Annual spend × Vulnerability Weight):\n")
print(
    df_vuln_sorted[
        ['PORTFOLIO', 'CATEGORY', 'Risk_norm', 'Annual spend', 'Vulnerability Score']
    ]
    .head(10)
    .to_string(index=False)
)

# 3.2 Bar Chart of Top 10 Vulnerable Categories
top10_vuln = df_vuln_sorted.head(10)
plt.figure(figsize=(12, 5))
bars = plt.bar(
    top10_vuln['CATEGORY'],
    top10_vuln['Vulnerability Score'],
    color='tomato',
    edgecolor='black'
)
plt.title('Top 10 Vulnerable Categories (Spend × Weight)', fontsize=16)
plt.xlabel('Category', fontsize=14)
plt.ylabel('Vulnerability Score', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=12)
# Annotate each bar with its exact value
for bar in bars:
    h = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        h + top10_vuln['Vulnerability Score'].max() * 0.01,
        f"{h:.0f}",
        ha='center',
        va='bottom',
        fontsize=12
    )
plt.tight_layout()
plt.show()


# ——————————————————————————————————————————
# 4. IMPORT TRENDS & TARIFF IMPACT SIMULATION
# ——————————————————————————————————————————
# 4.1 Load the “Import Values” sheet
df_imports = pd.read_excel(FILE_PATH, sheet_name=SHEET_IMPORT_VALUES)
# Rename “HS  Code” to “HS Code” for consistency
df_imports.rename(columns={'HS  Code': 'HS Code'}, inplace=True)

# 4.2 Load HS → (Portfolio, Category) mapping
df_hsmap = pd.read_excel(FILE_PATH, sheet_name=SHEET_HS_MAPPING)
df_hsmap['HS Code'] = df_hsmap['HS Code'].astype(str)
df_imports['HS Code'] = df_imports['HS Code'].astype(str)

# 4.3 Filter imports: Canada importing from United States
df_imp_canada = df_imports[
    (df_imports['Country Importing'] == IMPORTER) &
    (df_imports['Country Exporting'] == TARIFF_SOURCE)
].copy()

# 4.4 Keep only HS codes that appear in HS → Category mapping
relevant_hs = df_hsmap['HS Code'].unique()
df_imp_canada = df_imp_canada[df_imp_canada['HS Code'].isin(relevant_hs)]

# 4.5 Merge in the “Category” and “Portfolio” from HS mapping
df_imp_canada = pd.merge(
    df_imp_canada,
    df_hsmap[['HS Code', 'Category', 'Portfolio']],
    on='HS Code',
    how='left'
)

# 4.6 Create a trend‐over‐years table of import values by Category
df_trend = df_imp_canada.groupby(['Year', 'Category'])['value'].sum().reset_index()

# 4.7 Plot the time series of import values (in million USD) for each category
plt.figure(figsize=(12, 6))
years = sorted(df_trend['Year'].unique())
for cat in df_trend['Category'].unique():
    dfc = df_trend[df_trend['Category'] == cat]
    plt.plot(
        dfc['Year'],
        dfc['value'] / 1e6,  # convert to million USD
        marker='o',
        label=cat
    )
plt.title('Trend of Import Value by Category (Canada from US)', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Import Value (Million USD)', fontsize=14)
plt.xticks(years, fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12, title='Category')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# 4.8 Focus on the latest year, sum import value by Category
LATEST_YEAR = df_imp_canada['Year'].max()
df_latest = (
    df_imp_canada[df_imp_canada['Year'] == LATEST_YEAR]
    .groupby('Category')['value']
    .sum()
    .reset_index()
    .rename(columns={'value': 'Import_Value_USD'})
)

# 4.9 Simulate “Additional Cost” = Import_Value_USD × Tariff_Rate for each category at each rate
records = []
for _, row in df_latest.iterrows():
    cat = row['Category']
    val = row['Import_Value_USD']
    for rate in TARIFF_RATES:
        records.append({
            'Category':        cat,
            'Tariff_Rate':     rate,
            'Tariff_Impact':   val * rate
        })
df_tariff_imp = pd.DataFrame(records)

# 4.10 Plot “Tariff Impact vs Tariff Rate” (y‐axis in million USD)
plt.figure(figsize=(12, 6))
for cat in df_tariff_imp['Category'].unique():
    dfc = df_tariff_imp[df_tariff_imp['Category'] == cat]
    plt.plot(
        dfc['Tariff_Rate'] * 100,
        dfc['Tariff_Impact'] / 1e6,  # convert to million USD
        marker='o',
        label=cat
    )
plt.title(f'Tariff Impact vs. Tariff Rate (Year {LATEST_YEAR})', fontsize=16)
plt.xlabel('Tariff Rate (%)', fontsize=14)
plt.ylabel('Additional Cost (Million USD)', fontsize=14)
plt.xticks([r * 100 for r in TARIFF_RATES], fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12, title='Category')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# 4.11 Print a summary table of incremental tariff impacts in the terminal
print(f"\nTariff Impact by Category in {LATEST_YEAR} (for selected rates):")
for _, row in df_latest.iterrows():
    cat = row['Category']
    val = row['Import_Value_USD']
    impacts = {f"{int(r * 100)}%": val * r for r in TARIFF_RATES}
    impact_str = ", ".join([f"{k}: ${v:,.0f}" for k, v in impacts.items()])
    print(f"  • {cat}: {impact_str}")


# ——————————————————————————————————————————
# 5. PRINT FINAL RECOMMENDATIONS BASED ON FINDINGS
# ——————————————————————————————————————————
print("\n\n=== RECOMMENDATIONS ===\n")

# Vulnerability‐based recommendations
print("1) Vulnerability Insights:")
print("   • The Top 10 vulnerable categories (by “Annual spend × weight”) should be prioritized for risk mitigation.")
print("   • For categories rated “Low” risk but with very high spend—especially in Major Equipment—consider dual sourcing or building local inventory buffers.\n")

# Tariff‐shock recommendations
print("2) Tariff‐Shock Insights:")
print(f"   • If a {int(TARIFF_RATES[-1]*100)}% U.S. tariff is applied, the categories incurring the highest incremental costs in {LATEST_YEAR} are:")
for cat, imp in df_tariff_imp[df_tariff_imp['Tariff_Rate'] == TARIFF_RATES[-1]][['Category','Tariff_Impact']].values:
    print(f"     ‣ {cat}: +${imp:,.0f} USD")
print("   • At the portfolio level, explicit attention is needed for Major Equipment categories that rely heavily on U.S. imports.\n")

# Short‐term actions
print("3) Short‐Term Mitigation Options:")
print("   • Negotiate multi‐year contracts with U.S. suppliers to cap any sudden tariff spikes under USMCA/NAFTA rules.")
print("   • Increase safety stock for high‐impact categories (e.g., Distribution Transformers, Power Transformers).")
print("   • Explore Canadian or alternate non‐U.S. suppliers for Medium‐risk categories to reduce immediate tariff exposure.\n")

# Long‐term strategy
print("4) Long‐Term Strategic Recommendations:")
print("   • Diversify across global suppliers—focus on Europe and Asia—for categories flagged as highly vulnerable or tariff‐exposed.")
print("   • Invest in domestic manufacturing partnerships (e.g., Canadian transformer or switchgear producers) to bypass future tariffs.")
print("   • Use LPI (Logistics Performance Index) and Doing Business metrics to optimize trade corridors and reduce lead times.")
print("   • Explore vertical integration or joint‐venture models for critical equipment portfolios to mitigate single‐vendor risk.")
print("   • Implement a continuous monitoring dashboard combining real‐time trade data and risk tolerance metrics for dynamic decision‐making.\n")

print("=== END OF ANALYSIS ===\n")


# ——————————————————————————————————————————
# 6. EXTENSION: Low‐Risk, High‐Spend Categories “By Country” Performance
# ——————————————————————————————————————————
# (A) IDENTIFY THE TOP LOW‐RISK CATEGORIES BY SPEND
# ——————————————————————————————————————————
df_low_risk = df_risk[df_risk['Risk_norm'] == 'Low'].copy()

# Select the top N low‐risk categories by Annual spend
top_low_risk_cats = (
    df_low_risk
    .nlargest(TOP_LOW_N, 'Annual spend')['CATEGORY']
    .tolist()
)

print(f"\nTop {TOP_LOW_N} Low‐Risk Categories (by Annual spend):")
for cat in top_low_risk_cats:
    spend_val = df_low_risk.loc[df_low_risk['CATEGORY'] == cat, 'Annual spend'].values[0]
    print(f"  • {cat}  –  ${spend_val:,.2f}")

# ——————————————————————————————————————————
# (B) LOAD VENDOR & INVENTORY DATA FOR LOW‐RISK CATEGORIES
# ——————————————————————————————————————————
df_vendor_data = pd.read_excel(FILE_PATH, sheet_name=SHEET_VENDOR_DATA)
df_inventory   = pd.read_excel(FILE_PATH, sheet_name=SHEET_INVENTORY)

# Sanity check columns
for col in ['VendorNumber', 'Vendor performance']:
    if col not in df_vendor_data.columns:
        raise KeyError(f"Missing column in '{SHEET_VENDOR_DATA}': {col}")
for col in ['VendorNumber', 'Category NAME', 'Country Exporting', 'Average Lead time (days)']:
    if col not in df_inventory.columns:
        raise KeyError(f"Missing column in '{SHEET_INVENTORY}': {col}")

# Filter inventory to only top low-risk categories
df_inv_low_spend = df_inventory[df_inventory['Category NAME'].isin(top_low_risk_cats)].copy()

# Merge in “Annual spend” and “Risk_norm” for those categories from df_risk
df_inv_low_spend = df_inv_low_spend.merge(
    df_low_risk[['CATEGORY', 'Annual spend', 'Risk_norm']],
    left_on='Category NAME',
    right_on='CATEGORY',
    how='left'
)

# Merge in Vendor Performance to assign numeric score per vendor
PERFORMANCE_MAP = {
    'Excellent': 5,
    'Good':      4,
    'Average':   3,
    'Developing':2,
    'Poor':      1
}
df_inv_low_spend = df_inv_low_spend.merge(
    df_vendor_data[['VendorNumber', 'Vendor performance']],
    on='VendorNumber',
    how='left'
)
df_inv_low_spend['Perf_Score'] = df_inv_low_spend['Vendor performance'].map(PERFORMANCE_MAP)

# ——————————————————————————————————————————
# (C) AGGREGATE TO COUNTRY‐LEVEL METRICS
# ——————————————————————————————————————————
# Average Lead Time per country
df_country_lead = (
    df_inv_low_spend
    .groupby('Country Exporting')['Average Lead time (days)']
    .mean()
    .reset_index(name='Avg_Lead_Time')
)

# Average Vendor Performance Score per country
df_country_perf = (
    df_inv_low_spend
    .groupby('Country Exporting')['Perf_Score']
    .mean()
    .reset_index(name='Avg_Vendor_Score')
)

# Vendor Count per country
df_country_vcount = (
    df_inv_low_spend
    .groupby('Country Exporting')['VendorNumber']
    .nunique()
    .reset_index(name='Vendor_Count')
)

# Total Spend per country (sum of category‐level spend for unique Category‐Country pairs)
df_unique_cat_country = df_inv_low_spend[['Country Exporting', 'CATEGORY', 'Annual spend']].drop_duplicates()
df_country_spend = (
    df_unique_cat_country
    .groupby('Country Exporting')['Annual spend']
    .sum()
    .reset_index(name='Total_Spend')
)

# Combine all country‐level metrics
df_country_metrics = (
    df_country_lead
    .merge(df_country_perf,    on='Country Exporting', how='left')
    .merge(df_country_vcount,  on='Country Exporting', how='left')
    .merge(df_country_spend,   on='Country Exporting', how='left')
)

# Round numeric columns
df_country_metrics['Avg_Lead_Time']    = df_country_metrics['Avg_Lead_Time'].round(2)
df_country_metrics['Avg_Vendor_Score'] = df_country_metrics['Avg_Vendor_Score'].round(2)
df_country_metrics['Total_Spend']      = df_country_metrics['Total_Spend'].round(2)

# Sort by Total_Spend descending
df_country_metrics = df_country_metrics.sort_values(by='Total_Spend', ascending=False)

print("\n\n=== Low‐Risk (Top‐Spend) Category Metrics by Country ===\n")
print(df_country_metrics.to_string(index=False))


# ——————————————————————————————————————————
# (D) VISUALIZATIONS: COUNTRY‐LEVEL PERFORMANCE
# ——————————————————————————————————————————
countries        = df_country_metrics['Country Exporting'].tolist()
lead_times       = df_country_metrics['Avg_Lead_Time'].tolist()
perf_scores      = df_country_metrics['Avg_Vendor_Score'].tolist()
vendor_counts    = df_country_metrics['Vendor_Count'].tolist()
countries_spend  = df_country_metrics['Total_Spend'].tolist()

# 6.13 Bar Chart: Total Spend by Country
plt.figure(figsize=(10, 5))
bars_spend = plt.bar(
    countries,
    countries_spend,
    color='steelblue',
    edgecolor='black'
)
plt.title(f'Total Annual Spend for Top {TOP_LOW_N} Low‐Risk Categories by Country', fontsize=16)
plt.xlabel('Country', fontsize=14)
plt.ylabel('Total Spend (USD)', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=12)
for bar in bars_spend:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + max(countries_spend)*0.01,
        f"${height:,.0f}",
        ha='center',
        va='bottom',
        fontsize=11
    )
plt.tight_layout()
plt.show()

# 6.14 Bar Chart: Avg Vendor Performance Score by Country
plt.figure(figsize=(10, 5))
bars_perf = plt.bar(
    countries,
    perf_scores,
    color='seagreen',
    edgecolor='black'
)
plt.title(f'Average Vendor Performance Score for Top {TOP_LOW_N} Low‐Risk Categories', fontsize=16)
plt.xlabel('Country', fontsize=14)
plt.ylabel('Avg Vendor Score (1–5)', fontsize=14)
plt.ylim(0, 5.5)
plt.xticks(rotation=45, ha='right', fontsize=12)
for bar in bars_perf:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.05,
        f"{height:.2f}",
        ha='center',
        va='bottom',
        fontsize=11
    )
plt.tight_layout()
plt.show()

# 6.15 Bar Chart: Average Lead Time by Country
plt.figure(figsize=(10, 5))
bars_lead = plt.bar(
    countries,
    lead_times,
    color='darkorange',
    edgecolor='black'
)
plt.title(f'Average Lead Time (Days) for Top {TOP_LOW_N} Low‐Risk Categories by Country', fontsize=16)
plt.xlabel('Country', fontsize=14)
plt.ylabel('Avg Lead Time (days)', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=12)
for bar in bars_lead:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + max(lead_times)*0.01,
        f"{height:.1f}",
        ha='center',
        va='bottom',
        fontsize=11
    )
plt.tight_layout()
plt.show()

# 6.16 Scatter Plot: Spend vs. Lead Time (bubble size = Vendor_Count)
plt.figure(figsize=(8, 6))
bubble_sizes = (np.array(vendor_counts) / max(vendor_counts)) * 300  # scale bubbles by vendor count
plt.scatter(
    lead_times,
    countries_spend,
    s=bubble_sizes,
    c='violet',
    edgecolor='black',
    alpha=0.7
)
for i, country in enumerate(countries):
    plt.text(
        lead_times[i] + max(lead_times)*0.01,
        countries_spend[i] + max(countries_spend)*0.015,
        country,
        fontsize=11
    )
plt.title(f'Spend vs. Lead Time for Top {TOP_LOW_N} Low‐Risk Categories (by Country)', fontsize=16)
plt.xlabel('Avg Lead Time (days)', fontsize=14)
plt.ylabel('Total Spend (USD)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()


# ——————————————————————————————————————————
# (E) FINAL TEXT SUMMARY / TERMINAL OUTPUT
# ——————————————————————————————————————————
print("\n\n=== SUMMARY & NEXT STEPS ===\n")
print("• We focused on the TOP LOW‐RISK categories (to minimize disruption) with the largest annual spend.")
print(f"  These are: {top_low_risk_cats}\n")

print("• Country‐level metrics for those categories:")
print(df_country_metrics.to_string(index=False))
print("\n• Key observations:")
print("  1. Countries with the highest Total Spend (for our low‐risk list) likely represent single points of failure.")
print("  2. If a single country has both high spend and a high average lead time, that is a clear area to diversify suppliers.")
print("  3. Vendor performance (1–5 scale) below 4.0 might indicate quality or reliability concerns for that country’s vendors.\n")

print("• Recommended next steps:")
print("  a) For countries with extremely high Total Spend (e.g., USA, Germany…), investigate whether alternate low‐risk suppliers exist (e.g., in local or nearshore markets).")
print("  b) For countries where Avg Lead Time > 200 days, negotiate faster shipping or pre‐stock according to predicted demand.")
print("  c) For countries with Avg Vendor Score < 4.0, set up additional QA checks or shift volume to higher‐scoring vendors.")
print("  d) Consider “Country‐Risk Diversification”—spread procurement of the same low‐risk category across multiple geographies to reduce single‐source dependence.\n")

print("=== END OF EXTENDED ANALYSIS ===\n")
