import pandas as pd
import matplotlib.pyplot as plt

# ———————————— 1. Load the Excel File ————————————
file_path = '/Users/siddhant/Desktop/Beedie Hackathon/EVERYTHING.xlsx'

# Read in the two sheets we will visualize
df_supply = pd.read_excel(file_path, sheet_name='Supply chain risk & spend')
df_tariff = pd.read_excel(file_path, sheet_name='Tariffs Rates by Country')


# ———————————— 2. Visualization of Product-Cost Vulnerability (Spend vs. Risk) ————————————

# 2.1 Top 10 Categories by Annual Spend (High Risk Tolerance)
#    • Filter for rows where “Risk Tolerance of the category” == 'High'
#    • Sort by Annual spend descending, take top 10
df_high_risk = df_supply[df_supply['Risk Tolerance of the category'] == 'High'].dropna(subset=['Annual spend'])
df_high_risk_top = df_high_risk.sort_values(by='Annual spend', ascending=False).head(10)

plt.figure(figsize=(12, 6))
plt.bar(
    df_high_risk_top['CATEGORY'],
    df_high_risk_top['Annual spend'] / 1e6,      # Convert to millions for readability
    color='orange',
    edgecolor='black'
)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.title('Top 10 Categories by Annual Spend (High Risk Tolerance)', fontsize=14)
plt.xlabel('Category', fontsize=12)
plt.ylabel('Annual Spend (Millions)', fontsize=12)
plt.tight_layout()
plt.show()


# 2.2 Annual Spend Distribution by Risk Tolerance
#    • Drop rows missing either “Annual spend” or “Risk Tolerance of the category”
#    • Create a boxplot to compare spend distributions across Low, Medium, and High
df_box = df_supply.dropna(subset=['Annual spend', 'Risk Tolerance of the category'])
risk_order = ['Low', 'Medium', 'High']

plt.figure(figsize=(8, 6))
# The boxplot’s x-axis will automatically order by category names, 
# so we reindex the DataFrame to ensure “Low, Medium, High” ordering if needed.
df_box['Risk Tolerance of the category'] = pd.Categorical(
    df_box['Risk Tolerance of the category'],
    categories=risk_order,
    ordered=True
)
df_box.boxplot(
    column='Annual spend',
    by='Risk Tolerance of the category',
    grid=False,
    showfliers=False,
    patch_artist=True,
    boxprops=dict(facecolor='orange', edgecolor='black'),
    medianprops=dict(color='red'),
)
plt.title('Annual Spend Distribution by Risk Tolerance', fontsize=14)
plt.suptitle('')  # Remove the automatic “Boxplot grouped by …” title
plt.xlabel('Risk Tolerance', fontsize=12)
plt.ylabel('Annual Spend (USD)', fontsize=12)
plt.tight_layout()
plt.show()


# ———————————— 3. Visualization of Tariff Data ————————————

# 3.1 Distribution of Average Tariff Rates (All Rows)
#    • Drop any rows where 'Avg. Duty (%)' is NaN
df_tariff_clean = df_tariff.dropna(subset=['Avg. Duty (%)'])

plt.figure(figsize=(8, 6))
plt.hist(
    df_tariff_clean['Avg. Duty (%)'],
    bins=20,
    edgecolor='black',
    color='orange'
)
plt.title('Distribution of Average Tariff Rates', fontsize=14)
plt.xlabel('Avg. Duty (%)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.tight_layout()
plt.show()


# 3.2 Top 10 Exporting Countries by Average Tariff to Canada (Year 2021)
#    • Filter for rows where “Country Importing” == 'Canada' and Year == 2021
#    • Group by “Country Exporting,” compute mean(Avg. Duty (%)), sort descending, take top 10
df_canada_2021 = df_tariff_clean[
    (df_tariff_clean['Country Importing'] == 'Canada') &
    (df_tariff_clean['Year'] == 2021)
]
df_canada_group = (
    df_canada_2021
    .groupby('Country Exporting')['Avg. Duty (%)']
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(12, 6))
plt.bar(
    df_canada_group.index,
    df_canada_group.values,
    color='orange',
    edgecolor='black'
)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.title('Top 10 Exporting Countries by Average Tariff to Canada (2021)', fontsize=14)
plt.xlabel('Exporting Country', fontsize=12)
plt.ylabel('Avg. Duty (%)', fontsize=12)
plt.tight_layout()
plt.show()
