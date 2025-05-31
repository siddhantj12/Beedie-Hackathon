import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set up the environment
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_data():
    """Load all datasets and perform initial data quality checks."""
    data = {}
    
    # Load internal data
    data['vendors'] = pd.read_excel('BC Hydro - All Vendor Data.xlsx')
    data['risk_tolerance'] = pd.read_excel('BC Hydro - SC risk tolerance by category.xlsx')
    data['inventory'] = pd.read_excel('BC Hydro - Vendor and Inventory Data.xlsx')
    
    # Load external data
    data['tariffs'] = pd.read_excel('WTO tariff rate data - HS Code specific.xlsx')
    data['imports'] = pd.read_excel('Imports from trading partners - HS Code specific.xlsx')
    data['lpi'] = pd.read_excel('Logistics Performance Index (LPI) - 2023.xlsx')
    data['doing_business'] = pd.read_excel('Doing Business - World Bank - Export Import Data.xlsx')
    
    return data

def data_quality_check(data):
    """Perform data quality checks and print summary statistics and columns."""
    print("\n=== Data Quality Report ===")
    
    for name, df in data.items():
        print(f"\n{name.upper()} Dataset:")
        print(f"Shape: {df.shape}")
        print("\nColumns:")
        print(list(df.columns))
        print("\nMissing Values:")
        print(df.isnull().sum())
        print("\nSummary Statistics:")
        print(df.describe(include='all'))
        
        # Check for duplicates
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            print(f"\nWARNING: Found {duplicates} duplicate rows")

def calculate_risk_scores(data):
    """Calculate risk scores for each category."""
    print("\nVendors columns:", data['vendors'].columns)
    print("\nInventory columns:", data['inventory'].columns)
    print("\nRisk Tolerance columns:", data['risk_tolerance'].columns)
    # Merge vendors and inventory on 'VendorNumber CLEAN'
    vendor_inventory = pd.merge(
        data['vendors'],
        data['inventory'],
        on='VendorNumber CLEAN',
        how='left'
    )
    print("\nColumns after vendor-inventory merge:", vendor_inventory.columns)
    print("\nSample rows after vendor-inventory merge:")
    print(vendor_inventory.head(10))
    # Merge with risk tolerance on 'Category NAME_x' and 'CATEGORY'
    vendor_category = pd.merge(
        vendor_inventory,
        data['risk_tolerance'],
        left_on='Category NAME_x',
        right_on='CATEGORY',
        how='left'
    )
    print("\nColumns after risk-tolerance merge:", vendor_category.columns)
    print("\nSample rows after risk-tolerance merge:")
    print(vendor_category.head(10))
    # Lead time variability (use std dev from inventory)
    vendor_category['lead_time_variability'] = vendor_category['Standard Deviation of Lead time (days)']
    # Spend (use 'Annual spend' from risk_tolerance)
    vendor_category['total_spend'] = vendor_category['Annual spend']
    # Tariff Rate (set to 0 for now)
    vendor_category['Tariff Rate'] = 0
    # Calculate risk score
    vendor_category['risk_score'] = (
        vendor_category['lead_time_variability'].fillna(0) *
        vendor_category['total_spend'].fillna(0)
    )
    return vendor_category

def plot_risk_scores(risk_data):
    """Create visualizations for risk analysis and show them interactively."""
    print("\nRisk Data for Plotting (first 10 rows):")
    print(risk_data[['PORTFOLIO', 'risk_score', 'Risk Tolerance of the category']].head(10))
    print("\nRisk Score value counts:")
    print(risk_data['risk_score'].value_counts())
    print("\nPORTFOLIO value counts:")
    print(risk_data['PORTFOLIO'].value_counts())
    print("\nNumber of nonzero risk scores:", (risk_data['risk_score'] > 0).sum())
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=risk_data,
        x='PORTFOLIO',
        y='risk_score',
        hue='Risk Tolerance of the category'
    )
    plt.xticks(rotation=45)
    plt.title('Risk Scores by Portfolio')
    plt.tight_layout()
    plt.show()
    if 'Country of Origin' in risk_data.columns and 'Category NAME_x' in risk_data.columns:
        pivot_data = risk_data.pivot_table(
            values='total_spend',
            index='Country of Origin',
            columns='Category NAME_x',
            aggfunc='sum'
        )
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_data, cmap='YlOrRd')
        plt.title('Total Spend by Country and Category')
        plt.tight_layout()
        plt.show()

def main():
    # Load data
    print("Loading data...")
    data = load_data()
    
    # Perform data quality checks
    print("\nPerforming data quality checks...")
    data_quality_check(data)
    
    # Calculate risk scores
    print("\nCalculating risk scores...")
    risk_data = calculate_risk_scores(data)
    
    # Create visualizations
    print("\nCreating visualizations...")
    plot_risk_scores(risk_data)
    
    # Print top 5 high-vulnerability categories
    print("\nTop 5 High-Vulnerability Categories:")
    top_risks = risk_data.nlargest(5, 'risk_score')
    print(top_risks[['Category NAME', 'PORTFOLIO', 'risk_score']])

if __name__ == "__main__":
    main() 