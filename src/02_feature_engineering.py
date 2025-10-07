# --- src/02_feature_engineering.py ---

import pandas as pd
import numpy as np
import os

# --- 1. Define File Paths ---
# Use the files saved in the previous step
MONTHLY_PRICES_FILE = "C:/Users/jbred/OneDrive/Desktop/food-risk-predictor/data/02_monthly_prices.csv"
GHI_SCORES_FILE = "C:/Users/jbred/OneDrive/Desktop/food-risk-predictor/data/02_ghi_cleaned.csv"
OUTPUT_FINAL_FILE = "C:/Users/jbred/OneDrive/Desktop/food-risk-predictor/data/03_final_features_data.csv"

# --- 2. Country Code Mapping (The Bridge) ---
# CRITICAL STEP: WFP uses 3-letter ISO codes (e.g., AFG), GHI uses Country Names.
# We create a mapping dictionary for the countries we can see to demonstrate the technique.
# In a real project, this mapping would come from a comprehensive external source.
ISO_MAP = {
    'Afghanistan': 'AFG', 'Albania': 'ALB', 'Angola': 'AGO', 'Argentina': 'ARG', 
    'Armenia': 'ARM', 'Azerbaijan': 'AZE', 'Bahrain': 'BHR', 'Bangladesh': 'BGD', 
    'Belarus': 'BLR', 'Benin': 'BEN', 'Bhutan': 'BTN', 'Bolivia (Plurinat. State of)': 'BOL',
    'Bosnia & Herzegovina': 'BIH', 'Botswana': 'BWA', 'Brazil': 'BRA', 'Bulgaria': 'BGR', 
    'Burkina Faso': 'BFA', 'Burundi': 'BDI', 'Cabo Verde': 'CPV', 'Cambodia': 'KHM'
    # ... more countries here
}

# --- 3. Feature Generation Function ---

def generate_features(prices_df, ghi_df):
    """
    Generates time-series features (Volatility, Lag) and merges the GHI score.
    """
    
    print("Generating time-series features (Volatility, Lag)...")
    
    # Ensure 'Month' is sorted and correctly typed for rolling calculations
    prices_df.rename(columns={'Month': 'Date'}, inplace=True)
    prices_df['Date'] = pd.to_datetime(prices_df['Date'])
    prices_df.sort_values(by=['Country_Code', 'Commodity', 'Date'], inplace=True)

    # --- FEATURE 1: PRICE VOLATILITY ---
    # Volatility is measured as the Standard Deviation of price over a rolling window.
    # .rolling(3) looks at the current month and the previous 2 months (3 total)
    prices_df['Volatility_3M'] = prices_df.groupby(['Country_Code', 'Commodity'])['Price_USD'].transform(
        lambda x: x.rolling(3, min_periods=1).std() # min_periods=1 allows calculation on first month
    )
    
    # --- FEATURE 2: LAGGED PRICE (Price Momentum) ---
    # Lag is the price from the previous month. It helps predict momentum.
    prices_df['Price_Lag_1M'] = prices_df.groupby(['Country_Code', 'Commodity'])['Price_USD'].shift(1)
    
    # Fill the first month's Lagged Price (which is NaN) with the current price to prevent data loss.
    prices_df['Price_Lag_1M'] = prices_df['Price_Lag_1M'].fillna(prices_df['Price_USD'])

    print("Merging GHI scores with price data...")
    
    # --- FEATURE 3 & DATA MERGE: GHI Risk Score ---
    
    # 3a. Map the GHI Data to ISO Codes
    # Apply the mapping to create a column compatible with the price data
    ghi_df['Country_Code'] = ghi_df['Country_Name'].map(ISO_MAP)
    
    # Filter GHI to only countries that we successfully mapped
    ghi_df.dropna(subset=['Country_Code'], inplace=True)

    # 3b. Merge the two datasets
    # Merge on Country_Code and Year (using 2024 as the Year for GHI)
    # Since prices are monthly, and GHI is yearly (2024), we create a 'Year' column in prices to match
    prices_df['Year'] = prices_df['Date'].dt.year

    final_data = pd.merge(
        prices_df,
        ghi_df[['Country_Code', 'Hunger_Score']], # Only need the country code and score
        on='Country_Code',
        how='left' # Keep all price records, match GHI score if available
    )
    
    # Final cleanup of any rows that failed the merge
    final_data.dropna(subset=['Volatility_3M', 'Hunger_Score'], inplace=True)

    # Reorder columns for presentation
    final_data = final_data[[
        'Date', 'Country_Code', 'Commodity', 'Price_USD', 
        'Price_Lag_1M', 'Volatility_3M', 'Hunger_Score'
    ]]
    
    print(f"Final Merged Data Rows: {len(final_data)}")
    return final_data


# --- 4. Main Execution ---

if __name__ == "__main__":
    
    # Load the cleaned data from the previous step
    monthly_prices = pd.read_csv(MONTHLY_PRICES_FILE)
    ghi_cleaned = pd.read_csv(GHI_SCORES_FILE)
    
    # Run feature engineering
    final_data = generate_features(monthly_prices, ghi_cleaned)
    
    # Save the final file
    final_data.to_csv(OUTPUT_FINAL_FILE, index=False)

    print("\n--- Feature Engineering Complete ---")
    print(f"Final dataset saved to: {OUTPUT_FINAL_FILE}")
    print("\nSample of Final Feature Set:")
    print(final_data.head().to_markdown(index=False))