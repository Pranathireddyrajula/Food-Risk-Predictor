import pandas as pd
import numpy as np
import os

# --- 1. Define File Paths ---
# This ensures the code can find the data regardless of your computer (Windows/Mac)
# '..' navigates up one level from 'src' to the root folder.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', 'data')

WFP_FILE = os.path.join(DATA_PATH, "C:/Users/jbred/OneDrive/Desktop/food-risk-predictor/data/raw/wfp_food_prices.csv")
GHI_FILE = os.path.join(DATA_PATH, "C:/Users/jbred/OneDrive/Desktop/food-risk-predictor/data/raw/global_hunger_index.xlsx")

OUTPUT_MONTHLY_PRICES = os.path.join(DATA_PATH, '02_monthly_prices.csv')
OUTPUT_GHI_SCORES = os.path.join(DATA_PATH, '02_ghi_cleaned.csv')


print("Starting data cleaning and merging...")


# --- 2. Clean Price Data (WFP) ---
def clean_prices(file_path):
    """Loads and cleans the World Food Programme price data."""
    
    # parse_dates=['date'] is CRITICAL: it converts the 'date' column into a usable date format
    wfp_raw = pd.read_csv(file_path, low_memory=False, parse_dates=['date'])
    
    print(f"Loaded WFP data. Original records: {len(wfp_raw)}")

    # 2a. Select and Rename Columns for Simplicity
    # We rename columns to simple, clear English terms
    wfp = wfp_raw[[
        'date', 'countryiso3', 'commodity', 'usdprice'
    ]].copy()
    
    wfp.columns = ['Date', 'Country_Code', 'Commodity', 'Price_USD']

    wfp['Date'] = pd.to_datetime(wfp['Date'], errors='coerce')

    wfp['Price_USD'] = pd.to_numeric(wfp['Price_USD'], errors='coerce') 
    
    # 2b. Clean Data Types and Missing Values
    # Remove any rows where the price is missing (we can't forecast without a price)
    wfp.dropna(subset=['Date', 'Price_USD'], inplace=True)
    
    # Sort by date and country (essential for any time-series analysis)
    wfp.sort_values(by=['Country_Code', 'Commodity', 'Date'], inplace=True)
    
    print(f"Cleaned WFP data rows: {len(wfp)}")
    return wfp


# --- 3. Clean Hunger Index Data (GHI) ---
def clean_hunger_index(file_path):
    """Loads and cleans the Global Hunger Index scores."""
    
    # skiprows=2 tells pandas to ignore the first 2 rows (metadata/titles)
    # The actual column headers start on the 3rd row (index 2)
    ghi_raw = pd.read_excel(file_path, skiprows=2, header=0)
    
    # 3a. Select and Rename Columns
    # We select the first two columns, which contain the Country name and the 2024 Score
    ghi = ghi_raw.iloc[:, 0:2].copy()
    ghi.columns = ['Country_Name', 'Hunger_Score']
    
    # 3b. Clean Scores and Add Year
    # Replace the text score '<5' with the numeric value 4.5 
    ghi['Hunger_Score'] = ghi['Hunger_Score'].astype(str).str.replace('<5', '4.5')
    
    # Convert score column to numeric, setting any un-cleanable text to NaN
    ghi['Hunger_Score'] = pd.to_numeric(ghi['Hunger_Score'], errors='coerce')
    
    # Remove rows that are clearly non-country metadata
    ghi.dropna(subset=['Country_Name', 'Hunger_Score'], inplace=True)
    
    # Add a column for the year, since this GHI data is for 2024
    ghi['Year'] = 2024
    
    print(f"Cleaned GHI data rows: {len(ghi)}")
    return ghi


# --- 4. Main Execution (The Master Plan) ---
if __name__ == "__main__":
    
    # Step 1: Run the Cleaning Functions
    wfp_cleaned = clean_prices(WFP_FILE)
    ghi_cleaned = clean_hunger_index(GHI_FILE)

    # Step 2: Prepare for Merging (The Bridge)
    # The WFP data uses Country CODES (e.g., 'AFG'), but the GHI data uses Country NAMES (e.g., 'Afghanistan').
    # We need a bridge! For simplicity now, we will save both files and address the full
    # country code mapping challenge in the next session (Phase 2).
    # Since the WFP data is very detailed, we will aggregate it first to simplify.
    
    # Aggregate: Calculate the average price per Country/Commodity/Month
    wfp_cleaned['Month'] = wfp_cleaned['Date'].dt.to_period('M')
    
    # This creates a single time-series point for every month/country/commodity
    monthly_prices = wfp_cleaned.groupby(
        ['Country_Code', 'Commodity', 'Month']
    )['Price_USD'].mean().reset_index()
    
    # --- Step 3: Save the Interim Files ---
    monthly_prices.to_csv(OUTPUT_MONTHLY_PRICES, index=False)
    ghi_cleaned.to_csv(OUTPUT_GHI_SCORES, index=False)
    
    print("\n--- Data Preparation Complete ---")
    print(f"1. Monthly Prices saved to: {OUTPUT_MONTHLY_PRICES}")
    print(f"2. GHI Scores saved to: {OUTPUT_GHI_SCORES}")
    
    print("\nSample of Cleaned GHI Scores (The missing file):")
    print(ghi_cleaned.head().to_markdown(index=False))