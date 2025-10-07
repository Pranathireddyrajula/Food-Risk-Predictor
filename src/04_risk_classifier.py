# --- src/04_risk_classifier.py ---

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanSquaredError
# --- 1. Define File Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', 'data')

FINAL_FEATURES_FILE = os.path.join(DATA_PATH, 'C:/Users/jbred/OneDrive/Desktop/food-risk-predictor/data/03_final_features_data.csv')
MODEL_FILE = os.path.join(BASE_DIR, 'C:/Users/jbred/OneDrive/Desktop/food-risk-predictor/data/price_prediction_model.h5')

# Model Lookback (Must match the N_STEPS used in model_train.py)
N_STEPS = 3 

# --- 2. Risk Classification Logic ---

def classify_risk(row, price_forecast, volatility, initial_hunger_score):
    """
    Combines price forecast, volatility, and GHI score into an overall risk category.
    The logic is simplified for demonstration purposes.
    """
    
    # 1. Define Risk Thresholds
    # This matrix reflects how a price spike affects the GHI score's base level.
    # We use GHI categories: 0-9.9 (Low), 10-19.9 (Moderate), 20-34.9 (Serious), 35+ (Alarming)
    
    # Calculate Forecasted Price Change (Spike)
    price_spike = (price_forecast - row['Price_USD']) / row['Price_USD'] # % change from last known price

    # Base the risk initially on the GHI score
    if initial_hunger_score >= 35.0:
        base_risk = "Emergency"
    elif initial_hunger_score >= 20.0:
        base_risk = "Crisis"
    elif initial_hunger_score >= 10.0:
        base_risk = "Stress"
    else:
        base_risk = "Low"

    # 2. Adjust Risk based on Price Volatility and Spikes
    # Rule 1: A massive price spike pushes the risk up by one level.
    if price_spike > 0.20 and volatility > 0.05: # >20% price spike AND high volatility
        if base_risk == "Crisis":
            return "Emergency"
        elif base_risk == "Stress":
            return "Crisis"
    
    # Rule 2: Extremely high volatility alone indicates major market instability.
    if volatility > 0.10 and base_risk == "Stress":
        return "Crisis"

    return base_risk

# --- 3. Main Prediction and Classification ---

if __name__ == "__main__":
    
    # Load Model and Data
    print("Loading model and final data...")
    model = load_model(
        MODEL_FILE,
        custom_objects={
            'Adam': Adam,  # Ensure the optimizer is recognized
            'mse': MeanSquaredError # Ensure the metric/loss is recognized
        }
    )
    final_data = pd.read_csv(FINAL_FEATURES_FILE)
    
    # --- Step A: Re-setup Scalers (Crucial for Inverse Transform) ---
    
    # Re-fit the scalers using the training data columns (must match model_train.py)
    TARGET_COL = 'Price_USD'
    
    # Identify numeric features
    numeric_cols = final_data.select_dtypes(include=np.number).columns.tolist()
    cols_to_scale = [col for col in numeric_cols if col not in ['Year', 'Hunger_Score']]
    
    # The scalers must be fitted to the *original* data ranges.
    feature_scaler = MinMaxScaler(feature_range=(0, 1)).fit(final_data[cols_to_scale])
    target_scaler = MinMaxScaler(feature_range=(0, 1)).fit(final_data[[TARGET_COL]]) 

    # --- Step B: Prepare Latest Data for Prediction (The Lookback Window) ---
    
    
    X_predict_list = []
    # List to track only the groups that have enough data (N_STEPS) for prediction
    groups_to_predict = [] 
    
    # Group by Country and Commodity
    for (country, commodity), group in final_data.groupby(['Country_Code', 'Commodity']):
        
        # CRITICAL FIX: Only process groups with enough historical data
        if len(group) >= N_STEPS: 
            
            # Get the latest N_STEPS (3 months) of feature data
            last_n_steps = group.tail(N_STEPS) 
            
            # Apply SCALING and add the 3D sequence
            features_to_scale = last_n_steps[cols_to_scale]
            scaled_features = feature_scaler.transform(features_to_scale)
            
            X_predict_list.append(scaled_features)
            groups_to_predict.append((country, commodity))
        
    X_predict = np.array(X_predict_list)


    # --- Step C: Generate Prediction ---
    
    # Run the prediction
    y_pred_scaled = model.predict(X_predict)
    
    # Inverse Transform: Convert the scaled prediction back to real USD prices
    y_pred_usd = target_scaler.inverse_transform(y_pred_scaled)
    
    # --- Step D: Format Results and Classify Risk ---
    
    # Create the results DataFrame (simplified to the last observation before prediction)
    # --- Step D: Format Results and Classify Risk (Aligning the results) ---
    
    # 1. Filter the final_data to ONLY include the groups we successfully predicted for
    # We use .set_index().index.isin() to filter the main dataframe by the groups_to_predict list.
    predicted_groups_df = final_data[
        final_data.set_index(['Country_Code', 'Commodity']).index.isin(groups_to_predict)
    ].copy()
    
    # 2. Get the last known row for each predicted series (Length of this matches the prediction)
    # This creates the report base with exactly 260 rows (or however many passed the check).
    results_df = predicted_groups_df.groupby(['Country_Code', 'Commodity']).tail(1).reset_index(drop=True)
    
    # 3. Add the predicted price (Lengths now match!)
    results_df['Predicted_Price_USD'] = y_pred_usd.flatten()
    
    # 4. Calculate the prediction month
    results_df['Prediction_Month'] = pd.to_datetime(results_df['Date']) + pd.DateOffset(months=1)
    
    print("\n--- Generating Risk Classification ---")

    # 5. Apply the classification function
    results_df['Hunger_Risk_Level'] = results_df.apply(
        lambda row: classify_risk(
            row,
            row['Predicted_Price_USD'],
            row['Volatility_3M'],
            row['Hunger_Score']
        ),
        axis=1
    )
    # ... (Rest of the script continues here)

    # Final presentation table
    final_output = results_df[[
        'Prediction_Month', 'Country_Code', 'Commodity', 
        'Price_USD', 'Predicted_Price_USD', 'Volatility_3M', 
        'Hunger_Score', 'Hunger_Risk_Level'
    ]]
    
    print("\n--- FINAL ACTIONABLE RISK REPORT (NEXT MONTH) ---")
    print(final_output.head(10).to_markdown(index=False))