# --- src/03_model_train.py ---

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# --- 1. Define File Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', 'data')

FINAL_FEATURES_FILE = os.path.join(DATA_PATH, 'C:/Users/jbred/OneDrive/Desktop/food-risk-predictor/data/03_final_features_data.csv')
MODEL_OUTPUT_FILE = os.path.join(BASE_DIR, 'C:/Users/jbred/OneDrive/Desktop/food-risk-predictor/data/price_prediction_model.h5')

# --- 2. Model Hyperparameters and Data Setup ---
# TUNEABLE PARAMETERS: You can change these later to optimize the model
N_STEPS = 3       # Number of past months (timesteps) to look back for each prediction
N_EPOCHS = 20     # Number of training iterations (kept low for fast execution)
BATCH_SIZE = 32   # Number of samples processed before updating model internal parameters

# --- 3. LSTM Data Sequence Preparation ---

def create_sequences(data, n_steps, target_column):
    """
    Converts time-series data into the 3D structure required by LSTM networks.
    """
    X, y = [], []
    
    # 1. Split data by Country and Commodity to process each time series independently
    for (country, commodity), group in data.groupby(['Country_Code', 'Commodity']):
        
        # 2. Extract the features (all numeric columns except the target)
        features = group.drop(columns=['Date', 'Country_Code', 'Commodity', target_column]).values
        
        # 3. Extract the target (Price_USD)
        targets = group[target_column].values
        
        # 4. Create Sequences
        for i in range(len(group) - n_steps):
            # X: The sequence of 'n_steps' historical data points (the lookback window)
            X.append(features[i:i + n_steps])
            
            # y: The single target value *immediately following* the sequence
            y.append(targets[i + n_steps])
            
    return np.array(X), np.array(y)

# --- 4. Main Model Building ---

if __name__ == "__main__":
    
    # --- Step 1: Load and Prepare Data ---
    final_data = pd.read_csv("C:/Users/jbred/OneDrive/Desktop/food-risk-predictor/data/03_final_features_data.csv")
    
    # Separate features and target before splitting/scaling
    TARGET_COL = 'Price_USD'
    
    # --- Step 2: Normalization (Scaling) ---
    # Scaling is crucial for Deep Learning stability. We scale all numeric features.
    
    # Identify numeric features
    numeric_cols = final_data.select_dtypes(include=np.number).columns.tolist()
    
    # Remove Country_Code, Year, Hunger_Score (static) from time-series scaling, but keep Price_USD for the target scaler
    cols_to_scale = [col for col in numeric_cols if col not in ['Year', 'Hunger_Score']]
    
    # Create two separate scalers: one for features, one for the target (Price_USD)
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1)) 
    
    # Apply feature scaling to the columns used for sequence generation
    final_data[cols_to_scale] = feature_scaler.fit_transform(final_data[cols_to_scale])
    
    # Fit the target scaler separately (needed to inverse transform the predictions later)
    target_scaler.fit(final_data[[TARGET_COL]])
    
    # --- Step 3: Reshape Data for LSTM ---
    X, y = create_sequences(final_data, N_STEPS, TARGET_COL)
    
    # --- Step 4: Train/Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False # Shuffle=False is CRITICAL for time-series data
    )

    # --- Step 5: Build the LSTM Model ---
    
    # The Sequential model stacks layers linearly
    model = Sequential([
        # LSTM Layer: The core of the time-series model. Input shape is (timesteps, features)
        LSTM(units=50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
        
        # Dense Layer: Standard neural network layer to map LSTM output to the final prediction
        Dense(units=1)
    ])
    
    # Compile: Configure the learning process
    model.compile(optimizer='adam', loss='mse') # 'mse' (Mean Squared Error) is standard for regression/forecasting
    
    print("\n--- Model Summary ---")
    model.summary()

    # --- Step 6: Train the Model ---
    print("\nStarting Model Training...")
    
    # EarlyStopping: Stops training if the model performance stops improving (prevents overfitting)
    history = model.fit(
        X_train, y_train, 
        epochs=N_EPOCHS, 
        batch_size=BATCH_SIZE, 
        validation_split=0.1, 
        callbacks=[EarlyStopping(monitor='val_loss', patience=5)],
        verbose=1
    )

    # --- Step 7: Evaluate and Save ---
    
    # Save the model to a file
    model.save(MODEL_OUTPUT_FILE)
    print(f"\nModel successfully trained and saved to: {MODEL_OUTPUT_FILE}")
    
    # Evaluate model performance (loss) on the test set
    loss = model.evaluate(X_test, y_test, verbose=0)
    print(f"Final Test Loss (MSE): {loss:.6f}")
    
    print("\nNext Step: Use this model to predict prices and classify hunger risk!")