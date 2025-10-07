import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from io import StringIO

# --- 1. CONFIGURATION AND UTILITIES ---
N_STEPS = 3  # Lookback window for LSTM

# CRITICAL FIX: Define the explicit feature set (all numeric features needed for LSTM)
FEATURE_COLS = [
    'Price_USD', 
    'Price_Lag_1M', 
    'Volatility_3M', 
    'Hunger_Score' # This static score is treated as a feature at every timestep
]

# --- RISK CLASSIFICATION LOGIC (From 04_risk_classifier.py) ---

def classify_risk(row, price_forecast, volatility, initial_hunger_score):
    """
    Combines price forecast, volatility, and GHI score into an overall risk category.
    """
    # Calculate Price Spike (%)
    price_spike = (price_forecast - row['Price_USD']) / row['Price_USD']
    base_risk = "Low"
    
    # Base Risk based on GHI Score thresholds
    if initial_hunger_score >= 35.0:
        base_risk = "Emergency"
    elif initial_hunger_score >= 20.0:
        base_risk = "Crisis"
    elif initial_hunger_score >= 10.0:
        base_risk = "Stress"

    # Adjustment based on Price Volatility and Spikes
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

# --- DATA CLEANING FUNCTIONS (From data_prep.py) ---

def clean_col_name(col):
    """Aggressively cleans column names by removing non-alphanumeric symbols."""
    col = str(col).lower()
    # Remove common symbols found in messy headers: #, +, ., spaces, etc.
    for char in ['.', ' ', '#', '+', '_', '-', '(', ')', '%']:
        col = col.replace(char, '')
    return col

@st.cache_data
def clean_prices(uploaded_file):
    """Loads, cleans, and standardizes WFP price data using flexible column matching."""
    
    # 1. Load Data
    try:
        wfp_raw = pd.read_csv(uploaded_file, low_memory=False)
    except:
        wfp_raw = pd.read_csv(uploaded_file, low_memory=False, encoding='ISO-8859-1')
        
    # Standardize column names using the robust cleaning function
    raw_cols_map = {col: clean_col_name(col) for col in wfp_raw.columns}
    wfp_raw.rename(columns=raw_cols_map, inplace=True)
    
    # 2. Flexible Column Selection (Robustness Fix)
    
    # Keywords to search for in the standardized column names
    keywords = {
        'date': 'Date', 
        'countryiso3': 'Country_Code', # Matches 'countryiso3'
        'code': 'Country_Code',        # Matches 'countrycode'
        'commodity': 'Commodity', 
        'itemname': 'Commodity',       # Matches 'itemname'
        'usdprice': 'Price_USD',       # Matches 'usdprice'
        'valueusd': 'Price_USD'        # Matches 'valueusd'
    }

    found_cols = {}
    
    # Search for the columns using keywords
    for raw_col in wfp_raw.columns:
        for keyword, standardized_name in keywords.items():
            # Check if the keyword is *part* of the cleaned column name
            if keyword in raw_col and standardized_name not in found_cols:
                found_cols[standardized_name] = raw_col
                break

    # Validate mandatory columns
    required = ['Date', 'Country_Code', 'Commodity', 'Price_USD']
    missing = [req for req in required if req not in found_cols]
    
    if missing:
        # Raise a specific error if mandatory columns are missing
        raise ValueError(f"WFP Data Missing mandatory columns: {missing}. Found: {list(found_cols.keys())}")
    
    # 3. Apply Selection and Rename
    # Use the original (but cleaned) column names as keys, and the standardized names as values
    wfp = wfp_raw[list(found_cols.values())].copy()
    wfp.columns = list(found_cols.keys())
    
    # 4. Type Conversion Fixes
    wfp['Date'] = pd.to_datetime(wfp['Date'], errors='coerce') 
    wfp['Price_USD'] = pd.to_numeric(wfp['Price_USD'], errors='coerce') 

    # Final cleanup
    wfp.dropna(subset=['Date', 'Price_USD'], inplace=True)
    wfp.sort_values(by=['Country_Code', 'Commodity', 'Date'], inplace=True)
    
    # Aggregate to Monthly (to match feature engineering logic)
    wfp['Month'] = wfp['Date'].dt.to_period('M')
    monthly_prices = wfp.groupby(
        ['Country_Code', 'Commodity', 'Month']
    )['Price_USD'].mean().reset_index()
    monthly_prices.rename(columns={'Month': 'Date'}, inplace=True)
    monthly_prices['Date'] = monthly_prices['Date'].astype(str) # Convert Period back to string for saving in session
    
    return monthly_prices

@st.cache_data
def clean_hunger_index(uploaded_file):
    """Loads and cleans the Global Hunger Index scores."""
    # NOTE: Using uploaded_file directly for pandas functions is best practice in Streamlit
    ghi_raw = pd.read_csv(uploaded_file, skiprows=2, header=0, encoding='ISO-8859-1')
    
    # Selection and Renaming
    # GHI is slightly less flexible as the column names in the raw file are ambiguous,
    # but based on the files provided, index 0 is Country and index 1 is Score.
    ghi = ghi_raw.iloc[:, 0:2].copy()
    ghi.columns = ['Country_Name', 'Hunger_Score']
    
    # Score Cleaning and Type Conversion
    ghi['Hunger_Score'] = ghi['Hunger_Score'].astype(str).str.replace('<5', '4.5')
    ghi['Hunger_Score'] = pd.to_numeric(ghi['Hunger_Score'], errors='coerce')
    
    # Final cleanup
    ghi.dropna(subset=['Country_Name', 'Hunger_Score'], inplace=True)
    ghi['Year'] = 2024
    return ghi


# --- FEATURE ENGINEERING (From 02_feature_engineering.py) ---

@st.cache_data
def feature_engineer(prices_df, ghi_df):
    """Generates time-series features (Volatility, Lag) and merges the GHI score."""
    
    # CRITICAL: Convert date string back to datetime for calculation
    prices_df['Date'] = pd.to_datetime(prices_df['Date'])
    prices_df.sort_values(by=['Country_Code', 'Commodity', 'Date'], inplace=True)

    # --- Feature 1: Price Volatility (3-month rolling STD) ---
    prices_df['Volatility_3M'] = prices_df.groupby(['Country_Code', 'Commodity'])['Price_USD'].transform(
        lambda x: x.rolling(3, min_periods=1).std()
    )
    
    # --- Feature 2: Lagged Price (1-month shift) ---
    prices_df['Price_Lag_1M'] = prices_df.groupby(['Country_Code', 'Commodity'])['Price_USD'].shift(1)
    prices_df['Price_Lag_1M'] = prices_df['Price_Lag_1M'].fillna(prices_df['Price_USD'])

    # --- Feature 3 & Merge: GHI Risk Score ---
    
    # 2. Country Code Mapping (The Bridge) 
    ISO_MAP = {
        'Afghanistan': 'AFG', 'Albania': 'ALB', 'Angola': 'AGO', 'Argentina': 'ARG', 
        'Armenia': 'ARM', 'Azerbaijan': 'AZE', 'Bahrain': 'BHR', 'Bangladesh': 'BGD', 
        'Belarus': 'BLR', 'Benin': 'BEN', 'Bhutan': 'BTN', 'Bolivia (Plurinat. State of)': 'BOL',
        'Bosnia & Herzegovina': 'BIH', 'Botswana': 'BWA', 'Brazil': 'BRA', 'Bulgaria': 'BGR', 
        'Burkina Faso': 'BFA', 'Burundi': 'BDI', 'Cabo Verde': 'CPV', 'Cambodia': 'KHM',
        'Chad': 'TCD', 'China': 'CHN', 'Colombia': 'COL', 'Comoros': 'COM', 
        'Dem. Rep. of the Congo': 'COD', 'Djibouti': 'DJI', 'Ecuador': 'ECU', 'Egypt': 'EGY',
        'El Salvador': 'SLV', 'Eritrea': 'ERI', 'Eswatini': 'SWZ', 'Ethiopia': 'ETH',
        'Fiji': 'FJI', 'Gabon': 'GAB', 'Gambia': 'GMB', 'Georgia': 'GEO', 'Ghana': 'GHA',
        'Guatemala': 'GTM', 'Guinea': 'GIN', 'Guinea-Bissau': 'GNB', 'Guyana': 'GUY',
        'Haiti': 'HTI', 'Honduras': 'HND', 'India': 'IND', 'Indonesia': 'IDN', 'Iran (Islamic Republic of)': 'IRN',
        'Iraq': 'IRQ', 'Jamaica': 'JAM', 'Jordan': 'JOR', 'Kazakhstan': 'KAZ', 'Kenya': 'KEN',
        'Korea (DPR)': 'PRK', 'Kuwait': 'KWT', 'Kyrgyzstan': 'KGZ', 'Lao PDR': 'LAO',
        'Latvia': 'LVA', 'Lebanon': 'LBN', 'Lesotho': 'LSO', 'Liberia': 'LBR', 'Libya': 'LBY',
        'Lithuania': 'LTU', 'Madagascar': 'MDG', 'Malawi': 'MWI', 'Malaysia': 'MYS',
        'Maldives': 'MDV', 'Mali': 'MLI', 'Mauritania': 'MRT', 'Mauritius': 'MUS',
        'Mexico': 'MEX', 'Mongolia': 'MNG', 'Morocco': 'MAR', 'Mozambique': 'MOZ',
        'Myanmar': 'MMR', 'Namibia': 'NAM', 'Nepal': 'NPL', 'Nicaragua': 'NIC', 'Niger': 'NER',
        'Nigeria': 'NGA', 'North Macedonia': 'MKD', 'Oman': 'OMN', 'Pakistan': 'PAK',
        'Palestine': 'PSE', 'Panama': 'PAN', 'Papua New Guinea': 'PNG', 'Paraguay': 'PRY',
        'Peru': 'PER', 'Philippines': 'PHL', 'Romania': 'ROU', 'Russian Federation': 'RUS',
        'Rwanda': 'RWA', 'Sao Tome and Principe': 'STP', 'Saudi Arabia': 'SAU', 'Senegal': 'SEN',
        'Serbia': 'SRB', 'Sierra Leone': 'SLE', 'Slovakia': 'SVK', 'Slovenia': 'SVN',
        'Solomon Islands': 'SLB', 'Somalia': 'SOM', 'South Africa': 'ZAF', 'South Sudan': 'SSD',
        'Sri Lanka': 'LKA', 'Sudan': 'SDN', 'Syrian Arab Republic': 'SYR', 'Tajikistan': 'TJK',
        'Tanzania (United Rep. of)': 'TZA', 'Thailand': 'THA', 'Timor-Leste': 'TLS', 'Togo': 'TGO',
        'Trinidad and Tobago': 'TTO', 'Tunisia': 'TUN', 'Turkey': 'TUR', 'Uganda': 'UGA',
        'Ukraine': 'UKR', 'United Arab Emirates': 'ARE', 'United Kingdom': 'GBR', 'United States': 'USA',
        'Uruguay': 'URY', 'Uzbekistan': 'UZB', 'Vanuatu': 'VUT', 'Venezuela (Bolivarian Rep. of)': 'VEN',
        'Viet Nam': 'VNM', 'Yemen': 'YEM', 'Zambia': 'ZMB', 'Zimbabwe': 'ZWE',
    } # Expanded mapping for better merge coverage
    
    ghi_df['Country_Code'] = ghi_df['Country_Name'].map(ISO_MAP)
    ghi_df.dropna(subset=['Country_Code'], inplace=True)
    
    prices_df['Year'] = prices_df['Date'].dt.year

    final_data = pd.merge(
        prices_df,
        ghi_df[['Country_Code', 'Hunger_Score']], 
        on='Country_Code',
        how='left' 
    )
    
    # Final cleanup (must drop NaNs created by lagging/rolling, and failed GHI merges)
    final_data.dropna(subset=['Volatility_3M', 'Hunger_Score', 'Price_Lag_1M'], inplace=True)

    return final_data

# --- MODEL TRAINING AND PREDICTION LOGIC ---

# Placeholder for model and scalers persistence
MODEL_STATE = {
    'model': None,
    'feature_scaler': None,
    'target_scaler': None,
    'is_trained': False
}

def build_and_train_model(data, n_steps):
    """
    Builds the LSTM model structure and trains it using the final feature set.
    """
    with st.spinner("Building sequences and scaling data..."):
        
        TARGET_COL = 'Price_USD'
        
        # Setup Scalers
        feature_scaler = MinMaxScaler(feature_range=(0, 1))
        target_scaler = MinMaxScaler(feature_range=(0, 1)) 
        
        # FIX 1: Use FEATURE_COLS list for scaling
        data[FEATURE_COLS] = feature_scaler.fit_transform(data[FEATURE_COLS])
        target_scaler.fit(data[[TARGET_COL]])
        
        # --- Sequence Preparation ---
        X, y = [], []
        
        for _, group in data.groupby(['Country_Code', 'Commodity']):
            
            # FIX 2: Use FEATURE_COLS for sequence input
            features = group[FEATURE_COLS].values
            targets = group[TARGET_COL].values
            
            # Create sequences for training
            for i in range(len(group) - n_steps):
                X.append(features[i:i + n_steps])
                y.append(targets[i + n_steps])
                
        X, y = np.array(X), np.array(y)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

    with st.spinner("Training LSTM Model (This may take a few minutes)..."):
        
        # Build Model - Input shape must match X_train shape
        model = Sequential([
            # Input shape: (N_STEPS, number of features) -> (3, 4)
            LSTM(units=50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
            Dense(units=1)
        ])
        model.compile(optimizer='adam', loss='mse')
        
        # Train Model
        model.fit(
            X_train, y_train, 
            epochs=20, 
            batch_size=32, 
            validation_split=0.1, 
            callbacks=[EarlyStopping(monitor='val_loss', patience=5)],
            verbose=0
        )
        
        # Update Global State
        MODEL_STATE['model'] = model
        MODEL_STATE['feature_scaler'] = feature_scaler
        MODEL_STATE['target_scaler'] = target_scaler
        MODEL_STATE['is_trained'] = True
        
        st.session_state['model_trained'] = True
        st.success("Model Training Complete!")
        
        loss = model.evaluate(X_test, y_test, verbose=0)
        st.info(f"Model Test Loss (MSE): {loss:.5f}")


def predict_and_classify(data, n_steps):
    """Runs prediction and risk classification on the latest data points."""
    
    if not MODEL_STATE['is_trained']:
        st.error("Model is not trained. Please upload data and click 'Train Model'.")
        return None

    with st.spinner("Generating prediction sequences..."):
        
        # Load necessary components from state
        model = MODEL_STATE['model']
        feature_scaler = MODEL_STATE['feature_scaler']
        target_scaler = MODEL_STATE['target_scaler']
        
        X_predict_list = []
        groups_to_predict = [] 

        # Group data to ensure we only predict the next step for complete series
        for (country, commodity), group in data.groupby(['Country_Code', 'Commodity']):
            
            # FIX 3: Filtering for complete sequences (N_STEPS) must happen here too
            if len(group) >= n_steps: 
                last_n_steps = group.tail(n_steps) 
                
                # Apply SCALING to the lookback window using FEATURE_COLS
                scaled_features = feature_scaler.transform(last_n_steps[FEATURE_COLS])
                
                X_predict_list.append(scaled_features)
                groups_to_predict.append((country, commodity))
            
        X_predict = np.array(X_predict_list) 

    with st.spinner("Forecasting next month prices..."):
        
        # Generate Prediction
        # NOTE: If X_predict is empty due to insufficient data, this will raise a warning, 
        # but the prediction itself should run with the correct shapes.
        if X_predict.size == 0:
            st.warning("Insufficient data: No time series had the required 3 months of history for prediction.")
            return None

        y_pred_scaled = model.predict(X_predict, verbose=0)
        y_pred_usd = target_scaler.inverse_transform(y_pred_scaled)
        
        # Filter the final_data to ONLY include the groups we successfully predicted for
        predicted_groups_df = data[
            data.set_index(['Country_Code', 'Commodity']).index.isin(groups_to_predict)
        ].copy()
        
        # Get the last known row for each predicted series
        results_df = predicted_groups_df.groupby(['Country_Code', 'Commodity']).tail(1).reset_index(drop=True)
        
        # Add the predicted price
        results_df['Predicted_Price_USD'] = y_pred_usd.flatten()
        
        # Calculate the prediction month
        results_df['Prediction_Month'] = pd.to_datetime(results_df['Date']) + pd.DateOffset(months=1)

        # Apply the classification function
        results_df['Hunger_Risk_Level'] = results_df.apply(
            lambda row: classify_risk(
                row,
                row['Predicted_Price_USD'],
                row['Volatility_3M'],
                row['Hunger_Score']
            ),
            axis=1
        )
        
        # Final presentation table
        final_output = results_df[[
            'Prediction_Month', 'Country_Code', 'Commodity', 'Hunger_Risk_Level',
            'Price_USD', 'Predicted_Price_USD', 'Volatility_3M', 'Hunger_Score'
        ]].sort_values(by='Hunger_Risk_Level', ascending=False)
        
        st.success("Risk Forecasts Generated!")
        return final_output


# --- STREAMLIT APP LAYOUT ---

st.set_page_config(layout="wide", page_title="Global Hunger Risk Predictor")

st.title("🌍 Global Food Price Volatility & Hunger Risk Predictor")
st.markdown("""
This application demonstrates an end-to-end Machine Learning pipeline using an **LSTM Deep Learning model** to forecast food prices and assess **Acute Hunger Risk**. 
Upload the WFP Price Data and GHI Score Data below to run the analysis.
""")

# Initialize session state for training flag
if 'model_trained' not in st.session_state:
    st.session_state['model_trained'] = MODEL_STATE['is_trained']
if 'final_data' not in st.session_state:
    st.session_state['final_data'] = None

# --- DATA UPLOAD SIDEBAR ---
with st.sidebar:
    st.header("1. Upload Data Files")
    wfp_file = st.file_uploader("Upload WFP Prices (CSV)", type=["csv"], key="wfp")
    ghi_file = st.file_uploader("Upload 2024 GHI Scores (CSV)", type=["csv"], key="ghi")

    if wfp_file and ghi_file:
        st.success("Files uploaded successfully!")
        
        # Cleaning and Feature Engineering button
        if st.button("2. Clean Data & Engineer Features", key="engineer"):
            try:
                # 1. Clean Data
                with st.spinner("Cleaning WFP Price Data..."):
                    monthly_prices = clean_prices(wfp_file)
                with st.spinner("Cleaning GHI Scores..."):
                    ghi_cleaned = clean_hunger_index(ghi_file)
                
                # 2. Feature Engineering & Merge
                with st.spinner("Generating Volatility, Lag Features & Merging..."):
                    final_data = feature_engineer(monthly_prices.copy(), ghi_cleaned.copy())
                
                st.session_state['final_data'] = final_data
                st.session_state['data_ready'] = True
                st.success(f"Data Ready! Final merged dataset has {len(final_data)} rows.")
            except Exception as e:
                st.error(f"Data Processing Failed: {e}")
                st.info("The mandatory WFP columns must contain a date, a country code, a commodity name, and a USD price value.")


# --- MAIN CONTENT TABS ---

tab1, tab2, tab3 = st.tabs(["📊 Data Overview", "🧠 Train Model", "🚨 Risk Report"])

# TAB 1: DATA OVERVIEW
with tab1:
    st.header("Feature Matrix (Ready for LSTM)")
    if st.session_state['final_data'] is not None:
        st.markdown(f"**Total Features:** Price\_USD, Price\_Lag\_1M, Volatility\_3M, Hunger\_Score")
        st.dataframe(st.session_state['final_data'].head(20).astype(str))

        # Basic statistics check
        st.subheader("Statistical Summary of Key Features")
        numeric_summary = st.session_state['final_data'][[
            'Price_USD', 'Volatility_3M', 'Hunger_Score'
        ]].describe().T
        st.dataframe(numeric_summary.style.format(precision=3))

    else:
        st.info("Upload and process files in the sidebar to view the final feature matrix.")


# TAB 2: MODEL TRAINING
with tab2:
    st.header("LSTM Model Training")
    
    if st.session_state['final_data'] is not None:
        st.markdown("Click below to train the Deep Learning model using the generated features.")
        if st.button("3. Start LSTM Model Training", disabled=st.session_state['model_trained']):
            # Execute model training (updates MODEL_STATE and session_state['model_trained'])
            build_and_train_model(st.session_state['final_data'].copy(), N_STEPS)
    else:
        st.warning("Feature engineering must be complete before training the model.")

# TAB 3: RISK REPORT
with tab3:
    st.header("Actionable Hunger Risk Forecast")
    
    if st.session_state['model_trained']:
        st.markdown("""
        The report below shows the forecast for the next available month (the first unobserved timestep) 
        for all analyzed commodities and countries.
        """)
        
        # Execute prediction and classification
        report = predict_and_classify(st.session_state['final_data'].copy(), N_STEPS)
        
        if report is not None:
            st.subheader(f"Forecast for: {report['Prediction_Month'].iloc[0].strftime('%B %Y')}")
            
            # Highlight the highest risk countries
            def color_risk(val):
                color = 'red' if val in ['Emergency', 'Crisis'] else 'orange' if val == 'Stress' else 'white'
                return f'background-color: {color}'

            st.dataframe(
                report.head(50).style.applymap(color_risk, subset=['Hunger_Risk_Level']),
                column_config={
                    "Price_USD": st.column_config.NumberColumn("Last Known Price", format="$%.3f"),
                    "Predicted_Price_USD": st.column_config.NumberColumn("Predicted Price", format="$%.3f"),
                    "Volatility_3M": st.column_config.NumberColumn("Volatility (STD)", format="%.4f"),
                },
                use_container_width=True
            )
            
            st.markdown("---")
            st.markdown(f"**Total Time Series Predicted:** {len(report)}")
    
    elif st.session_state['final_data'] is not None:
        st.info("The model must be trained in the 'Train Model' tab before generating the final report.")
