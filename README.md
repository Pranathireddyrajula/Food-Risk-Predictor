🌍 Food-Risk-Predictor Engine
Executive Summary
The Food-Risk-Prediction (FRP) Engine is an advanced, end-to-end data science application designed to serve as an Early Warning System (EWS) for acute food insecurity globally.

It uses a Hybrid Deep Learning (LSTM) Model to forecast staple food prices and combines these forecasts with exogenous geopolitical and vulnerability data (Global Hunger Index scores) to generate a clear, Actionable Risk Classification (Low, Stress, Crisis, Emergency) for humanitarian agencies and policymakers.

Key Technical Highlights
Model: Hybrid LSTM (Long Short-Term Memory) Neural Network for non-linear time-series forecasting.

Feature Engineering: Calculated market Volatility (Rolling Standard Deviation) and Price Momentum (Lagged Features).

Data Pipeline: Robust, modular Python scripts and Streamlit application for resilient handling of messy, real-world data sources (WFP, GHI).

Deployment: Interactive Streamlit Web Application for easy demonstration and consumption of forecasts.

1. Project Motivation and Impact
The primary challenge in humanitarian logistics is reacting proactively rather than reactively to crises. High food prices are a symptom of instability (conflict,), but predicting the severity of the human impact requires combining economic data with vulnerability data.

The  Engine solves this by:

Forecasting Price Shock: Predicting the magnitude of the next price change.

Classifying Risk: Using that forecast, along with a country’s inherent vulnerability (Hunger_Score), to classify the region's overall risk level. This transforms raw numbers into high-impact, immediate decisions.

2. Technical Architecture and Pipeline
The solution is structured as a clear MLOps pipeline, demonstrating proficiency across data engineering, modeling, and deployment.

A. Data Sources
Source

Role

Example Feature

WFP Market Prices

Target Variable (Price_USD) & Time Series

date, countryiso3, usdprice

Global Hunger Index (GHI)

Exogenous Risk Factor

Hunger_Score (Static baseline vulnerability)

B. MLOps Workflow
Step

Script/Tool

Key Technique

1. Data Ingestion & Cleaning

app.py (via Streamlit)

Robust Column Matching (handling messy headers like #date), Data Aggregation (Daily → Monthly), Explicit Type Casting.

2. Feature Engineering

app.py (Feature Function)

Calculation of Rolling Volatility (STD) and Lagged Features for time-series momentum.

3. Modeling & Training

app.py (Training Function)

LSTM on 3D Tensor Data (Samples×Timesteps×Features), using MinMaxScaler and EarlyStopping to prevent overfitting.

4. Prediction & Deployment

app.py (Predict Function)

Inverse Scaling of predictions, custom Risk Classification Logic, and Streamlit UI presentation.

3. Getting Started
Prerequisites
Python 3.8+

pip package manager

Installation
Clone the Repository:

git clone https://github.com/Pranathireddyrajula/Food-Risk-Predictor.git
cd food-risk-predictor

Create and Activate Virtual Environment:

python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

Install Dependencies:

pip install streamlit pandas numpy tensorflow scikit-learn

Running the Application
Place Raw Data: Add your raw WFP Price CSV and GHI Score CSV files to the designated data/ folder (or upload them directly in the app).

Launch Streamlit:

streamlit run app.py

Navigate: Open the browser window and follow the three steps in the sidebar:

Upload Files.

Clean Data & Engineer Features.

Start LSTM Model Training.

View the final Risk Report tab.

4. Advanced Technical Discussion
LSTM Sequence Management
A crucial component was handling the Inhomogeneous Shape challenge. The LSTM model requires inputs of identical shape (N_STEPS = 3). Custom Python logic was implemented to group data by Country and Commodity and dynamically filter out any time series lacking the required 3 historical timesteps, ensuring a uniform 3D Tensor input for the model.

Risk Model Logic
The final risk classification is a composite decision, not a pure price forecast. It uses the following hierarchy:

Risk Level=f(Hunger Score 
base
​
 ,Forecasted Price Spike,Current Market Volatility)
Regions with a high historical Hunger_Score are automatically elevated (e.g., to 'Crisis' baseline).

A high forecasted price spike (e.g., >20%) combined with high Volatility_3M triggers a promotion to the next highest risk level (e.g., 'Stress' → 'Crisis').

This system successfully transforms predictive accuracy into humanitarian relevance.

5. Repository Structure
food-risk-predictor/
├── data/                       # Stores input/output data (cleaned data is generated here)
├── src/                        # Contains core pipeline logic (modular Python scripts, e.g., 01_data_prep.py)
├── app.py                      # Main Streamlit application and deployment file
└── README.md                   # This file

