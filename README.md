📈 Sales Demand Forecasting

A machine learning project that forecasts future monthly sales using historical sales patterns and an XGBoost regression model. The project includes a Streamlit frontend for generating and visualizing future sales forecasts.

Features
Monthly sales aggregation
Time-based features: year, month, and quarter
Lag features: lag_1, lag_2, lag_3, lag_12
Rolling statistics: 3-month mean and standard deviation
XGBoost-based sales forecasting
Recursive multi-month forecasting
Forecast horizon selection from 1–12 months
Forecast summary metrics
Interactive forecast visualization
Streamlit web interface
Tech Stack
Python
Pandas
NumPy
Matplotlib
Scikit-learn
XGBoost
Joblib
Streamlit
Project Structure
Sales-Demand-Forecasting/
│
├── app.py
├── sales_data.csv
├── sales_forecasting_model.pkl
├── salesForcasting.ipynb
└── README.md
How It Works
Historical Sales Data
        ↓
Monthly Sales Aggregation
        ↓
Feature Engineering
        ↓
Lag & Rolling Features
        ↓
XGBoost Model
        ↓
Recursive Forecasting
        ↓
Future Sales Predictions
        ↓
Streamlit Dashboard

The forecasting process generates one future month at a time. Each prediction is added to the forecasting data and used to generate the next month's prediction.

Run Locally

Clone the repository and install the dependencies:

pip install pandas numpy matplotlib scikit-learn xgboost joblib streamlit

Run the Streamlit application:

streamlit run app.py

The application allows users to select the number of months to forecast and generate the corresponding sales predictions.

Model

The project uses XGBRegressor for forecasting.

The trained model is saved as:

sales_forecasting_model.pkl

The Streamlit application loads this saved model rather than retraining it.

Forecast Output

The dashboard provides:

Forecasted sales for each future month
Total forecasted sales
Average monthly sales
Highest monthly forecast
Future sales trend visualization
Future Improvements
Add model comparison with other forecasting algorithms
Add confidence intervals
Add interactive Plotly visualizations
Deploy the Streamlit application
Add automated data updates
Improve forecasting accuracy with additional features