import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Page Configuration

st.set_page_config(
    page_title="Sales Demand Forecasting",
    page_icon="📈",
    layout="wide"
)
 
# Load Model

@st.cache_resource
def load_model():
    return joblib.load("sales_forecasting_model.pkl")

model = load_model()

# Load Dataset
 

@st.cache_data
def load_data():
    df = pd.read_csv("sales_data.csv")
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    return df

# Title
 
st.title("📈 Sales Demand Forecasting")

st.write(
    "Forecast future sales using historical sales patterns and machine learning."
)

# Forecast Settings
st.subheader("🔮 Forecast Settings")

forecast_months = st.slider(
    "Number of months to forecast",
    min_value=1,
    max_value=12,
    value=12
)

st.write(
    f"Forecasting the next **{forecast_months} month(s)**"
)

# Load Data
try:
    df = load_data()

except FileNotFoundError:
    st.error(
        "sales_data.csv was not found in the project folder."
    )
    st.stop()

# Validate Required Columns
required_columns = [
    "Order Date",
    "Sales"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    st.error(
        f"Missing required columns: {', '.join(missing_columns)}"
    )
    st.stop()

# Monthly Sales
monthly_sales = (
    df.groupby(
        pd.Grouper(
            key="Order Date",
            freq="ME"
        )
    )["Sales"]
    .sum()
    .reset_index()
)

# Feature Engineering
monthly_sales["year"] = (
    monthly_sales["Order Date"].dt.year
)

monthly_sales["month"] = (
    monthly_sales["Order Date"].dt.month
)

monthly_sales["quarter"] = (
    monthly_sales["Order Date"].dt.quarter
)

# Lag Features
monthly_sales["lag_1"] = (
    monthly_sales["Sales"].shift(1)
)

monthly_sales["lag_2"] = (
    monthly_sales["Sales"].shift(2)
)

monthly_sales["lag_3"] = (
    monthly_sales["Sales"].shift(3)
)

monthly_sales["lag_12"] = (
    monthly_sales["Sales"].shift(12)
)

# Rolling Features
monthly_sales["rolling_mean_3"] = (
    monthly_sales["Sales"]
    .shift(1)
    .rolling(3)
    .mean()
)

monthly_sales["rolling_std_3"] = (
    monthly_sales["Sales"]
    .shift(1)
    .rolling(3)
    .std()
)


monthly_sales.dropna(inplace=True)

# Generate Forecast
if st.button(
    "🔮 Generate Forecast",
    type="primary"
):

    last_data = monthly_sales.copy()

    future_predictions = []

    for i in range(forecast_months):

        # Last available row
        last_row = last_data.iloc[[-1]]

        # Next month
        next_month = (
            last_row["Order Date"]
            + pd.DateOffset(months=1)
        )

        # Create New Prediction Row
        new_row = pd.DataFrame({
            "Order Date": next_month,
            "year": next_month.dt.year,
            "month": next_month.dt.month,
            "quarter": next_month.dt.quarter,

            "lag_1": last_row["Sales"].values,

            "lag_2": last_row["lag_1"].values,

            "lag_3": last_row["lag_2"].values,

            "lag_12": last_row["lag_12"].values,

            "rolling_mean_3":
                last_row["rolling_mean_3"].values,

            "rolling_std_3":
                last_row["rolling_std_3"].values
        })

        # Model Features
        X_new = new_row[
            [
                "year",
                "month",
                "quarter",
                "lag_1",
                "lag_2",
                "lag_3",
                "lag_12",
                "rolling_mean_3",
                "rolling_std_3"
            ]
        ]


        # Prediction
        pred = model.predict(X_new)[0]


        # Add prediction to row

        new_row["Sales"] = pred


        # Store prediction

        future_predictions.append(new_row)

        # Add prediction to historical data
        last_data = pd.concat(
            [
                last_data,
                new_row
            ],
            ignore_index=True
        )


    # Combine Forecast Data

    future_predictions = pd.concat(
        future_predictions,
        ignore_index=True
    )


    # Store result in session state

    st.session_state["forecast"] = future_predictions

    st.success(
        f"Successfully generated a {forecast_months}-month forecast!"
    )

# Display Forecast
 
if "forecast" in st.session_state:

    future_predictions = st.session_state["forecast"]

    # Forecast Data
    st.subheader("📊 Forecast Data")

    display_data = future_predictions[
        [
            "Order Date",
            "Sales"
        ]
    ].copy()

    display_data["Order Date"] = (
        display_data["Order Date"]
        .dt.strftime("%Y-%m-%d")
    )

    display_data["Sales"] = (
        display_data["Sales"]
        .round(2)
    )

    st.dataframe(
        display_data,
        hide_index=True,
        use_container_width=True
    )

    # Forecast Metrics

    total_sales = (
        future_predictions["Sales"]
        .sum()
    )

    average_sales = (
        future_predictions["Sales"]
        .mean()
    )

    max_sales = (
        future_predictions["Sales"]
        .max()
    )


    col1, col2, col3 = st.columns(3)


    col1.metric(
        "Total Forecasted Sales",
        f"₹{total_sales:,.0f}"
    )


    col2.metric(
        "Average Monthly Sales",
        f"₹{average_sales:,.0f}"
    )


    col3.metric(
        "Highest Monthly Forecast",
        f"₹{max_sales:,.0f}"
    )

    # Forecast Trend
  
    st.subheader("📈 Forecast Trend")


    fig, ax = plt.subplots(
        figsize=(10, 5)
    )


    ax.plot(
        future_predictions["Order Date"],
        future_predictions["Sales"],
        marker="o"
    )


    ax.set_title(
        "Future Sales Forecast"
    )

    ax.set_xlabel(
        "Date"
    )

    ax.set_ylabel(
        "Sales"
    )


    plt.xticks(
        rotation=45
    )

    plt.tight_layout()

    st.pyplot(fig)