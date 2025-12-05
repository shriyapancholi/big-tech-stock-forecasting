# app.py  (clean version – handles yfinance MultiIndex columns)

import pandas as pd
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
import streamlit as st

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="Big Tech Stock Forecaster",
    layout="wide",
)

st.title("📈 Big Tech Stock Price Forecaster")
st.write(
    "Select a company on the left, choose how many years to forecast, "
    "and the app will download historical data and generate a Prophet forecast."
)

# ---------------------------
# Sidebar controls
# ---------------------------
st.sidebar.header("Settings")

TICKERS = {
    "Apple (AAPL)": "AAPL",
    "Google (GOOG)": "GOOG",
    "Microsoft (MSFT)": "MSFT",
    "Amazon (AMZN)": "AMZN",
    "Meta (META)": "META",
    "NVIDIA (NVDA)": "NVDA",
    "Tesla (TSLA)": "TSLA",
}

company_label = st.sidebar.selectbox("Choose a company", list(TICKERS.keys()))
ticker = TICKERS[company_label]

years = st.sidebar.slider("Years to forecast into the future", 1, 5, 2)
periods = years * 365

st.sidebar.write(f"Ticker: **{ticker}**")

# ---------------------------
# 1. Fetch stock data with yfinance
# ---------------------------

@st.cache_data
def fetch_stock_data(ticker_symbol: str) -> pd.DataFrame:
    """
    Download daily OHLC data using yfinance and return a clean
    dataframe with columns: ds (datetime), y (close price).
    """

    df = yf.download(ticker_symbol, start="2015-01-01", progress=False)

    # Index -> Date column
    df = df.reset_index()

    # If columns are MultiIndex (e.g. ('Close', 'AAPL')), flatten them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Keep only Date + Close
    df = df[["Date", "Close"]].copy()

    # Build Prophet-format columns
    df["ds"] = pd.to_datetime(df["Date"], errors="coerce")
    df["y"] = df["Close"].astype(float)

    # Keep only ds,y and drop NaNs
    df = df[["ds", "y"]].dropna()

    # Sort by date
    df = df.sort_values("ds").reset_index(drop=True)

    return df


st.subheader(f"Downloading data for {ticker}")
try:
    data = fetch_stock_data(ticker)
except Exception as e:
    st.error(f"Could not download data for {ticker}: {e}")
    st.stop()

# ---------------------------
# 2. Historical price chart
# ---------------------------
st.subheader("Historical Closing Price")
st.line_chart(data.set_index("ds")["y"])

# ---------------------------
# 3. Train Prophet model
# ---------------------------
st.subheader("Training forecast model...")
m = Prophet(daily_seasonality=True)
m.fit(data)

future = m.make_future_dataframe(periods=periods)
forecast = m.predict(future)

# ---------------------------
# 4. Forecast plot
# ---------------------------
st.subheader(f"{years}-Year Forecast (including history)")
fig_forecast = plot_plotly(m, forecast)
st.plotly_chart(fig_forecast, use_container_width=True)

# ---------------------------
# 5. Future-only table
# ---------------------------
last_date = data["ds"].max()
future_forecast = forecast[forecast["ds"] > last_date][
    ["ds", "yhat", "yhat_lower", "yhat_upper"]
]

st.subheader("Future forecast values (only)")
st.dataframe(future_forecast.head(30))

# ---------------------------
# 6. Trend & seasonality
# ---------------------------
st.subheader("Trend & Seasonality Components")
fig_components = m.plot_components(forecast)
st.pyplot(fig_components)