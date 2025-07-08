# crypto_price_dashboard.py (Streamlit version)

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense

# --------------------------
# Data Loader
# --------------------------
def fetch_crypto_data(symbol, start="2015-01-01", end="2024-12-31"):
    df = yf.download(symbol, start=start, end=end)
    df = df[['Close']].dropna()
    return df

# --------------------------
# Preprocessing
# --------------------------
def preprocess_data(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)
    return scaled_data, scaler

def create_dataset(data, window_size=60):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i, 0])
        y.append(data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y

# --------------------------
# LSTM Model
# --------------------------
def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# --------------------------
# Dashboard App
# --------------------------
def main():
    st.title("📈 Crypto Price Predictor (BTC/ETH) using LSTM")
    crypto_choice = st.selectbox("Select Cryptocurrency", ["Bitcoin", "Ethereum"])
    symbol = "BTC-USD" if crypto_choice == "Bitcoin" else "ETH-USD"

    st.write(f"Fetching data for {crypto_choice}...")
    df = fetch_crypto_data(symbol)
    st.line_chart(df['Close'], use_container_width=True)

    scaled_data, scaler = preprocess_data(df)
    X, y = create_dataset(scaled_data)

    split_idx = int(len(X) * 0.9)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    st.write("Training LSTM model...")
    model = build_model((X_train.shape[1], 1))
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
    actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    mse = mean_squared_error(actual, predictions)
    mae = mean_absolute_error(actual, predictions)

    st.metric(label="Mean Absolute Error (MAE)", value=f"${mae:.2f}")
    st.metric(label="Mean Squared Error (MSE)", value=f"${mse:.2f}")

    st.subheader("Actual vs Predicted Prices")
    plot_df = pd.DataFrame({
        "Actual": actual.flatten(),
        "Predicted": predictions.flatten()
    })
    st.line_chart(plot_df, use_container_width=True)

if __name__ == "__main__":
    main()