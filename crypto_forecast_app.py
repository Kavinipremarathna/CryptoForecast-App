import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor

st.set_page_config(page_title="Crypto Forecast", page_icon="📈", layout="wide")
st.title("📈 Real-Time Crypto Price Forecast")

crypto = st.selectbox("Select Crypto", ["BTC-USD", "ETH-USD", "BNB-USD"])
years = st.slider("Training data (years)", 1, 5, 2)
window = st.slider("Look-back window (days)", 30, 120, 60, step=10)
epochs = st.slider("Training iterations", 1, 15, 5)

with st.spinner("Downloading data..."):
    df = yf.download(crypto, period=f"{years}y")
    df = df.dropna()

st.subheader("Latest Prices")
st.write(df.tail())

data = df[['Close']].values

# Scale
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Build sequences
x, y = [], []
for i in range(window, len(scaled_data)):
    x.append(scaled_data[i-window:i, 0])
    y.append(scaled_data[i, 0])

x, y = np.array(x), np.array(y)
x_2d = x

# Model
model = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    max_iter=epochs * 100,
    random_state=42,
)

with st.spinner("Training model..."):
    model.fit(x_2d, y)

# Predict
pred = model.predict(x_2d).reshape(-1, 1)
pred_prices = scaler.inverse_transform(pred)

actual_prices = data[window:]

st.subheader("📊 Actual vs Predicted")
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(actual_prices, label="Actual")
ax.plot(pred_prices, label="Predicted")
ax.set_title(f"{crypto} Forecast")
ax.set_xlabel("Time")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)

st.success("Done ✅ Try changing crypto / window / iterations to see differences.")
