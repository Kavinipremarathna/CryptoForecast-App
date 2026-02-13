import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

st.set_page_config(page_title="Crypto Forecast", page_icon="📈", layout="wide")
st.title("📈 Real-Time Crypto Price Forecast (LSTM)")

crypto = st.selectbox("Select Crypto", ["BTC-USD", "ETH-USD", "BNB-USD"])
years = st.slider("Training data (years)", 1, 5, 2)
window = st.slider("Look-back window (days)", 30, 120, 60, step=10)
epochs = st.slider("Epochs", 1, 15, 5)

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
x = np.reshape(x, (x.shape[0], x.shape[1], 1))

# Model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(x.shape[1], 1)),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")

with st.spinner("Training model..."):
    model.fit(x, y, epochs=epochs, batch_size=32, verbose=0)

# Predict
pred = model.predict(x, verbose=0)
pred_prices = scaler.inverse_transform(pred)

actual_prices = data[window:]

st.subheader("📊 Actual vs Predicted")
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(actual_prices, label="Actual")
ax.plot(pred_prices, label="Predicted")
ax.set_title(f"{crypto} Forecast (LSTM)")
ax.set_xlabel("Time")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)

st.success("Done ✅ Try changing crypto / window / epochs to see differences.")
