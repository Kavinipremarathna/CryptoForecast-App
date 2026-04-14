# 🚀 Crypto Forecast System (LSTM + FastAPI + Streamlit)

## 📌 Overview
This project is a full-stack machine learning system that performs real-time cryptocurrency price forecasting using LSTM networks.

It demonstrates software engineering principles such as modular architecture, API design, and containerization.

---

## 🧱 Architecture

Frontend (Streamlit) → Backend API (FastAPI) → ML Services (TensorFlow)

---

## ⚙️ Features

- Real-time crypto data retrieval using yfinance
- LSTM-based time series forecasting
- Proper train-test split to avoid data leakage
- RMSE evaluation metric
- EarlyStopping for overfitting control
- REST API for model inference
- Interactive UI for parameter tuning
- Dockerized deployment

---

## 🛠 Tech Stack

- Python
- TensorFlow / Keras
- FastAPI
- Streamlit
- Scikit-learn
- Docker

---

## ▶️ How to Run

### Option 1: Without Docker

```bash
pip install -r requirements.txt
uvicorn backend.main:app --reload
streamlit run app.py# CryptoForecast-App
