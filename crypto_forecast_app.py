import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor

st.set_page_config(page_title="Crypto Forecast", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --bg: #07111f;
        --panel: rgba(10, 19, 35, 0.78);
        --panel-strong: rgba(14, 28, 48, 0.96);
        --border: rgba(148, 163, 184, 0.18);
        --text: #e5eefb;
        --muted: #8fa3bf;
        --accent: #4fd1c5;
        --accent-2: #60a5fa;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(79, 209, 197, 0.18), transparent 32%),
            radial-gradient(circle at top right, rgba(96, 165, 250, 0.16), transparent 28%),
            linear-gradient(180deg, #05101d 0%, #07111f 48%, #0b1527 100%);
        color: var(--text);
    }

    .stApp, .stApp p, .stApp label, .stApp span, .stApp div {
        color: var(--text);
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(7, 17, 31, 0.98), rgba(8, 20, 36, 0.96));
        border-right: 1px solid var(--border);
    }

    .hero {
        background: linear-gradient(135deg, rgba(10, 19, 35, 0.92), rgba(18, 32, 57, 0.82));
        border: 1px solid var(--border);
        border-radius: 24px;
        padding: 1.5rem 1.5rem 1.25rem 1.5rem;
        box-shadow: 0 18px 60px rgba(0, 0, 0, 0.28);
        margin-bottom: 1rem;
    }

    .hero h1 {
        margin: 0;
        font-size: 2.4rem;
        line-height: 1.05;
        letter-spacing: -0.04em;
    }

    .hero p {
        margin: 0.75rem 0 0;
        color: var(--muted);
        font-size: 1rem;
        max-width: 56rem;
    }

    .section-title {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        margin: 1rem 0 0.75rem;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.18em;
        color: var(--muted);
    }

    .stat-card {
        background: linear-gradient(180deg, rgba(11, 22, 39, 0.92), rgba(11, 22, 39, 0.76));
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 1rem 1rem 0.9rem;
        box-shadow: 0 12px 34px rgba(0, 0, 0, 0.2);
        height: 100%;
    }

    .stat-label {
        color: var(--muted);
        font-size: 0.8rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.45rem;
    }

    .stat-value {
        font-size: 1.7rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        margin: 0;
    }

    .stat-subtext {
        margin-top: 0.35rem;
        color: var(--muted);
        font-size: 0.9rem;
    }

    .positive {
        color: #67e8a8;
    }

    .negative {
        color: #fda4af;
    }

    .data-shell {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 0.75rem;
        box-shadow: 0 12px 34px rgba(0, 0, 0, 0.18);
    }

    button[title="Edit"],
    button[aria-label="Edit"],
    a[title="Edit"],
    a[aria-label="Edit"],
    a[href*="edit"],
    a[href*="?edit"],
    a[href*="&edit"] {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_data(ttl=900)
def load_price_history(symbol: str, years: int) -> pd.DataFrame:
    history = yf.download(symbol, period=f"{years}y", progress=False, auto_adjust=False)
    return history.dropna()


def build_sequences(series: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    feature_rows = []
    target_rows = []

    for index in range(window, len(series)):
        feature_rows.append(series[index - window:index, 0])
        target_rows.append(series[index, 0])

    if not feature_rows:
        return np.empty((0, window)), np.empty((0,))

    return np.array(feature_rows), np.array(target_rows)


st.sidebar.markdown("### Controls")
crypto = st.sidebar.selectbox("Select crypto", ["BTC-USD", "ETH-USD", "BNB-USD"])
years = st.sidebar.slider("Training history (years)", 1, 5, 2)
window = st.sidebar.slider("Look-back window (days)", 30, 120, 60, step=10)
iterations = st.sidebar.slider("Training iterations", 1, 15, 5)

st.sidebar.markdown("### Model")
st.sidebar.caption("The app trains a lightweight scikit-learn forecasting model that deploys cleanly on Streamlit Cloud.")

st.markdown(
    """
    <div class="hero">
        <h1>Crypto Forecast</h1>
        <p>Track recent market history, compare predicted and actual prices, and tweak the training window to see how the forecast responds.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.spinner("Loading market data..."):
    df = load_price_history(crypto, years)

if df.empty:
    st.error("No market data was returned for the selected asset and time range.")
    st.stop()

close_prices = df[["Close"]].dropna()
close_series = close_prices.iloc[:, 0].astype(float)

if len(close_series) <= window:
    st.warning("The selected look-back window is too large for the available data. Reduce the window or increase the training history.")
    st.stop()

latest_close = float(close_series.iloc[-1])
previous_close = float(close_series.iloc[-2]) if len(close_series) > 1 else latest_close
price_change = latest_close - previous_close
price_change_pct = (price_change / previous_close * 100) if previous_close else 0.0
period_high = float(close_series.max())
period_low = float(close_series.min())
first_close = float(close_series.iloc[0])
period_return_pct = ((latest_close - first_close) / first_close * 100) if first_close else 0.0

st.markdown('<div class="section-title">Market Overview</div>', unsafe_allow_html=True)
overview_col_1, overview_col_2, overview_col_3, overview_col_4 = st.columns(4)

metric_style = lambda label, value, subtext, trend_class="": f"""
    <div class="stat-card">
        <div class="stat-label">{label}</div>
        <p class="stat-value {trend_class}">{value}</p>
        <div class="stat-subtext">{subtext}</div>
    </div>
"""

with overview_col_1:
    st.markdown(metric_style("Latest close", f"${latest_close:,.2f}", f"{crypto} on the latest session"), unsafe_allow_html=True)

with overview_col_2:
    change_class = "positive" if price_change >= 0 else "negative"
    st.markdown(
        metric_style(
            "Recent change",
            f"{price_change:+,.2f} ({price_change_pct:+.2f}%)",
            "Compared with the previous close",
            change_class,
        ),
        unsafe_allow_html=True,
    )

with overview_col_3:
    st.markdown(metric_style("Range", f"${period_low:,.2f} - ${period_high:,.2f}", "Minimum and maximum close in the selected period"), unsafe_allow_html=True)

with overview_col_4:
    return_class = "positive" if period_return_pct >= 0 else "negative"
    st.markdown(
        metric_style(
            "Period return",
            f"{period_return_pct:+.2f}%",
            "From the first available close to the latest close",
            return_class,
        ),
        unsafe_allow_html=True,
    )

st.markdown('<div class="section-title">Forecast workspace</div>', unsafe_allow_html=True)

data = close_series.to_numpy().reshape(-1, 1)

# Scale
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Build sequences
features, targets = build_sequences(scaled_data, window)

if len(features) == 0:
    st.error("Not enough data to build a training set with the selected look-back window.")
    st.stop()

# Model
model = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    max_iter=iterations * 100,
    random_state=42,
)

train_col, forecast_col = st.columns([2, 1])

with st.spinner("Training the model..."):
    model.fit(features, targets)

# Predict
predicted_scaled = model.predict(features).reshape(-1, 1)
predicted_prices = scaler.inverse_transform(predicted_scaled)
actual_prices = data[window:]

latest_window = scaled_data[-window:].reshape(1, -1)
next_scaled_price = float(model.predict(latest_window)[0])
next_price = float(scaler.inverse_transform([[next_scaled_price]])[0][0])

comparison_df = pd.DataFrame(
    {
        "Actual": actual_prices.flatten(),
        "Predicted": predicted_prices.flatten(),
    },
    index=close_prices.index[window:],
)

with train_col:
    st.markdown('<div class="data-shell">', unsafe_allow_html=True)
    st.line_chart(comparison_df, height=360)
    st.caption("Predicted values are trained on the selected look-back window and compared against the actual close series.")
    st.markdown('</div>', unsafe_allow_html=True)

with forecast_col:
    st.markdown('<div class="data-shell">', unsafe_allow_html=True)
    st.metric("Forecast for next close", f"${next_price:,.2f}")
    st.write("This is the model's one-step-ahead estimate based on the latest window of prices.")

    with st.expander("Model settings", expanded=False):
        st.write(f"Crypto: {crypto}")
        st.write(f"Training history: {years} year(s)")
        st.write(f"Look-back window: {window} days")
        st.write(f"Training iterations: {iterations}")
        st.write(f"Training rows: {len(features):,}")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-title">Recent market data</div>', unsafe_allow_html=True)
st.markdown('<div class="data-shell">', unsafe_allow_html=True)
market_table = df.tail(12).reset_index()
st.dataframe(market_table, use_container_width=True, hide_index=True, height=320)
st.caption("Latest rows are shown here so you can inspect the underlying market data used by the forecast.")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-title">Insights</div>', unsafe_allow_html=True)
insight_col_1, insight_col_2 = st.columns(2)

with insight_col_1:
    st.info(
        "The actual series shows the observed close values. The forecast series is the model's in-sample fit over the same window length."
    )

with insight_col_2:
    st.info(
        "For a stronger signal, increase training history and reduce the look-back window. Larger windows can smooth noise but need more data."
    )

st.success("Forecast ready. Adjust the controls in the sidebar to explore a different market view.")
