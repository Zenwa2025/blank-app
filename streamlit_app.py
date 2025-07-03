import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from arch import arch_model
import random
import time

# ============================
# Core Parameters
# ============================
TRADING_DAYS_PER_YEAR = 252
TARGET_RETURN = 0.02
HORIZON_DAYS = 10

# ============================
# Sentiment Module (Mock API)
# ============================
def fetch_sentiment_score(ticker):
    return random.uniform(-0.5, 0.5)

# ============================
# Market Data and Analytics
# ============================
def fetch_data(ticker, period="1y"):
    d = yf.download(ticker, period=period, progress=False)["Close"]
    returns = d.pct_change().dropna()
    return d, returns

def estimate_drift(returns):
    return returns.mean()

def estimate_daily_sigma_garch(returns):
    model = arch_model(returns * 100, vol="Garch", p=1, o=0, q=1, rescale=False)
    res = model.fit(disp="off")
    var = res.forecast(horizon=1).variance.iloc[-1, 0]
    return np.sqrt(var) / 100

def monte_carlo_days(S0, mu, sigma, target=TARGET_RETURN, sims=5000, max_days=HORIZON_DAYS):
    days_to_hit = []
    for _ in range(sims):
        price = S0
        for day in range(1, max_days + 1):
            price *= (1 + mu + sigma * np.random.randn())
            if price >= S0 * (1 + target):
                days_to_hit.append(day)
                break
        else:
            days_to_hit.append(np.nan)
    days = np.nanmean(days_to_hit)
    prob_hit = np.nansum(~np.isnan(days_to_hit)) / sims
    return days, prob_hit

# ============================
# Report Generator
# ============================
def generate_report(tickers):
    report = []
    for ticker in tickers:
        try:
            d, returns = fetch_data(ticker)
            S0 = d.iloc[-1]
            mu = estimate_drift(returns)
            sigma_garch = estimate_daily_sigma_garch(returns)
            sentiment_score = fetch_sentiment_score(ticker)

            sentiment_adjustment = mu * (1 + sentiment_score * 2)
            adjusted_mu = mu + sentiment_adjustment

            expected_days_mc, prob_10d = monte_carlo_days(S0, adjusted_mu, sigma_garch, sims=2000)

            if expected_days_mc <= HORIZON_DAYS:
                report.append({
                    'Ticker': ticker,
                    'Price': round(S0, 2),
                    'μ (drift)': round(mu, 5),
                    'Sentiment Score': round(sentiment_score, 3),
                    'Adjusted μ': round(adjusted_mu, 5),
                    'σ_GARCH': round(sigma_garch, 5),
                    'Expected Days': round(expected_days_mc, 2),
                    'Prob(2% in 10d)': f"{round(prob_10d * 100, 2)}%"
                })
        except Exception as e:
            st.warning(f"Error processing {ticker}: {e}")
            continue

    df = pd.DataFrame(report)
    return df.sort_values('Expected Days')

# ============================
# Streamlit Dashboard
# ============================
st.set_page_config(page_title="CORE 2% Live Dashboard", layout="wide")

st.title("📈 CORE 2% Strategy Live Dashboard")

st.sidebar.header("Settings")
tickers_input = st.sidebar.text_input("Enter tickers separated by commas:", "AAPL,MSFT,JPM,GOOGL,XOM,TSLA")
refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 30, 600, 120)

if st.sidebar.button("Start Live Dashboard"):
    tickers = [ticker.strip().upper() for ticker in tickers_input.split(",")]
    placeholder = st.empty()

    while True:
        with placeholder.container():
            st.subheader(f"Updated Report - {time.strftime('%Y-%m-%d %H:%M:%S')}")
            report_df = generate_report(tickers)

            if not report_df.empty:
                st.dataframe(report_df.style
                             .background_gradient(cmap="YlGn", subset=["Prob(2% in 10d)"])
                             .background_gradient(cmap="coolwarm", subset=["Sentiment Score"]))
            else:
                st.write("No qualifying stocks with expected days ≤ 10.")

            st.write(f"Next refresh in {refresh_interval} seconds...")
            time.sleep(refresh_interval)
            st.experimental_rerun()
