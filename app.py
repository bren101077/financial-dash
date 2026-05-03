import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from datetime import timedelta

st.set_page_config(page_title="ML Stock Predictor", layout="wide")
st.title("🕯️ Candlestick Analysis & 5-Day ML Prediction")

ticker = st.sidebar.text_input("Enter Ticker", "AAPL").upper()
period = st.sidebar.selectbox("Training Data Period", ("6mo", "1y", "2y", "5y"), index=1)

@st.cache_data
def get_data(symbol, time_period):
    df = yf.download(symbol, period=time_period, interval="1d", auto_adjust=True)
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    
    # Technical Indicators
    df['EMA14'] = df['Close'].ewm(span=14, adjust=False).mean()
    df['EMA27'] = df['Close'].ewm(span=27, adjust=False).mean()
    
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))
    
    tr = pd.concat([df['High']-df['Low'], abs(df['High']-df['Close'].shift()), abs(df['Low']-df['Close'].shift())], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()
    return df

def predict_future(df, days_to_predict=5):
    # Prepare data: use day numbers as X to find the trend
    df_pred = df.reset_index()
    df_pred['Day_Num'] = np.arange(len(df_pred))
    
    X = df_pred[['Day_Num']].values
    y = df_pred['Close'].values
    
    # Train Linear Regression model
    model = LinearRegression()
    model.fit(X, y)
    
    # Generate next 5 days
    last_day = df_pred['Day_Num'].iloc[-1]
    future_days = np.array([last_day + i for i in range(1, days_to_predict + 1)]).reshape(-1, 1)
    future_preds = model.predict(future_days)
    
    # Create future dates (skipping weekends for simplicity)
    last_date = df_pred['Date'].iloc[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, days_to_predict + 1)]
    
    return pd.DataFrame({'Date': future_dates, 'Predicted_Close': future_preds})

if ticker:
    df = get_data(ticker, period)
    if df is not None:
        # Get Predictions
        predictions = predict_future(df)
        chart_df = df.reset_index()
        
        # UI: Top Metrics
        last_price = chart_df['Close'].iloc[-1]
        pred_price = predictions['Predicted_Close'].iloc[-1]
        change = ((pred_price - last_price) / last_price) * 100
        
        col1, col2 = st.columns(2)
        col1.metric("Current Price", f"${last_price:.2f}")
        col2.metric("5-Day Prediction", f"${pred_price:.2f}", f"{change:+.2f}%")

        # Main Chart
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2])

        # 1. Candlestick + EMAs + Prediction Line
        fig.add_trace(go.Candlestick(x=chart_df['Date'], open=chart_df['Open'], high=chart_df['High'], 
                                     low=chart_df['Low'], close=chart_df['Close'], name='Historical'), row=1, col=1)
        fig.add_trace(go.Scatter(x=chart_df['Date'], y=chart_df['EMA14'], name='EMA 14', line=dict(color='#00d4ff', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=chart_df['Date'], y=chart_df['EMA27'], name='EMA 27', line=dict(color='#ff9900', width=1)), row=1, col=1)
        
        # Prediction line (connecting last actual to predicted)
        full_dates = pd.concat([chart_df['Date'].tail(1), predictions['Date']])
        full_preds = np.append(chart_df['Close'].iloc[-1], predictions['Predicted_Close'].values)
        fig.add_trace(go.Scatter(x=full_dates, y=full_preds, name='5-Day Forecast', 
                                 line=dict(color='white', dash='dash', width=2)), row=1, col=1)

        # 2. RSI & 3. ATR
        fig.add_trace(go.Scatter(x=chart_df['Date'], y=chart_df['RSI'], name='RSI', line=dict(color='#b400ff')), row=2, col=1)
        fig.add_trace(go.Scatter(x=chart_df['Date'], y=chart_df['ATR'], name='ATR', line=dict(color='yellow')), row=3, col=1)

if ticker:
    ticker1 = yf.Ticker(ticker)
    news = ticker1.news
    
    st.subheader(f"Latest News for {ticker}")
    
    if news:
        for item in news:
            # Extract content from the news dictionary
            content = item.get('content', {})
            title = content.get('title', 'No Title')
            click_url_obj = content.get('clickThroughUrl') or {}
            link = click_url_obj.get('url', '#')
            
            # Display news as markdown with a link
            st.markdown(f"**[{title}]({link})**")
            # st.write(f"Source: {provider}")
            st.divider()
    else:
        st.write("No news found for this ticker.")

        fig.update_layout(height=800, template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
