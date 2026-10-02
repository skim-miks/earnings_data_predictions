import numpy as np
import pandas as pd

def calculate_stock_changes(df):
    df = df.sort_values(by=['ticker', 'Date'])
    df['daily_change'] = df.groupby('ticker')['Close'].pct_change()
    df['1w_change'] = df.groupby('ticker')['Close'].pct_change(periods=5)
    df['1m_change'] = df.groupby('ticker')['Close'].pct_change(periods=21)
    df['3m_change'] = df.groupby('ticker')['Close'].pct_change(periods=63)
    df['1yr_change'] = df.groupby('ticker')['Close'].pct_change(periods=252)

    df['52wk_high'] = df.groupby('ticker')['High'].rolling(window=252).max().reset_index(0, drop=True)
    df['52wk_low'] = df.groupby('ticker')['Low'].rolling(window=252).min().reset_index(0, drop=True)

    # market_returns = df[df['ticker'] == 'SPY'][['Date', 'daily_change']].copy()
    # df['corr_w_market'] = df.groupby('ticker')['daily_change'].rolling(window=30).corr(market_returns['daily_change']).reset_index(0, drop=True)

    df['sma20'] = df.groupby('ticker')['Close'].rolling(window=20).mean().reset_index(0, drop=True)
    df['sma50'] = df.groupby('ticker')['Close'].rolling(window=50).mean().reset_index(0, drop=True)
    df['sma200'] = df.groupby('ticker')['Close'].rolling(window=200).mean().reset_index(0, drop=True)
    df['sma20_change'] = (df['Close'] - df['sma20']) / df['sma20']
    df['sma50_change'] = (df['Close'] - df['sma50']) / df['sma50']
    df['sma200_change'] = (df['Close'] - df['sma200']) / df['sma200']

    return df

def get_ratios(df):
    df = df.sort_values(by=['ticker', 'Date_x'])
    df['shifted_rep_eps'] = df.groupby('ticker')['Reported EPS'].shift(1)
    df['TTM_EPS'] = df.groupby('ticker')['shifted_rep_eps'].rolling(window=4).sum().reset_index(0, drop=True)
    df['PE_Ratio'] = df['Close'] / df['TTM_EPS']
    df['Forward_PE'] = df['Close'] / df['EPS Estimate']
    return df

def rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def volatility(series, window=5):
    return series.rolling(window=window).std()
