import yfinance as yf
import pandas as pd
import numpy as np
import time

f1000 = pd.read_csv('data/fortune1000_2024.csv')
f1000 = f1000[f1000['CompanyType'] == 'Public']
dates_df = pd.read_csv('data/dates_df2.csv')

tickers = f1000['Ticker']
# tickers = np.setdiff1d(tickers, dates_df['ticker'].unique())
## get earnings dates dataframe ##
# yf.Ticker('WMT').get_earnings_dates(limit=20).reset_index()

# dates_df = pd.DataFrame()
ticker_err = []
date_lim = 30
for ticker in tickers:
    print(f'now collecting {ticker} dates')
    df = yf.Ticker(ticker)
    # edates = df.get_earnings_dates(limit=date_lim).reset_index()
    try:
        edates = df.get_earnings_dates(limit=date_lim).reset_index()
        edates['ticker'] = ticker
        dates_df = pd.concat([dates_df, edates])
        dates_df = dates_df.dropna()
    except KeyError:
    # except (KeyError, AttributeError) as error:
        print('KeyError occurred')
        ticker_err.append(ticker)
        time.sleep(60)
        pass

# dates_df.to_csv('data/dates_df2.csv', index=False)

tickers = dates_df['ticker'].unique().tolist()
# multi_data = yf.download(tickers, start="2017-12-01", end="2024-09-17", auto_adjust=True)
# aapl = yf.download('AAPL', start="2017-12-01", end="2024-09-17", multi_level_index=False, auto_adjust=True)
# aapl['ticker'] = 'aapl'

tickers = tickers + ['SPY']
stocks_df = pd.DataFrame()
i = 1
for ticker in tickers:
    # print(ticker)
    print(f'now fetching {ticker} data')
    print(f'{i} out of {len(tickers)}')
    df = yf.download(ticker, start="2017-12-01", end="2025-03-12", multi_level_index=False, auto_adjust=True)
    df['ticker'] = ticker
    stocks_df = pd.concat([stocks_df, df])
    # if i in (300, 600):
    #     print('saving data')
    #     stocks_df = stocks_df.reset_index()
    #     stocks_df.to_csv(f'data/stocks_df{i}.csv', index=False)
    #     stocks_df = pd.DataFrame()
    i += 1

stocks_df = stocks_df.reset_index()
stocks_df.to_csv('data/stocks_df2.csv', index=False)