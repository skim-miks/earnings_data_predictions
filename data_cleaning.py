import pandas as pd
from datetime import datetime, timedelta
from functions import calculate_stock_changes, rsi, volatility, get_ratios

stocks = pd.read_csv('data/stocks_df2.csv')

stocks = calculate_stock_changes(stocks)
stocks['rsi_14'] = stocks.groupby('ticker')['Close'].apply(rsi)
stocks['volatility_wk'] = stocks.groupby('ticker')['Close'].apply(volatility, window=5)
stocks['volatility_1m'] = stocks.groupby('ticker')['Close'].apply(volatility, window=21)

# market_returns = stocks[stocks['ticker'] == 'SPY']['daily_change'].copy()
# stocks['corr_w_market'] = stocks.groupby('ticker')['daily_change'].transform(
#     lambda x: x.rolling(window=30).corr(market_returns).reset_index(0, drop=True))

stocks['Date'] = pd.to_datetime(stocks['Date']).dt.date
stocks['next_trading_date'] = stocks.groupby('ticker')['Date'].shift(-1)

dates = pd.read_csv('data/dates_df2.csv')
dates['Date'] = dates['Earnings Date'].str.split(' ').str[0]
dates['Date'] = pd.to_datetime(dates['Date']).dt.date
dates['Quarter'] = pd.to_datetime(dates['Date']).dt.quarter

f1000 = pd.read_csv('data/fortune1000_2024.csv')
f1000 = f1000[['Ticker', 'Sector', 'Industry', 'HeadquartersCity', 'HeadquartersState']]

# dates['Earnings Date'] = pd.to_datetime(dates['Earnings Date']).dt.date
df = dates.merge(stocks, on=['ticker', 'Date'], how='inner')
df = df.merge(stocks[['Date', 'ticker', 'daily_change']], left_on=['ticker', 'next_trading_date'], right_on=['ticker', 'Date'], how='inner')
df = get_ratios(df)

df = df.merge(f1000, left_on='ticker', right_on='Ticker', how='inner')
df.to_csv('data/combined.csv', index=False)

df['y'] = (df['daily_change_y'] >= 0.05).astype(int)
df['y'].sum() / len(df)
# df = df[df['Date_x'] > pd.Timestamp('2020-04-01')]

df = df[['ticker', 'EPS Estimate', 'TTM_EPS', 'PE_Ratio', 'Forward_PE', 'Open', 'High', 'Low', 'Volume', 'daily_change_x', 
         '1w_change', '1m_change', '3m_change', '1yr_change', '52wk_high', '52wk_low', 'sma20_change', 
         'sma50_change', 'sma200_change', 'rsi_14', 'volatility_wk', 'volatility_1m', 'Quarter',
         'Sector', 'Industry', 'HeadquartersState', 'y']]
df = df.dropna()
df = df[df['PE_Ratio'] < 10000]
df = df[df['PE_Ratio'] > -10000]

## split by covid dates
# df = df[df['Date'] >= pd.Timestamp('2020-04-01')]
# df2 = df[df['Date_x'] < pd.Timestamp('2020-04-01')]

df.head()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler

X = df.drop(columns=['y', 'ticker'])
X = pd.get_dummies(X)
y = df['y']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=31)

cols_to_scale = ['PE_Ratio', 'Forward_PE', 'Open', 'High', 'Low', 'Volume', '52wk_high', '52wk_low', 'rsi_14']
scaler = StandardScaler().fit(x_train[cols_to_scale])
x_train[cols_to_scale] = scaler.transform(x_train[cols_to_scale])
x_test[cols_to_scale] = scaler.transform(x_test[cols_to_scale])

clf = LogisticRegression(max_iter=1000).fit(x_train, y_train)
clf.score(x_test, y_test)

roc_auc_score(y_test, clf.predict_proba(x_test)[:, 1])
f1_score(y_test, clf.predict(x_test))
print(classification_report(y_test, clf.predict(x_test)))

clf_rf = RandomForestClassifier(n_estimators=1000).fit(x_train, y_train)
preds = clf_rf.predict(x_test)
roc_auc_score(y_test, clf_rf.predict_proba(x_test)[:, 1])
f1_score(y_test, preds)
print(classification_report(y_test, preds))

import xgboost as xgb

xgb_params = {
    'n_estimators': 5000,
    'learning_rate': 0.01,
    'booster': 'gbtree',
    'max_depth': 6,
    'objective': 'binary:logistic',
    'eval_metric': 'aucpr',
    'early_stopping_rounds': 250
}
clf_xgb = xgb.XGBClassifier(**xgb_params)
clf_xgb.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose=100)
preds = clf_xgb.predict(x_test)
roc_auc_score(y_test, clf_xgb.predict_proba(x_test)[:, 1])
f1_score(y_test, preds)
print(classification_report(y_test, preds))

importances = pd.Series(clf_xgb.feature_importances_, index=x_train.columns).sort_values(ascending=False)

importances.head(15).plot(kind='barh')
plt.title("Top 15 Feature Importances")
plt.gca().invert_yaxis()
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Plot the importance scores
feature_importance[:20].plot.bar()

plt.show()

import catboost as cat
from catboost import CatBoostClassifier, Pool

cat_feats = ['Quarter', 'Sector', 'Industry', 'HeadquartersState']
train_pool = Pool(x_train, y_train, cat_features=cat_feats)
test_pool = Pool(x_test, y_test, cat_features=cat_feats)

clf_cat = CatBoostClassifier(iterations=5000, learning_rate=0.01, depth=5, eval_metric='AUC', early_stopping_rounds=250)
clf_cat.fit(train_pool, eval_set=test_pool, verbose=100)
preds = clf_cat.predict(test_pool)
roc_auc_score(y_test, clf_cat.predict_proba(test_pool)[:, 1])
f1_score(y_test, preds)
print(classification_report(y_test, preds))

TOP = 25

feature_importance = clf_cat.feature_importances_
sorted_idx = np.argsort(feature_importance)
fig = plt.figure(figsize=(10, 8))
plt.barh(np.arange(len(sorted_idx))[-TOP:], feature_importance[sorted_idx][-TOP:], align='center')
plt.yticks(np.arange(len(sorted_idx))[-TOP:], np.array(x_train.columns)[sorted_idx][-TOP:])
plt.title(f'Feature Importance - Top {TOP}')
plt.show()

x_test_w_prob = x_test.copy()
x_test_w_prob['y'] = y_test
x_test_w_prob['preds'] = clf_xgb.predict_proba(x_test)[:, 1]
x_test_w_prob.head(40)

# trial = [2.36, 6.29, 29.95, 23.07, 193.92, 203.15, 186.44, 125402855, -0.0729, 
#          -0.141, -0.1951, -0.2362, 0.1003, 260.1, 164.44, -0.1301, -0.1775, -0.1774, 
#          25.18, 0.0364, 0.0296]

# test = pd.DataFrame([trial], columns=x_test.columns)
# clf_xgb.predict_proba(test)[:, 1]