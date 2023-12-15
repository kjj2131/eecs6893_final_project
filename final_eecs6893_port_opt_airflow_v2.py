
# ===========================================================
# --- proj3c_airflow_dry_edshare_v7_csv_report_today_ed02.py---
# ===========================================================


# ===========================================================
# ---- IMPORT Packages
# ===========================================================
# --- location /home/edsharecourse/projdatabucket/
import finnhub
finnhub_client = finnhub.Client(api_key="ckjk2vpr01qq18o5vn40ckjk2vpr01qq18o5vn4g")

import numpy as np
import pandas as pd

from numpy.linalg import inv

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.metrics import r2_score
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score

from sklearn.metrics import mean_squared_error


from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import RepeatedKFold

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestRegressor

import warnings
warnings.filterwarnings("ignore")


import xgboost as xgb

from datetime import date, datetime, tzinfo, timedelta
import datetime as dt

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import yfinance as yf

import pypfopt

from pypfopt import risk_models
from pypfopt import plotting

from pypfopt import EfficientFrontier

from pypfopt import risk_models
from pypfopt import plotting

from pypfopt import black_litterman, risk_models
from pypfopt import BlackLittermanModel, plotting

from pypfopt import EfficientFrontier, objective_functions

import pickle 

from google.cloud import storage
import os

# --------------------------------------------------------------------------- 
from datetime import datetime, timedelta
from textwrap import dedent
import time

# The DAG object; we'll need this to instantiate a DAG
from airflow import DAG

# Operators; we need this to operate!
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
# These args will get passed on to each operator
# You can override them on a per-task basis during operator initialization


####################################################
# DEFINE PYTHON FUNCTIONS
####################################################

# -------- parameters ---------------------
count = 0  # --- this is example of global variable in the def

tickers_pull = ['XOM', 'HON', 'RTX', 'AMZN', 'PEP', 'UNH', 'JNJ', 'V', 'NVDA', 'AAPL', 'MSFT', 'GOOGL']
# tickers_pull = ['XOM', 'HON', 'RTX', 'AMZN', 'PEP', 'UNH']
# ======= initial parameters ============

noofdays_test = 250
# ======= lag 30 days for live runs ============
noofdays_lag_live = 30

# Get today's date
today_real = date.today()
# after mid night 
# today = today_real - timedelta(days = 1)
today = today_real
# print("Today is: ", today)

# Yesterday date
yesterday = today - timedelta(days = 1)
# print("Yesterday was: ", yesterday)

# --- fixed mcaps dec 12
mcaps = {'XOM': 398158000000,'HON': 132107000000,'RTX': 117750000000,'AMZN': 1508000000000,'PEP': 230729000000,'UNH': 502863000000,'JNJ': 373273000000, 'V': 527197000000,'NVDA': 1152000000000,'AAPL': 3004000000000,'MSFT': 2760000000000,'GOOGL': 1676000000000}

# ---- 3 diff --- python commands ----
# ------  def correct_sleeping_function():   def sleeping_cmd_fn():   def count_cmd_fn():
# -------  def print_cmd_fn():  def wrong_sleeping_function():

# ----- regular functions ----------
def convert_date(x):
    func_date = dt.datetime.fromtimestamp(x).strftime('%Y-%m-%d')
    return func_date


def convert_date_time(x):
    func_date_time = dt.datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S')
    return func_date_time


def categorise(row):
    if row['pred'] > 0 and row['sentiment'] ==1:
        return 1
    elif row['pred'] < 0 and row['sentiment'] ==-1:
        return 1
    elif row['pred'] > 0 and row['sentiment'] ==-1:
        return 0
    elif row['pred'] < 0 and row['sentiment'] ==1:
        return 0
    return 0.5

# -------- python execution functions ----------------

def step1_pull_stock_data():
    tickers = tickers_pull
    ohlc = yf.download(tickers, period="60mo") # 60 months
    prices = ohlc["Adj Close"]
    # prices
    # market_prices_BL_raw = yf.download("SPY", period="60mo")["Adj Close"]
    returns_full = prices.pct_change()
    returns= returns_full.dropna()
    # returns
    # returns_full =  returns_full.reset_index()
    # df = returns        
    market_prices_BL_pull = yf.download("SPY", period="60mo")["Adj Close"]
    market_prices_BL_raw = market_prices_BL_pull
    returns.to_csv('/home/edsharecourse/projdatabucket/df.csv', index ='Date')
    prices.to_csv('/home/edsharecourse/projdatabucket/prices.csv', index ='Date')
    market_prices_BL_raw.to_csv('/home/edsharecourse/projdatabucket/market_prices_BL_raw.csv', index='Date') 
    
def step2_ml_xg():
    # ----- XGBoost Parameters ------------
    xg_max_depth=50
    xg_learning_rate=0.8
    xg_reg_lambda=8
    xg_subsample=0.4
    xg_grow_policy="lossguide"    
    # ----- ONE DAY PREDICTION:: LIVE  ------------    
    df = pd.read_csv('/home/edsharecourse/projdatabucket/df.csv', index_col='Date')    
    # training_prices_x = {}
    # training_prices_y = {}
    live_prices_x = {}
    live_prices_y = {}
    live_pred_x = {}
    pred_live_30 = df.tail(noofdays_lag_live)
    # live_pred_x = {}
    for col in df.columns:
        company_live = df[col].to_numpy()
        company_live_30 = pred_live_30[col].to_numpy()
        company_live_x = [company_live[i:i+15] for i in range(len(company_live)-15)]
        company_live_y = [company_live[i+1] for i in range(14,len(company_live)-1)]
        # company_live_pred_x = [company_live[i:i+30+1] for i in range(len(company_live)-30+1)]
        company_live_pred_x= [company_live_30[i:i+15] for i in range(len(company_live)-15)]
        live_prices_x[col] = company_live_x
        live_prices_y[col] = company_live_y
        live_pred_x[col] = [company_live_pred_x[0]]
        # live_pred_x
    # ----- ONE DAY PREDICTION LIVE ------------
    next_day_preds ={}
    for col in df.columns:
        bst = xgb.XGBRegressor(max_depth=xg_max_depth, learning_rate=xg_learning_rate,
                               reg_lambda=xg_reg_lambda, subsample=xg_subsample, grow_policy=xg_grow_policy)
        bst = bst.fit(live_prices_x[col],live_prices_y[col])
        next_day_preds[col] = bst.predict(live_pred_x[col])
    # next_day_preds    
    next_day_preds_df = pd.DataFrame.from_dict(next_day_preds)
    next_day_preds_df.index = ['pred']*1
    # next_day_preds_df
    next_day_preds_df.to_csv('/home/edsharecourse/projdatabucket/next_day_preds_df.csv', index=True)
    with open('/home/edsharecourse/projdatabucket/saved_dictionary.pkl', 'wb') as f_next_day_preds:
        pickle.dump(next_day_preds, f_next_day_preds)

def step3_news_feed():    
    stock_list = tickers_pull
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model_finbert = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    # stock_list = ['XON']
    sentiment_list =[]
    # stock_list =  ['XOM']
    # count=3
    for stock in stock_list:
        # ohlc = yf.download(tickers_pull, start="2018-11-13", end="2023-11-13")
        # count = 0 ==> news_series = finnhub_client.company_news(stock, _from="2022-11-04", to="2023-10-20")
        # count = 1 ==> news_series = finnhub_client.company_news(stock, _from="2022-11-04", to="2023-09-12")
        # count = 2 ==> news_series = finnhub_client.company_news(stock, _from="2022-11-04", to="2023-07-25")
        news_series = finnhub_client.company_news(stock, _from=yesterday, to=today)
        if news_series == []:
            print("stock news "+stock+" is empty")
            sentiment_stock  = 0
            # result_daily[stock] = np.select(conditions, sentiment_values)
        else:
            news_series_df = pd.DataFrame.from_dict(news_series)
            result_date = []
            result_news = []
            for index, row in news_series_df.iterrows():
                result_news.append(row['headline'])
                result_date.append(convert_date(row['datetime']))
            # print(result_news)
            result_date
            inputs = tokenizer(result_news, padding = True, truncation = True, return_tensors='pt')
            outputs = model_finbert(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            # print(predictions)
            pred_arr = predictions.detach().cpu().numpy()
            result_date_df = pd.DataFrame({'date' : result_date}, columns=['date'])
            # print("this is result_date_df")
            # print(result_date_df)
            result_news_df = pd.DataFrame(data = pred_arr, columns = ["Positive", "Negative", "Neutral"])
            # print("this is result_news_df")
            # print(result_news_df)
            result_combined = pd.concat([result_date_df, result_news_df], axis=1)
            # print("result_combined")
            # print(result_combined)
            # print(type(result_combined))
            # result_daily = result_combined.groupby('date')['Positive', 'Negative', 'Neutral'].mean()
            # for daily use this
            result_daily_nodate = result_combined.drop(columns=['date'])
            result_daily = result_daily_nodate.mean()
            print("this is result_daily")
            result_daily
            conditions = [(result_daily['Positive'] > result_daily['Negative']) & (result_daily['Positive'] > result_daily['Neutral']),
            (result_daily['Negative'] > result_daily['Positive']) & (result_daily['Negative'] > result_daily['Neutral']),
            (result_daily['Neutral'] > result_daily['Positive']) & (result_daily['Neutral'] > result_daily['Positive'])]
            sentiment_values = [1, -1, 0]
            result_daily[stock] = np.select(conditions, sentiment_values)
            sentiment_stock  = np.ndarray.item(np.array([result_daily[stock]]))
        sentiment_list.append(sentiment_stock)
        print(stock)
        print (sentiment_list)
    sentiment_list_row =[sentiment_list]
    sentiment_daily = pd.DataFrame(sentiment_list_row, columns=stock_list, index=['sentiment']*1)
    print (sentiment_list)
    sentiment_daily.to_csv('/home/edsharecourse/projdatabucket/sentiment_daily.csv', index=True)
    
def step4_bl_weight():  
    # def black_litterman_weight(next_day_preds_df, sentiment_daily):
    # tickers_pull= ['HON', 'PEP', 'RTX', 'UNH', 'XOM']
    prices_BL = pd.read_csv('/home/edsharecourse/projdatabucket/prices.csv', index_col='Date')    
    df_market_csv_read = pd.read_csv('/home/edsharecourse/projdatabucket/market_prices_BL_raw.csv', index_col='Date')
    market_prices_BL_raw=df_market_csv_read[df_market_csv_read.columns[0]]    
    next_day_preds_df = pd.read_csv('/home/edsharecourse/projdatabucket/next_day_preds_df.csv', index_col=['Unnamed: 0'])
    sentiment_daily = pd.read_csv('/home/edsharecourse/projdatabucket/sentiment_daily.csv', index_col=['Unnamed: 0'])    
    with open('/home/edsharecourse/projdatabucket/saved_dictionary.pkl', 'rb') as f_next_day_preds: next_day_preds = pickle.load(f_next_day_preds)    
    # next_day_preds_df = next_day_preds_df.reset_index()
    next_day_preds_df_T = next_day_preds_df.T
    # next_day_preds_df_T
    sentiment_daily_T = sentiment_daily.T
    # sentiment_daily_T
    combine_pred_sentiment = next_day_preds_df_T.join(sentiment_daily_T)
    # combine_pred_sentiment
    combine_pred_sentiment['confidence'] = combine_pred_sentiment.apply(lambda row: categorise(row), axis=1)
    confidence = combine_pred_sentiment['confidence'].T
    # print("stock confidence "+str(confidence)) 
    # prices_BL = prices_BL_raw.head(1115+x)
    # prices_BL =  prices
    # market_prices_BL = market_prices_BL_raw.head(1115+x)
    market_prices_BL = market_prices_BL_raw
    viewdict_ML2_GBReg = next_day_preds
    confidences_ML2_GBReg = confidence
    S_BL = risk_models.CovarianceShrinkage(prices_BL).ledoit_wolf()
    delta = black_litterman.market_implied_risk_aversion(market_prices_BL)
    # delta
    market_prior = black_litterman.market_implied_prior_returns(mcaps, delta, S_BL)
    # market_prior
    bl_ML2_GBReg = BlackLittermanModel(S_BL, pi=market_prior, absolute_views=viewdict_ML2_GBReg, omega="idzorek", view_confidences=confidences_ML2_GBReg)
    bl = bl_ML2_GBReg
    omega_ML2_GBReg = bl.omega
    # We are using the shortcut to automatically compute market-implied prior
    bl_ML2_GBReg = BlackLittermanModel(S_BL, pi="market", market_caps=mcaps, risk_aversion=delta,
    absolute_views=viewdict_ML2_GBReg, omega=omega_ML2_GBReg)
    # Posterior estimate of returns
    ret_bl_ML2_GBReg = bl.bl_returns()
    # ret_bl_ML2_GBReg
    rets_df_ML2_GBReg = pd.DataFrame([market_prior, ret_bl_ML2_GBReg, pd.Series(viewdict_ML2_GBReg)],
     index=["Prior", "Posterior", "Views"]).T
    # rets_df_ML2_GBReg
    # rets_df_ML2_GBReg.plot.bar(figsize=(12,8));
    S_bl_ML2_GBReg = bl_ML2_GBReg.bl_cov()
    # plotting.plot_covariance(S_bl_ML2_GBReg);
    ef = EfficientFrontier(ret_bl_ML2_GBReg, S_bl_ML2_GBReg)
    ef.add_objective(objective_functions.L2_reg)
    ef.max_sharpe()
    weights_ML2_GBReg = ef.clean_weights()
    # weights_ML2_GBReg
    weights_ML2_GBReg_df = pd.DataFrame(weights_ML2_GBReg, index=[today])
    print(weights_ML2_GBReg_df)
    # weights_ML2_GBReg_df_out = weights_ML2_GBReg_df_out.append(weights_ML2_GBReg_df)
    weights_ML2_GBReg_df.to_csv('/home/edsharecourse/projdatabucket/weights_ML2_GBReg_df.csv')
    #   ---------------- ADDED FOR REPORT and STACKING UP -----------------------------
    w_hist = pd.read_csv('/home/edsharecourse/projdatabucket/w_hist.csv', index_col='Date')
    df = pd.read_csv('/home/edsharecourse/projdatabucket/df.csv', index_col='Date')
    df_ret_calc = df.add_prefix('ret_')
    df_ret_calc.index = pd.to_datetime(df_ret_calc.index).strftime('%m/%d/%Y')
    # df_ret_calc
    w_hist_calc = w_hist.add_prefix('w_')
    w_hist_calc.index = pd.to_datetime(w_hist_calc.index).strftime('%m/%d/%Y')
    port = w_hist_calc.merge(df_ret_calc, how='inner', left_index=True, right_index=True)
    stock_list = tickers_pull
    for stock_str in stock_list:
        port['x_'+stock_str] = port['w_'+stock_str]*port['ret_'+stock_str]
        port = port.drop('w_'+stock_str, axis=1)
        port = port.drop('ret_'+stock_str, axis=1)
    port_wavg_ret = port.sum(axis=1)
    sharpe_ratio_port = port_wavg_ret.mean()/port_wavg_ret.std()
    rm_ret = market_prices_BL_raw.pct_change()
    rm_ret = rm_ret.dropna()
    rm_ret.index = pd.to_datetime(rm_ret.index).strftime('%m/%d/%Y')
    rm = port_wavg_ret.to_frame().join(rm_ret.to_frame())
    sharp_ratio_rm = rm['Adj Close'].mean()/rm['Adj Close'].std()
    sharpe_summary_list = [[ port_wavg_ret.index.max(), sharpe_ratio_port, sharp_ratio_rm]]
    sharpe_summary = pd.DataFrame(sharpe_summary_list, columns=['Date', 'Sharpe Ratio Portfolio (30 days)', 'Sharpe Ratio S&P (30 days)'])
    sharpe_summary_csv = sharpe_summary.set_index('Date')
    sharpe_summary_csv.to_csv('/home/edsharecourse/projdatabucket/sharpe_summary_csv.csv', index_label='Date')
    return_summary_list = [[ port_wavg_ret.index.max(), port_wavg_ret.mean(), port_wavg_ret.std(), rm['Adj Close'].mean(), rm['Adj Close'].std()]]
    return_summary = pd.DataFrame(return_summary_list, columns=['Date', 'Portfolio Avg Return (30 days)', 'Portfolio Return Std (30 days)', 'S&P Avg Return  (30 days)', 'S&P Return Std (30 days)'])
    return_summary_csv = return_summary.set_index('Date')
    return_summary_csv.to_csv('/home/edsharecourse/projdatabucket/return_summary_csv.csv', index_label='Date')
    # ---- finally we can stack up the weights
    weights_ML2_GBReg_df.index.name='Date'
    weights_ML2_GBReg_df.index = pd.to_datetime(weights_ML2_GBReg_df.index).strftime('%m/%d/%Y')
    # w_hist_app = w_hist.append(weights_ML2_GBReg_df)
    w_hist_app = pd.concat([w_hist, weights_ML2_GBReg_df])
    w_hist = w_hist_app.tail(30)
    w_hist.to_csv('/home/edsharecourse/projdatabucket/w_hist.csv', index_label='Date')

def step5_save_report():  
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/home/edsharecourse/eecs6893-399919-d331dd0aefd0.json" 
    storage_client = storage.Client()
    buckets = list(storage_client.list_buckets())
    bucket = storage_client.get_bucket("eecs6893_project") # your bucket name
    blob = bucket.blob('apache_data/weights_ML2_GBReg_df.csv')
    blob.upload_from_filename('/home/edsharecourse/projdatabucket/weights_ML2_GBReg_df.csv')
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/home/edsharecourse/eecs6893-399919-d331dd0aefd0.json" 
    storage_client = storage.Client()
    buckets = list(storage_client.list_buckets())
    bucket = storage_client.get_bucket("eecs6893_project") # your bucket name
    blob = bucket.blob('apache_data/w_hist.csv')
    blob.upload_from_filename('/home/edsharecourse/projdatabucket/w_hist.csv')
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/home/edsharecourse/eecs6893-399919-d331dd0aefd0.json" 
    storage_client = storage.Client()
    buckets = list(storage_client.list_buckets())
    bucket = storage_client.get_bucket("eecs6893_project") # your bucket name
    blob = bucket.blob('apache_data/return_summary_csv.csv')
    blob.upload_from_filename('/home/edsharecourse/projdatabucket/return_summary_csv.csv')  
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/home/edsharecourse/eecs6893-399919-d331dd0aefd0.json" 
    storage_client = storage.Client()
    buckets = list(storage_client.list_buckets())
    bucket = storage_client.get_bucket("eecs6893_project") # your bucket name
    blob = bucket.blob('apache_data/sharpe_summary_csv')
    blob.upload_from_filename('/home/edsharecourse/projdatabucket/sharpe_summary_csv.csv')   
    

############################################
# DEFINE AIRFLOW DAG (SETTINGS + SCHEDULE)
############################################

default_args = {
    'owner': 'ektanta',
    'depends_on_past': False,
    'email': ['et2676@columbia.edu@columbia.edu'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(seconds=30),
    # 'queue': 'bash_queue',
    # 'pool': 'backfill',
    # 'priority_weight': 10,
    # 'end_date': datetime(2016, 1, 1),
    # 'wait_for_downstream': False,
    # 'dag': dag,
    # 'sla': timedelta(hours=2),
    # 'execution_timeout': timedelta(seconds=300),
    # 'on_failure_callback': some_function,
    # 'on_success_callback': some_other_function,
    # 'on_retry_callback': another_function,
    # 'sla_miss_callback': yet_another_function,
    # 'trigger_rule': 'all_success'
}

with DAG(
    'final_eecs6893_port_opt_airflow_v2',
    default_args=default_args,
    description='final_eecs6893_port_opt_airflow_v2 DAG',
    schedule_interval='0 8 * * 1-5',  # == Here, set the schedule of running 8 am working days
    start_date=datetime(2021, 1, 1),          # == Here, set the start_date to present date or the past date
    catchup=False,
    tags=['example'],
) as dag:

##########################################
# DEFINE AIRFLOW OPERATORS
##########################################

    # t* examples of tasks created by instantiating operators

    tp_1 = PythonOperator(
        task_id='tp_1',
        python_callable=step1_pull_stock_data,
    )

    tp_2 = PythonOperator(
        task_id='tp_2',
        python_callable=step2_ml_xg,
    )

    tp_3 = PythonOperator(
        task_id='tp_3',
        python_callable=step3_news_feed,
    )

    tp_4 = PythonOperator(
        task_id='tp_4',
        python_callable=step4_bl_weight,
    )

    tp_5 = PythonOperator(
        task_id='tp_5',
        python_callable=step5_save_report,
    )
    tb_6 = BashOperator(
        task_id='tb_6',
        bash_command='rm ~/projdatabucket/df.csv ~/projdatabucket/prices.csv ~/projdatabucket/return_summary_csv.csv',
        retries=0,
    )

    tb_7 = BashOperator(
        task_id='tb_7',
        bash_command='rm ~/projdatabucket/sharpe_summary_csv.csv ~/projdatabucket/market_prices_BL_raw.csv',
        retries=0,
    )
    tb_8 = BashOperator(
        task_id='tb_8',
        bash_command='rm ~/projdatabucket/saved_dictionary.pkl ~/projdatabucket/next_day_preds_df.csv',
        retries=0,
    )

    tb_9 = BashOperator(
        task_id='tb_9',
        bash_command='rm ~/projdatabucket/sentiment_daily.csv ~/projdatabucket/weights_ML2_GBReg_df.csv',
        retries=0,
    )

##########################################
# DEFINE TASKS HIERARCHY
##########################################

    # task dependencies 

#    t1 >> [t2_1, t2_2, t2_3]
#    t2_1 >> t3_1
#    t2_2 >> t3_2
#    [t2_3, t3_1, t3_2] >> t4_1


tp_1 >> tp_2
tp_2 >> tp_3
tp_3 >> tp_4
tp_4 >> tp_5 
tp_5 >> tb_6 
tb_6 >> tb_7 
tb_7 >> tb_8 
tb_8 >> tb_9 

