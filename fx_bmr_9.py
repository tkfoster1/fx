# %run /Users/tfoster/Documents/p/fin/Python/Backtester/FX/fx_bmr_9.py
# TKF
# 7.11.2014
# Stooq.com/db/h

import sys
import datetime as dt
import pandas as pd
import pandas.io.data as web
from pandas import *
import numpy as np
import math
from collections import deque
from StringIO import StringIO

# ==============================================================
# ASSUMPTIONS
# ==============================================================
end_date = dt.datetime.today() #datetime(2014, 8, 31) 
start_date = end_date - pd.DateOffset(days=50)

top_n = 2
return_period1 = 5 # Days
return_period2 = 1#22
rp_weight1 = 1#.7
rp_weight2 = 0#.3
sma_period = 5

# currencies = str('AUDUSD EURCHF EURGBP EURJPY EURUSD GBPUSD USDCAD USDCHF USDJPY')
currencies = str('AUDUSD EURCHF EURGBP EURJPY EURUSD GBPUSD JPYUSD USDAUD USDCAD USDCHF USDEUR USDGBP USDJPY')
#currencies = str('AUDUSD USDAUD EURCHF CHFEUR EURGBP GBPEUR EURJPY JPYEUR EURUSD USDEUR GBPUSD USDGBP USDCAD CADUSD USDCHF CHFUSD USDJPY JPYUSD')
### MAJORS ONLY
currencies = str('AUDUSD USDAUD EURUSD USDEUR GBPUSD USDGBP USDCAD CADUSD USDCHF CHFUSD USDJPY JPYUSD')
ticks = [currencies]
const_file_path = 'C:/Users/tfoster/Documents/p/fin/Data/System/FX/current/'
number_of_columns = math.ceil(len(currencies)/7.0)

# ==============================================================
# FUNCTIONS
# ==============================================================
def get_ticker_data(start_date, end_date):
    """ """
    prices = pd.DataFrame()

    for t in ticks:
        symbols=t.split()
        for symb in symbols:
            data = pd.read_csv(const_file_path+symb+'.txt', index_col='Date', parse_dates=True, usecols=['Date', 'Close'])
            prices[symb] = data['Close'][start_date: end_date]
    return prices

def simple_moving_average(prices, sma_period):
    sma = pd.rolling_mean(prices, sma_period, min_periods=sma_period)
    return sma

def calculate_true_range(df):
    ### Sourced from: http://www.gbquant.com/?p=9
    true_range=[]
    x=0
    for i in range(len(df)):
        if i==0:
            true_range.append(0)
        else:
            hilo=df.iloc[i,1]-df.iloc[i-1,2]
            hiclo=abs(df.iloc[i,1]-df.iloc[i-1,3])
            loclo=abs(df.iloc[i,2]-df.iloc[i-1,3])
            if hilo>hiclo:
                x=hilo
            elif hiclo>loclo:
                x=hiclo
            else:
                x=loclo
            true_range.append(x)
    df['True Range']=true_range
    
    return df

def calculate_ATR(df):
    number_of_periods = 14 # ! <--- IMPORTANT variable!
    calculate_true_range(df)
    df['ATR']=rolling_mean(df['True Range'], number_of_periods)
    return df

# ==============================================================
# MAIN
# ==============================================================
if __name__ == "__main__":

    ### Prices
    prices = get_ticker_data(start_date, end_date)
    current_price = prices.tail(1)
    return_period1_price = prices[-return_period1-1:-return_period1]
    return_period2_price = prices[-return_period2-1:-return_period2]
        
    ### SMA
    sma = simple_moving_average(prices, sma_period)
    sma_price = sma.tail(1)
    ma_signal = current_price > sma_price

    ### Performance
    r1_perf = return_period1_price.append(current_price).pct_change().tail(1)
    r2_perf = return_period2_price.append(current_price).pct_change().tail(1)
    blend_perf = r1_perf * rp_weight1 + r2_perf * rp_weight2
    perf_rank = blend_perf.rank(axis=1, ascending=False)
    
    ### ATR
    symb = 'USDCHF'
    with open (const_file_path+symb+'.txt', 'r') as f:
        q = deque(f, 15) ### Gets the last 14 rows...magically.

    df_atr = pd.read_csv(StringIO(''.join(q)), header=None, names=['Date', 'o', 'h', 'l', 'Close', 'v', 'oi'])

    aa=calculate_true_range(df_atr)
    bb=calculate_ATR(aa)
    #print bb['ATR']
    atr_stop=bb['ATR'].tail(1) * 1 # ! <--- IMPORTANT value!

    
    ### Outputs
    y = (perf_rank * ma_signal).T
    print 'sma_period =', sma_period, ' | return_period1 =', return_period1, ' | return_period2 =', return_period2, ' | rp_weight1 =', rp_weight1, ' | rp_weight2 =', rp_weight2
    z = current_price.append(sma_price).append(ma_signal).append(return_period1_price).append(r1_perf).append(perf_rank)
    
    lbls = ['current_price', 'sma_price', 'ma_signal', 'return_period1_price', 'r1_perf', 'perf_rank']
    z['labels'] = lbls
    print '\n', y
    print '\n', symb, ' ATR Stop\n', atr_stop
    print '\n', z
