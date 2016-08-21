# %run "/Users/tfoster/Documents/p/fin/Data/System/FX655_Backtest v0.1.py"
# TKF
# Created: 8.8.2016

import pandas as pd
from pandas import *
import numpy as np
import sys

# ==============================================================
# ASSUMPTIONS
# ==============================================================
show_stats = True
const_file_path = 'Documents/p/fin/Data/System/FX/current/'
output_data_file_name = 'C:/Users/tfoster/Desktop/Temp/fx.csv'
output_data_file_name2 = 'C:/Users/tfoster/Desktop/Temp/fx_stats.csv'

initial_equity = 26000.00
pct_equity_risk = .015
atr_periods = 14 # ATR
ATR = .0112 # Will need to calculate this

start_date = '2016-06-01'
end_date = '2016-12-31'

# instruments = str('AUDUSD GBPUSD EURUSD USDCHF USDCAD USDJPY')
instruments = str('AUDUSD')
ticks = [instruments]
sma_period = 5
lstInstruments = []

# ==============================================================
# FUNCTIONS
# ==============================================================
def get_ticker_data(start_date, end_date, price_type):
	"""Return historical prices from txt"""
	prices = pd.DataFrame()
	
	for t in ticks:
		symbols=t.split()
		for symb in symbols:
			data = pd.read_csv(const_file_path + symb + '.txt', index_col = 'Date', parse_dates = True)
			prices[symb] = data[price_type][start_date:end_date]
			lstInstruments.append(symb)
	return prices

def calc_true_range (df):
	df['TR1'] = abs(df['High']-df['Low']).fillna(0)
	df['TR2'] = abs(df['High']-df['Close'].shift()).fillna(0)
	df['TR3'] = abs(df['Low']-df['Close'].shift()).fillna(0)
	df['TrueRange'] = df[['TR1','TR2','TR3']].max(axis=1)
	return df

def calc_atr (df, atr_periods):
	calc_true_range(df)
	df['ATR']=pd.Series.rolling(df['TrueRange'], atr_periods).mean()
	return df
	
def max_drawdown(df):
	roll_max = pd.Series.rolling(df, len(df.index)-1, min_periods=1).max()
	roll_min = pd.Series.rolling(df, len(df.index)-1, min_periods=1).min()
	dd = roll_max + roll_min
	# print roll_max, roll_min, dd
	return roll_max
	
# ==============================================================
# MAIN
# ==============================================================
if __name__ == "__main__":
	
	#
	# LOAD PRICE DATA
	#
	fields = ['Date', 'Open', 'High', 'Low', 'Close']
	data = pd.read_csv(const_file_path+'AUDUSD'+'.txt', index_col='Date', parse_dates=True, usecols = fields)
	data['DayOfWk'] = data.index.weekday
	data['WeekOfYear'] = data.index.week	
	prices = data[start_date:end_date]
	#print prices

	#
	# ENTRY EVALUATION
	#
	
	# SMA PRICE
	prices['SMA'] = pd.Series.rolling(prices['Close'], sma_period, min_periods=sma_period).mean()

	# WEEKLY PERFORMANCE
	prices['PctChange'] = (prices.loc[(prices.DayOfWk == 4), 'Close'] - prices.Open.shift(4))/prices.Open.shift(4) # % change between Monday (0) and Friday (4)
	prices.loc[prices.PctChange > 0,'LvS'] = 'L' 
	prices.loc[prices.PctChange < 0,'LvS'] = 'S'	
	prices.LvS = prices.LvS.shift(1) # Action to take for the following week. That is, after the Friday (4)
	prices['LvS'] = prices['LvS'].ffill()
	
	prices.loc[prices.LvS=='L', 'IsSma'] = prices.Close > prices['SMA']
	prices.loc[prices.LvS=='S', 'IsSma'] = prices.Close < prices['SMA']

	# ATR
	calc_atr(prices, atr_periods)
	
	# INITIAL STOP
	prices.loc[(prices.DayOfWk==0) & (prices.LvS=='L'), 'InitialStop'] = prices.Open-prices.ATR
	prices.loc[(prices.DayOfWk==0) & (prices.LvS=='S'), 'InitialStop'] = prices.Open+prices.ATR
	
	# ENTRY PRICE - Different than open in that it's static for the duration of the position for PvL calc_atr
	prices.loc[prices.DayOfWk==0, 'EntryPrice'] = prices['Open']*prices.IsSma
	prices['EntryPrice']=prices['EntryPrice'].ffill()
	
	# QTY / EQUITY
	# Initial Capital * Risk Percent / (Entry price - Initial StopIteration
	prices.ix[atr_periods, 'Equity'] = initial_equity # intialized equity
	prices['Equity']=prices['Equity'].ffill()
	prices['Qty']=0 # intialized Qty

	# Loop for shares and total updates -- SLOW
	# ALL the magic happens here. Need to figure out how to manage other exits than SMA
	for i in range(atr_periods+1, len(prices)):
		if prices.ix[i, 'DayOfWk'] == 0:
			# QTY 
			prices.ix[i , 'Qty'] = (prices.ix[i-1, 'Equity'] * pct_equity_risk / (prices.ix[i, 'EntryPrice']-prices.ix[i, 'InitialStop'])).round(-4) 
			prices['Qty']=prices['Qty'].ffill()
			prices['Qty']=prices['Qty'].fillna(0)
			# PROFIT OR LOSS
			prices.ix[i, 'PvL']=(prices.ix[i, 'Close']-prices.ix[i, 'EntryPrice']) * prices.ix[i, 'Qty'] * prices.ix[i, 'IsSma']
			# EQUITY CHANGE
			prices.ix[i, 'Equity'] = prices.ix[i-1, 'Equity'] + prices.ix[i, 'PvL']
		else:
			prices.ix[i, 'Qty']=prices.ix[i-1, 'Qty'] 
			if prices.ix[i-1, 'IsSma']:
				if prices.ix[i, 'IsSma']:
					prices.ix[i, 'PvL']=(prices.ix[i, 'Close']-prices.ix[i-1, 'Close']) * prices.ix[i, 'Qty'] 
				else: 
					prices.ix[i, 'PvL']=(prices.ix[i-1, 'SMA']-prices.ix[i-1, 'Close']) * prices.ix[i, 'Qty'] 
			else:
				prices.ix[i, 'PvL'] = 0
			prices.ix[i, 'Equity'] = prices.ix[i-1, 'Equity'] + prices.ix[i, 'PvL']

	# TOTAL DOLLAR RISK
	prices['$Risk']=abs((prices['Open']-prices['InitialStop'])*prices['Qty'])
	prices['$Risk']=prices['$Risk'].ffill() 
	
	# PERCENT RISK
	prices.loc[prices.DayOfWk==0, '%Risk']=prices['$Risk']/prices['Equity']
	
	# CUMULATIVE PROFIT / LOSS
	prices['CumPL']=prices['PvL'].cumsum()
	
	# POSITION PvL / WEEKLY
	prices.loc[(prices.DayOfWk==4) & (prices['$Risk'] != 0), 'PPvL']=prices['CumPL']-prices['CumPL'].shift(5)
	
	# R-MULT
	prices.loc[(prices.DayOfWk==4) & (prices['$Risk'] != 0), 'R-Mult']= prices['PPvL'] / prices['$Risk']
	
	# CUMULATIVE R
	prices['Cum-R']=prices['R-Mult'].cumsum().ffill()
	
	#===========================================================
	# Stats
	#===========================================================
	if show_stats:
		sd = ['Date Range', start_date+' to '+end_date]
		cpl = ['Gain / Loss', prices.CumPL.iloc[-1]]
		rm = ['R-Mult', prices['Cum-R'].iloc[-1]]
		exp = ['Expectancy', prices['R-Mult'].mean()]
		stdv = ['Std. Dev', prices['R-Mult'].std()]
		trades = prices.loc[prices['R-Mult'] !=0, 'R-Mult'].count()
		tc = ['Trades', trades]
		sqn = ['SQN', prices['R-Mult'].mean() / prices['R-Mult'].std() * np.sqrt(prices.loc[prices['R-Mult'] !=0, 'R-Mult'].count())]
		wins = prices.loc[prices['PPvL'] > 0, 'PPvL'].count()
		w = ['Wins', wins]
		wp = ['Win %', float(wins)/float(trades)]
		losses = prices.loc[prices['PPvL'] < 0, 'PPvL'].count()
		l = ['+R : -R', float(wins) / losses]
		max_dd = max_drawdown(prices['Cum-R'])
		# print 'Max DD', max_dd
		df_stats = pd.DataFrame([sd, cpl, rm, exp, stdv, w, tc, wp, l, sqn], columns=('Stat', 'Value'))
		print df_stats
		df_stats.to_csv(output_data_file_name2)
	# print prices.ix[:, (0,3,4,6,7,8,9,14,15,16)].tail()
	# print prices.loc[:, ('Open', 'Close', 'EntryPrice', 'SMA(5)', 'LvS', 'InitialStop', '$Risk', '%Risk', 'Qty', 'IsSma', 'PvL', 'Equity', 'DayOfWk')]
	prices.to_csv(output_data_file_name)