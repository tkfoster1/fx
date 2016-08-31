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
pct_equity_risk = .03
atr_periods = 14 # ATR

start_date = '2015-11-01'
end_date = '2015-12-31'

ticker = 'AUDUSD'
# instruments = str('AUDUSD GBPUSD EURUSD USDCHF USDCAD USDJPY')
instruments = str(ticker)
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
	
def max_drawdown(df, RorPct):
	roll_max = pd.Series.rolling(df, len(df.index)-1, min_periods=1).max()
	if RorPct == 'R':
		daily_drawdown = roll_max - df
		max_daily_dd = pd.Series.rolling(daily_drawdown, len(df.index)-1, min_periods=1).max()
		max_daily_dd = max_daily_dd.max()
	else:
		daily_drawdown = df/roll_max-1.0
		max_daily_dd = pd.Series.rolling(daily_drawdown, len(df.index)-1, min_periods=1).min()
		max_daily_dd = max_daily_dd.min()
	# print roll_max, daily_drawdown, max_daily_dd
	print max_daily_dd
	return max_daily_dd
	
def position_update(i): 
	# ATR close
	close_atr = (prices.ix[i, 'Open'] - prices.ix[i, 'InitialStop']) * prices.ix[i, 'Qty']
	# SMA close
	close_sma = (prices.ix[i, 'Open'] - prices.ix[i, 'SMA']) * prices.ix[i, 'Qty']	
	pu = min(abs(close_atr), abs(close_sma))
	# Daily change
	dc = (prices.ix[i, 'Close'] - prices.ix[i, 'Open']) * prices.ix[i, 'Qty'] 
	return pu
	
# ==============================================================
# MAIN
# ==============================================================
if __name__ == "__main__":
	
	#
	# LOAD PRICE DATA
	#
	fields = ['Date', 'Open', 'High', 'Low', 'Close']
	data = pd.read_csv(const_file_path+ticker+'.txt', 
						index_col='Date', parse_dates=True, 
						usecols = fields
						)
	data['DayOfWk'] = data.index.weekday
	data['WeekOfYear'] = data.index.week	
	prices = data[start_date:end_date]
	#print prices

	#
	# ENTRY EVALUATION
	#
	
	# SMA PRICE
	prices['SMA'] = pd.Series.rolling(prices['Close'], 
					sma_period, min_periods=sma_period).mean()

	# PCT CHANGE
	prices['PctChange'] = (prices.loc[(prices.DayOfWk == 4), 'Close'] 
							- prices.Open.shift(4))/prices.Open.shift(4) # % change between Monday (0) and Friday (4)
	prices.loc[prices.PctChange > 0,'LvS'] = 'L' 
	prices.loc[prices.PctChange < 0,'LvS'] = 'S'	
	prices.LvS = prices.LvS.shift(1) # Direction to take for the following week. That is, after the Friday (4)
	prices['LvS'] = prices['LvS'].ffill()
	
	# IS SMA
	prices.loc[prices.LvS=='L', 'IsSma'] = prices.Close > prices['SMA']
	prices.loc[prices.LvS=='S', 'IsSma'] = prices.Close < prices['SMA']

	# ATR
	calc_atr(prices, atr_periods)
	
	# INITIAL STOP
	prices.loc[(prices.DayOfWk==0) & (prices.LvS=='L'), 'InitialStop'] = prices.Open - prices.ATR
	prices.loc[(prices.DayOfWk==0) & (prices.LvS=='S'), 'InitialStop'] = prices.Open + prices.ATR
	prices['InitialStop']=prices['InitialStop'].ffill()
	
	# QTY / EQUITY
	prices.ix[atr_periods, 'Equity'] = initial_equity 
	prices['Equity']=prices['Equity'].ffill()
	prices['Qty']=0 # intialized Qty

	# IS OPEN
	prices.loc[(prices.DayOfWk==0) & (prices.LvS=='L'), 'IsOpen'] = prices.Close.shift(1) >  prices.SMA.shift(1)
	prices.loc[(prices.DayOfWk==0) & (prices.LvS=='S'), 'IsOpen'] = prices.Close.shift(1) <  prices.SMA.shift(1)	
	
	# Loop for shares and total updates -- SLOW
	# ALL the magic happens here. Need to figure out how to manage other exits than SMA
	for i in range(atr_periods + 1, len(prices)):
		if prices.ix[i, 'DayOfWk'] == 0:
			# QTY 
			prices.ix[i , 'Qty'] = (prices.ix[i-1, 'Equity'] * pct_equity_risk / (prices.ix[i, 'Open'] - prices.ix[i, 'InitialStop'])).round(-4) 
			prices['Qty']=prices['Qty'].ffill()
			prices['Qty']=prices['Qty'].fillna(0)
			# PROFIT OR LOSS
			if prices.ix[i, 'IsOpen']:
				# if Long entry has Low < InitialStop, then exit at InitialStop
				if (prices.ix[i, 'LvS']=='L') & (prices.ix[i, 'Low'] < prices.ix[i, 'InitialStop']): 
					# closed PvL at ATR
					prices.ix[i, 'PvL']=(prices.ix[i, 'InitialStop'] - prices.ix[i, 'Open']) * prices.ix[i, 'Qty'] 
					prices.ix[i, 'IsClosed'] = True
				# if Short entry has High > InitialStop, then exit at InitialStop
				elif (prices.ix[i, 'LvS']=='S') & (prices.ix[i, 'High'] > prices.ix[i, 'InitialStop']): 
					# closed PvL at ATR
					prices.ix[i, 'PvL']=(prices.ix[i, 'InitialStop'] - prices.ix[i, 'Open']) * prices.ix[i, 'Qty'] 
					prices.ix[i, 'IsClosed'] = True
				# position remains open and PvL is updated with close price.
				else:
					prices.ix[i, 'PvL']=(prices.ix[i, 'Close'] - prices.ix[i, 'Open']) * prices.ix[i, 'Qty'] 
					prices.ix[i, 'IsClosed'] = False
			# IsOpen is False
			else: 
				prices.ix[i, 'PvL'] = 0
				prices.ix[i, 'IsClosed'] = True						
			# EQUITY CHANGE
			prices.ix[i, 'Equity'] = prices.ix[i-1, 'Equity'] + prices.ix[i, 'PvL']
		# DayOfWk 1-4
		else:
			# QTY
			prices.ix[i, 'Qty']=prices.ix[i-1, 'Qty'] # is this necessary given the ffill up above on day 0?
			# PROFIT OR LOSS
			# if previous IsClosed is True...
			if prices.ix[i-1, 'IsClosed']:
				prices.ix[i, 'PvL'] = 0
				prices.ix[i, 'IsClosed'] = True
				prices.ix[i, 'IsOpen'] = False
			# position is open
			else:
				# if Long entry has Low < InitialStop, then exit at InitialStop
				if (prices.ix[i, 'LvS']=='L') & (prices.ix[i, 'Low'] < prices.ix[i, 'InitialStop']): 
					# closed PvL at ATR
					prices.ix[i, 'PvL']=(prices.ix[i, 'InitialStop'] - prices.ix[i, 'Open']) * prices.ix[i, 'Qty'] 
					prices.ix[i, 'IsClosed'] = True
					prices.ix[i, 'IsOpen'] = False
				# if Short entry has High > InitialStop, then exit at InitialStop
				elif (prices.ix[i, 'LvS']=='S') & (prices.ix[i, 'High'] > prices.ix[i, 'InitialStop']): 
					# closed PvL at ATR
					prices.ix[i, 'PvL']=(prices.ix[i, 'InitialStop'] - prices.ix[i, 'Open']) * prices.ix[i, 'Qty'] 
					prices.ix[i, 'IsClosed'] = True
					prices.ix[i, 'IsOpen'] = False
				# position remains open
				else:
					prices.ix[i, 'PvL']=(prices.ix[i, 'Close'] - prices.ix[i, 'Open']) * prices.ix[i, 'Qty'] 
					prices.ix[i, 'IsClosed'] = False
					prices.ix[i, 'IsOpen'] = True				
			# EQUITY CHANGE
			prices.ix[i, 'Equity'] = prices.ix[i-1, 'Equity'] + prices.ix[i, 'PvL']

	# TOTAL DOLLAR RISK
	prices.loc[prices.DayOfWk==0, '$Risk'] = abs((prices['Open']-prices['InitialStop']) * prices['Qty'])
	prices['$Risk']=prices['$Risk'].ffill() 
	
	# PERCENT RISK
	prices.loc[prices.DayOfWk==0, '%Risk']=prices['$Risk']/prices['Equity']
	
	# CUMULATIVE PROFIT / LOSS
	prices['CumPL']=prices['PvL'].cumsum()
	prices.ix[0 : atr_periods+1, 'CumPL'] = 0.0	
	
	# POSITION PvL / WEEKLY
	prices.loc[(prices.DayOfWk==4) & (prices['$Risk'] != 0), 'PPvL'] = prices['CumPL'] - prices['CumPL'].shift(5)
	
	# R-MULT
	prices.loc[(prices.DayOfWk==4) & (prices['$Risk'] != 0), 'R-Mult']= prices['PPvL'] / prices['$Risk']
	
	# CUMULATIVE R
	prices['Cum-R']=prices['R-Mult'].cumsum().ffill()
	
	# LOSING STREAK
	# toss = prices['CumPL'].fillna(0)
	# TICKER
	prices['Ticker'] = ticker
	
	#===========================================================
	# Stats
	#===========================================================
	if show_stats:
		sd = ['Start Date', start_date]
		ed = ['End Date', end_date] #['End Date', prices.index.values]#prices.iloc[-1]
		tix = ['Ticker', ticker]
		ie = ['Initial Equity', initial_equity]
		rsk = ['Risk %', pct_equity_risk]
		cpl = ['Gain / Loss', prices.CumPL.iloc[-1]]
		ret = ['% Return', prices.CumPL.iloc[-1] / float(initial_equity)]
		rm = ['R-Mult', prices['Cum-R'].iloc[-1]]
		exp = ['Expectancy', prices['R-Mult'].mean()]
		stdv = ['Std. Dev', prices['R-Mult'].std()]
		# trades = prices.loc[prices['R-Mult'] !=0, 'R-Mult'].count()
		# tc = ['Trades', trades]
		sqn = ['SQN', prices['R-Mult'].mean() / prices['R-Mult'].std() * np.sqrt(prices.loc[prices['R-Mult'] !=0, 'R-Mult'].count())]
		# wins = prices.loc[prices['PPvL'] > 0, 'PPvL'].count()
		# w = ['Wins', wins]
		# wp = ['Win %', float(wins)/float(trades)]
		# losses = prices.loc[prices['PPvL'] < 0, 'PPvL'].count()
		# l = 'fix this' #['+R : -R', float(wins) / losses]
		# max_dd_r = max_drawdown(prices['Cum-R'], 'R')
		# mdr = ['Max Drawdown: R', max_dd_r]
		# max_dd_p = max_drawdown(prices['CumPL'], '%')
		# mdp = ['Max Drawdown: %', max_dd_p]
		# df_stats = pd.DataFrame([tix, sd, ed, ie, rsk, cpl, ret, rm, exp, stdv, w, tc, wp, l, mdr, mdp, sqn], columns=('Stat', 'Value'))
		# print df_stats
		# df_stats.to_csv(output_data_file_name2)
	# print prices.ix[:, (0,3,4,6,7,8,9,14,15,16)].tail()
	# print prices.loc[:, ('Open', 'Close', 'EntryPrice', 'SMA(5)', 'LvS', 'InitialStop', '$Risk', '%Risk', 'Qty', 'IsSma', 'PvL', 'Equity', 'DayOfWk')]
	prices.to_csv(output_data_file_name)
