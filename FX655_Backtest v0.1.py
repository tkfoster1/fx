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
show_stats = False
const_file_path = 'Documents/p/fin/Data/System/FX/current/'
output_data_file_name = 'C:/Users/tfoster/Desktop/Temp/fx.csv'
output_data_file_name2 = 'C:/Users/tfoster/Desktop/Temp/fx_stats.csv'

initial_equity = 26000.00
pct_equity_risk = .03
atr_periods = 14 # ATR
atr_factor = 2

start_date = '2015-11-02'
end_date = '2015-12-31'

ticker = 'USDJPY'
# instruments = str('AUDUSD GBPUSD EURUSD USDCHF USDCAD USDJPY')
instruments = str(ticker)
ticks = [instruments]
sma_period = 5
sma2_period = 7
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
	df.loc[:, 'TR1'] = abs(df['High']-df['Low']).fillna(0)
	df.loc[:, 'TR2'] = abs(df['High']-df['Close'].shift()).fillna(0)
	df.loc[:, 'TR3'] = abs(df['Low']-df['Close'].shift()).fillna(0)
	df.loc[:, 'TrueRange'] = df[['TR1','TR2','TR3']].max(axis=1)
	return df

def calc_atr (df, atr_periods):
	calc_true_range(df)
	df.loc[:, 'ATR']=pd.Series.rolling(df['TrueRange'], atr_periods).mean()
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
	# print max_daily_dd
	return max_daily_dd

	
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
	prices.loc[:, 'SMA'] = pd.Series.rolling(prices['Close'], 
					sma_period, min_periods=sma_period).mean()
	prices.loc[:, 'SMA2'] = pd.Series.rolling(prices['Close'], 
					sma2_period, min_periods=sma2_period).mean()					

	# PCT CHANGE
	prices.loc[:, 'PctChange'] = (prices.loc[(prices.DayOfWk == 4), 'Close'] 
							- prices.Open.shift(4))/prices.Open.shift(4) # % change between Monday (0) and Friday (4)
	prices.loc[prices.PctChange > 0,'LvS'] = 'L' 
	prices.loc[prices.PctChange < 0,'LvS'] = 'S'	
	prices.LvS = prices.LvS.shift(1) # Direction to take for the following week. That is, after the Friday (4)
	prices.loc[:, 'LvS'] = prices['LvS'].ffill()
	# prices.loc[(prices.DayOfWk==0) & (abs(prices['PctChange'].shift(1)) < .005), 'IsPctChange'] = True
	
	# IS SMA
	prices.loc[prices.LvS=='L', 'IsSma'] = (prices.Close > prices['SMA']) & (prices['SMA'] >= prices['SMA2'])
	prices.loc[prices.LvS=='S', 'IsSma'] = (prices.Close < prices['SMA']) & (prices['SMA'] <= prices['SMA2'])

	# ATR
	calc_atr(prices, atr_periods)
	
	# INITIAL STOP
	# make it a trailing stop based on the ATR from day 4
	prices.loc[(prices.DayOfWk==0) & (prices.LvS=='L'), 'InitialStop'] = prices.Open - (prices.ATR.shift(1) * atr_factor)
	prices.loc[(prices.DayOfWk==0) & (prices.LvS=='S'), 'InitialStop'] = prices.Open + (prices.ATR.shift(1) * atr_factor)
	
	# TRAIL STOP
	prices.loc[(prices.DayOfWk==0), 'TrailStopAmt'] = prices.ATR * atr_factor
	prices.loc[:, 'TrailStopAmt']=prices['TrailStopAmt'].ffill()
	prices.loc[(prices.DayOfWk==0), 'TrailStopPrice'] = prices['InitialStop']
	
	# QTY / EQUITY
	prices.ix[atr_periods, 'Equity'] = initial_equity 
	prices.loc[:, 'Equity']=prices['Equity'].ffill()
	prices.loc[:, 'Qty']=0 # intialized Qty

	# IS OPEN
	# if the day 4 price is above SMA, then the day 0 IsOpen = True
	prices.loc[(prices.DayOfWk==0) & (prices.LvS=='L'), 'IsOpen'] = (prices.Close.shift(1) >  prices.SMA.shift(1)) & prices.IsSma == True#& prices.IsPctChange
	prices.loc[(prices.DayOfWk==0) & (prices.LvS=='S'), 'IsOpen'] = (prices.Close.shift(1) <  prices.SMA.shift(1)) & prices.IsSma == True#& prices.IsPctChange	
	
	# Loop for shares and total updates -- SLOW
	# ALL the magic happens here. 
	for i in range(atr_periods + 1, len(prices)):
		if prices.ix[i, 'DayOfWk'] == 0:
			# QTY 
			prices.ix[i, 'Qty'] = (prices.ix[i-1, 'Equity'] * pct_equity_risk / (prices.ix[i, 'Open'] - prices.ix[i, 'InitialStop'])).round(-4) 
			prices.loc[:, 'Qty']=prices['Qty'].ffill()
			prices.loc[:, 'Qty']=prices['Qty'].fillna(0)
			# PROFIT OR LOSS
			if prices.ix[i, 'IsOpen']:
				# if Long entry has Low < TrailStopPrice, then exit at TrailStopPrice
				if (prices.ix[i, 'LvS']=='L') & (prices.ix[i, 'Low'] < prices.ix[i, 'TrailStopPrice']): 
					# closed PvL 
					prices.ix[i, 'PvL']=(prices.ix[i, 'TrailStopPrice'] - prices.ix[i, 'Open']) * prices.ix[i, 'Qty'] 
					prices.ix[i, 'IsClosed'] = True
				# if Short entry has High > TrailStopPrice, then exit at TrailStopPrice
				elif (prices.ix[i, 'LvS']=='S') & (prices.ix[i, 'High'] > prices.ix[i, 'TrailStopPrice']): 
					# closed PvL at ATR
					prices.ix[i, 'PvL']=(prices.ix[i, 'TrailStopPrice'] - prices.ix[i, 'Open']) * prices.ix[i, 'Qty'] 
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
		# if DayOfWk == 1-4
		else:
			# QTY
			prices.ix[i, 'Qty']=prices.ix[i-1, 'Qty'] # is this necessary given the ffill up above on day 0?
			''' adjust the trailing stop. This is an optimistic assumption 
			that the best price of the day happens before the worst. So the 
			trail stop gets adjusted to the best possible exit. Would need 
			to dig in to hourly (or less) data to improve the accuracy. '''
			# TRAIL STOP
			if prices.ix[i, 'LvS']=='L':
				# for the long case, trail stop only moves up, not down
				if (prices.ix[i, 'High'] > prices.ix[i-1, 'High']):
					if (prices.ix[i, 'High'] - prices.ix[i, 'TrailStopAmt'] < prices.ix[i-1, 'TrailStopPrice']):
						prices.ix[i, 'TrailStopPrice'] = prices.ix[i-1, 'TrailStopPrice']
					else:
						prices.ix[i, 'TrailStopPrice'] = prices.ix[i, 'High'] - prices.ix[i, 'TrailStopAmt']
				else:
					prices.ix[i, 'TrailStopPrice'] = prices.ix[i-1, 'TrailStopPrice']
			elif prices.ix[i, 'LvS']=='S':
				# for short, trail stop only moves down, not up
				if prices.ix[i, 'Low'] < prices.ix[i-1, 'Low']:
					if (prices.ix[i, 'Low'] + prices.ix[i, 'TrailStopAmt'] > prices.ix[i-1, 'TrailStopAmt']):
						prices.ix[i, 'TrailStopPrice'] = prices.ix[i-1, 'TrailStopPrice']
					else: 
						prices.ix[i, 'TrailStopPrice'] = prices.ix[i, 'Low'] + prices.ix[i, 'TrailStopAmt']
				else:
					prices.ix[i, 'TrailStopPrice'] = prices.ix[i-1, 'TrailStopPrice']
			# # not a trailing stop but a static, initial stop. Comment out TRAIL STOP if used.
			# prices.ix[i, 'TrailStopPrice'] = prices.ix[i-1, 'TrailStopPrice']
			# PROFIT OR LOSS
			# if previous IsClosed is True...
			if prices.ix[i-1, 'IsClosed']:
				prices.ix[i, 'PvL'] = 0
				prices.ix[i, 'IsClosed'] = True
				prices.ix[i, 'IsOpen'] = False
			# if previous IsClosed is False, then position is open
			else:
				# if Long entry has Low < TrailStopPrice, then exit at TrailStopPrice
				if (prices.ix[i, 'LvS']=='L') & (prices.ix[i, 'Low'] < prices.ix[i, 'TrailStopPrice']): 
					# closed PvL at ATR
					prices.ix[i, 'PvL']=(prices.ix[i, 'TrailStopPrice'] - prices.ix[i, 'Open']) * prices.ix[i, 'Qty'] 
					prices.ix[i, 'IsClosed'] = True
					prices.ix[i, 'IsOpen'] = False
				# if Short entry has High > TrailStopPrice, then exit at TrailStopPrice
				elif (prices.ix[i, 'LvS']=='S') & (prices.ix[i, 'High'] > prices.ix[i, 'TrailStopPrice']): 
					# closed PvL at ATR
					prices.ix[i, 'PvL']=(prices.ix[i, 'TrailStopPrice'] - prices.ix[i, 'Open']) * prices.ix[i, 'Qty'] 
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
	prices.loc[:, '$Risk']=prices['$Risk'].ffill() 
	
	# PERCENT RISK
	prices.loc[prices.DayOfWk==0, '%Risk']=prices['$Risk']/prices['Equity']
	
	# CUMULATIVE PROFIT / LOSS
	prices.loc[:, 'CumPL']=prices['PvL'].cumsum()
	prices.ix[0 : atr_periods+1, 'CumPL'] = 0.0	
	
	# POSITION PvL / WEEKLY
	prices.loc[(prices.DayOfWk==4) & (prices['$Risk'] != 0), 'PPvL'] = prices['CumPL'] - prices['CumPL'].shift(5)
	
	# R-MULT
	prices.loc[(prices.DayOfWk==4) & (prices['$Risk'] != 0), 'R-Mult']= prices['PPvL'] / prices['$Risk']
	
	# CUMULATIVE R
	prices.loc[:, 'Cum-R']=prices['R-Mult'].cumsum().ffill()
	
	# LOSING STREAK
	# toss = prices['CumPL'].fillna(0)
	# TICKER
	prices.loc[:, 'Ticker'] = ticker
	
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
		trades = prices.loc[prices['R-Mult'] !=0, 'R-Mult'].count()
		tc = ['Trades', trades]
		sqn = ['SQN', prices['R-Mult'].mean() / prices['R-Mult'].std() * np.sqrt(prices.loc[prices['R-Mult'] !=0, 'R-Mult'].count())]
		wins = prices.loc[prices['PPvL'] > 0, 'PPvL'].count()
		w = ['Wins', wins]
		wp = ['Win %', float(wins)/float(trades)]
		losses = prices.loc[prices['PPvL'] < 0, 'PPvL'].count()
		l = ['+R : -R', float(wins) / losses]
		max_dd_r = max_drawdown(prices['Cum-R'], 'R')
		mdr = ['Max Drawdown: R', max_dd_r]
		max_dd_p = max_drawdown(prices['CumPL'], '%')
		mdp = ['Max Drawdown: %', max_dd_p]
		df_stats = pd.DataFrame([tix, sd, ed, ie, rsk, cpl, ret, rm, exp, stdv, w, tc, wp, l, mdr, mdp, sqn], columns=('Stat', 'Value'))
		print df_stats
		df_stats.to_csv(output_data_file_name2)
	# print prices.ix[:, (0,3,4,6,7,8,9,14,15,16)].tail()
	# print prices.loc[:, ('Open', 'Close', 'EntryPrice', 'SMA(5)', 'LvS', 'InitialStop', '$Risk', '%Risk', 'Qty', 'IsSma', 'PvL', 'Equity', 'DayOfWk')]
	prices.to_csv(output_data_file_name)
