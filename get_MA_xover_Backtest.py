# %run "/Users/tfoster/Documents/p/fin/Data/System/MA/get_MA_xover_Backtest.py"
# TKF
# Created: 12.01.2016

import pandas as pd
from pandas import *
import numpy as np
import sys
import os.path

# ==============================================================================
# FUNCTIONS
# ==============================================================================
def get_price_data(source_data_file_path, ticker):
	'''Load price data'''
	fields = ['Date', 'Open', 'High', 'Low', 'Close']
	data = pd.read_csv(source_data_file_path+ticker+'.txt', index_col='Date', 
						parse_dates=True, 
						usecols = fields
						)
	return data

def position_id(df, starting_time_period):
	df.ix[starting_time_period-1:, 'PosId'] = 0
	df[starting_time_period-1:].ix[(df['OpenClose']==1) & (df['OpenClose'].shift(1)!= 1), 'PosId'] = 1
	df.ix[:, 'PosId'] = df['PosId'].cumsum()+1
	df.ix[:, 'PosId'] = df['PosId'].ffill()
	return
	
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
	
def simple_moving_average(price_series, duration):
	'''Calculate a simple moving average'''
	simple_moving_average = pd.Series.rolling(price_series, duration, 
			min_periods=duration).mean()
	return simple_moving_average

def calc_sma_price(df, sma1, sma2):
	# sma3 = sma3_period #temp?
	df.loc[:, 'SMA1'] = simple_moving_average(df['Close'], sma1)
	df.loc[:, 'SMA2'] = simple_moving_average(df['Close'], sma2)
	# prices.loc[:, 'SMA3'] = simple_moving_average(prices['Close'], sma3)	
	return 
	
def is_sma(df, starting_time_period):
	'''DIRECTION: short SMA is above the long SMA == LONG
	-1 == Short and 1 == Long'''
	df.ix[starting_time_period-1:, 'IsSma']=0
	df.loc[(df['SMA1'] > df['SMA2']), 'IsSma'] = 1 	# LONG
	df.loc[(df['SMA1'] < df['SMA2']), 'IsSma'] = -1	# SHORT	
	# df.loc[(df['SMA1'] > df['SMA2']) & (df['SMA2'] > df['SMA3']), 'IsSma'] = 1 	# LONG
	# df.loc[(df['SMA1'] < df['SMA2']) & (df['SMA2'] < df['SMA3']), 'IsSma'] = -1	# SHORT		
	return	

def entry_long_or_short(df, i):
	'''Close price > short SMA and SMA is LONG.'''
	if(df.ix[i, 'Close'] > df.ix[i, 'SMA1']) & (df.ix[i, 'IsSma']==1):
		result = 1
	elif((df.ix[i, 'Close'] < df.ix[i, 'SMA1']) & (df.ix[i, 'IsSma']==-1)):
		result = -1
	else:
		result = 0
	return result
	
def is_price(df, i):
	'''CRITERIA: close is > sma = LONG
	Answers the question, "May I enter?"'''
	df.ix[i, 'IsPrice'] = entry_long_or_short(df, i)
	return
	
def calc_atr_stop_price(df):
	df.loc[(df['IsSma']==1), 'AtrStopPriceLong'] = df['High'] - (df['ATR'] * atr_factor)
	df.loc[(df['IsSma']==-1), 'AtrStopPriceShort'] = df['Low'] + (df['ATR'] * atr_factor)	
	return df
	
def trail_stop_adjust(df, i, atr_factor):
	#LONG ExitStop adjustment
	if (df.ix[i, 'IsSma'] == 1): 
		if(df.ix[i-1, 'IsSma']==1): # Direction continues
			if (df.ix[i, 'High'] - (df.ix[i, 'ATR'] * atr_factor)) > df.ix[i-1, 'ExitStop']: # Does the High price of the day move the trailing exit stop?
				df.ix[i, 'ExitStop'] = df.ix[i, 'High'] - (df.ix[i, 'ATR'] * atr_factor)				
			else: # no change, use previous
				df.ix[i, 'ExitStop'] = df.ix[i-1, 'ExitStop']
		else: # SMA reversal
			df.ix[i, 'ExitStop'] = df.ix[i, 'High'] - (df.ix[i, 'ATR'] * atr_factor)
	# SHORT ExitStop adjustment
	elif (df.ix[i, 'IsSma'] == -1): 
		if(df.ix[i-1, 'IsSma']==-1): # Direction continues
			if (df.ix[i, 'Low'] + (df.ix[i, 'ATR'] * atr_factor)) < df.ix[i-1, 'ExitStop']:
				df.ix[i, 'ExitStop'] = df.ix[i, 'Low'] + (df.ix[i, 'ATR'] * atr_factor)
			else: # no change, use previous
				df.ix[i, 'ExitStop'] = df.ix[i-1, 'ExitStop'] 
		else: # SMA reversal
			df.ix[i, 'ExitStop'] = df.ix[i, 'Low'] + (df.ix[i, 'ATR'] * atr_factor)			
	return			

def open_position(df, i):
	'''If the previous period OpenPos is closed, look for entry 
	criteria satisfaction.'''
	if(df.ix[i, 'IsSma']==df.ix[i, 'IsPrice']) & (df.ix[i-1, 'IsPrice']!=0) & (df.ix[i-2, 'IsPrice']==0):
		df.ix[i, 'OpenPos'] = 1
	return

# ==============================================================
# Exit criteria
# ==============================================================
def is_sma_reversal(df, i):
	if(df.ix[i, 'IsSma']!=df.ix[i-1, 'IsSma']):	
		result = 1
	else: 
		result = 0
	return result

def is_stop_violation(df, i, atr_factor):
	trail_stop_adjust(df, i, atr_factor)
	
	if(df.ix[i, 'IsSma']==1): # LONG
		if(df.ix[i, 'Low'] < df.ix[i, 'ExitStop']):
			result = 1
		else:
			result = 0
	elif(df.ix[i, 'IsSma']==-1): # SHORT
		if(df.ix[i, 'High'] > df.ix[i, 'ExitStop']):
			result = 1
		else: 
			result = 0
	else:
		result = 0
	return result
	
def close_position(df, i, atr_factor):
	'''Closing criteria are either SMA reversal or exit stop violation.'''
	df.ix[i, 'IsSmaRev']=is_sma_reversal(df, i)
	df.ix[i, 'IsStopViol']=is_stop_violation(df, i, atr_factor)	
	if(df.ix[i, 'IsSmaRev']==1) | (df.ix[i, 'IsStopViol']==1):
		df.ix[i,'ClosePos']=-1
	return	

# ==============================================================================
# STATS FUNCTIONs
# ==============================================================================
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
	return max_daily_dd
	
def get_date(df, StartDateOrEndDate):
    '''Return either a start or end date'''
    df1 = df.copy()
    df1 = df1.reset_index()
    if StartDateOrEndDate == 'end':
        get_date = df1.loc[len(df1.index)-1, 'Date']
    else: 
        get_date = df1.loc[0, 'Date']	
    return get_date

def sqn_calc(RMult, result):
	'''Calculate the SQN and return list values for display'''
	sqn_calc = RMult.mean() / RMult.std() * np.sqrt(result.loc[RMult !=0, 'R-Mult'].count())	
	# sqn_calc = ['SQN', RMult.mean() / RMult.std() * np.sqrt(result.loc[RMult !=0, 'R-Mult'].count())]
	return sqn_calc

def div_by_zero(num, denom):
	'''Handle divide by zero problem. Return 0.'''
	if denom==0:
		r = 0
	else:
		r = num / denom
	return r
	
# ==============================================================================
# MAIN
# ==============================================================================
def main(show_stats, source_data_file_path, destination_file_path, initial_equity, pct_equity_risk, sma1_period, sma2_period, atr_periods, atr_factor, start_date, end_date, ticker, output_stats_file, output_data_file_total):

	instruments = str(ticker)
	ticks = [instruments]
	lstInstruments = []

	starting_time_period = max(sma2_period, atr_periods)

	prices = get_price_data(source_data_file_path, ticker) # load price data
	prices = prices[start_date:end_date] # get relevant subset period of data
	
	calc_sma_price(prices, sma1_period, sma2_period) # SMA calcs
	calc_atr(prices, atr_periods) # ATR calc
	is_sma(prices, starting_time_period) # Long or short

	prices.loc[starting_time_period-1:, 'IsPrice']=0	
	prices.loc[starting_time_period-1:, 'OpenPos']=0
	prices.loc[starting_time_period-1:, 'ExitStop']=0
	prices.ix[starting_time_period-1, 'ExitStop']=prices.ix[starting_time_period-1, 'Close'] - (prices.ix[starting_time_period-1, 'ATR']*atr_factor)*prices.ix[starting_time_period-1, 'IsSma']
	prices.loc[starting_time_period-1:, 'IsSmaRev']=0
	prices.loc[starting_time_period-1:, 'IsStopViol']=0	
	prices.loc[starting_time_period-1:, 'ClosePos']=0
	prices.ix[starting_time_period-1, 'OpenClose']=0
	
	#Open and close positions
	for i in range(starting_time_period, len(prices)):
		close_position(prices, i, atr_factor)	
		is_price(prices, i) 
		open_position(prices, i)
	
	prices.loc[prices['OpenPos']!=0, 'OpenClose'] = prices.loc[:, 'OpenPos']
	prices.loc[prices['ClosePos']!=0, 'OpenClose'] = prices.loc[:, 'ClosePos']
	prices.ix[:, 'OpenClose'] = prices['OpenClose'].ffill()

	prices.ix[starting_time_period-1:, 'Equity'] = initial_equity 
	prices.ix[starting_time_period-1, 'Qty'] = 0
	prices.ix[starting_time_period-1, '$Risk'] = 0
	prices.ix[starting_time_period-1, 'PvL'] = 0	

	# Calculate position sizes and PvL
	for i in range(starting_time_period, len(prices)):
		if (prices.ix[i, 'OpenClose'] == 1) & (prices.ix[i-1, 'OpenClose']!=1): # Open a new position
			prices.ix[i, 'Qty'] = (prices.ix[i-1, 'Equity'] * pct_equity_risk / (prices.ix[i, 'Open'] - prices.ix[i-1, 'ExitStop'])).round(-4) # QTY 
			prices.ix[i, '$Risk'] = abs((prices.ix[i, 'Open'] - prices.ix[i-1, 'ExitStop']) * prices.ix[i, 'Qty']) # TOTAL DOLLAR RISK			
			prices.ix[i, 'PvL'] = (prices.ix[i, 'Close'] - prices.ix[i, 'Open']) * prices.ix[i, 'Qty'] # PROFIT OR LOSS
			prices.ix[i, 'Equity'] = prices.ix[i-1, 'Equity'] + prices.ix[i, 'PvL'] # EQUITY
		elif(prices.ix[i, 'OpenClose']==1) | ((prices.ix[i, 'OpenClose']==-1)&(prices.ix[i-1, 'OpenClose']==1) & (prices.ix[i, 'IsStopViol']==0)): # Continuation of open position
			prices.ix[i, 'Qty'] = prices.ix[i-1, 'Qty'] 
			prices.ix[i, '$Risk'] = prices.ix[i-1, '$Risk'] 
			prices.ix[i, 'PvL'] = (prices.ix[i, 'Close'] - prices.ix[i, 'Open']) * prices.ix[i, 'Qty'] 
			prices.ix[i, 'Equity'] = prices.ix[i-1, 'Equity'] + prices.ix[i, 'PvL']
		elif(prices.ix[i, 'OpenClose']==-1) & (prices.ix[i-1, 'OpenClose']==1) & (prices.ix[i, 'IsStopViol']==1): # Newly closed position : stop violation
			prices.ix[i, 'Qty']=prices.ix[i-1, 'Qty']
			prices.ix[i, '$Risk'] = prices.ix[i-1, '$Risk'] 
			prices.ix[i, 'PvL'] = (prices.ix[i, 'ExitStop']-prices.ix[i, 'Open'])*prices.ix[i, 'Qty'] 
			prices.ix[i, 'Equity'] = prices.ix[i-1, 'Equity'] + prices.ix[i, 'PvL']	
		elif(prices.ix[i-1, 'OpenClose']==-1) & (prices.ix[i-2, 'OpenClose']==1) & (prices.ix[i, 'IsSmaRev']==1) & (prices.ix[i-1, 'IsStopViol']!=1): # Newly closed position : MA reversal
			prices.ix[i, 'Qty']=prices.ix[i-1, 'Qty']
			prices.ix[i, '$Risk'] = prices.ix[i-1, '$Risk'] 
			prices.ix[i, 'PvL'] = (prices.ix[i, 'Open']-prices.ix[i-1, 'Open'])*prices.ix[i, 'Qty'] 
			prices.ix[i, 'Equity'] = prices.ix[i-1, 'Equity'] + prices.ix[i, 'PvL']
		else: # Continuation of closed postion
			prices.ix[i, 'PvL'] = 0
			prices.ix[i, 'Equity'] = prices.ix[i-1, 'Equity']

	# PERCENT RISK
	prices.loc[(prices['OpenClose'] == 1) & (prices['OpenClose'].shift(1)!=1), '%Risk'] = prices['$Risk'] / prices['Equity']
	
	# CUMULATIVE PROFIT / LOSS
	prices.loc[:, 'CumPL']=prices['PvL'].cumsum()
	
	position_id(prices, starting_time_period)

	# POSITION PvL 
	dfPPvL = pd.DataFrame(prices.groupby('PosId', as_index=False)['PvL'].sum())
	
	# R-MULT
	dfPrices2 = prices.ix[(prices['$Risk']!=0), :]
	dfRMult = dfPrices2.groupby(['PosId', '$Risk']).size().reset_index()
	result = pd.merge(dfPPvL, dfRMult, on='PosId')
	result.loc[:, 'R-Mult'] = result['PvL'] / result['$Risk']
	
	# CUMULATIVE R
	result.loc[:, 'Cum-R']=result['R-Mult'].cumsum().ffill()
	result.loc[:, 'CumPL']=result['PvL'].cumsum()

	# Results to CSV
	if (os.path.isfile(destination_file_path+ticker+'_result.csv')==True):
		with open(destination_file_path+ticker+'_result.csv', 'a') as f:
			result.to_csv(f, header=False)
	else:
		result.to_csv(destination_file_path+ticker+'_result.csv')	
	
	# Prices to CSV
	if (os.path.isfile(output_data_file_total)==True):
		with open(output_data_file_total, 'a') as f:
			prices.to_csv(f, header=False)
	else:
		prices.to_csv(output_data_file_total)			
	
	#===========================================================
	# Stats
	#===========================================================
	if show_stats==1:
		sd = ['Start Date', get_date(prices, 'start')]
		ed = ['End Date', get_date(prices, 'end')]
		tix = ['Ticker', ticker]
		ie = ['Initial Equity', initial_equity]
		rsk = ['Risk %', pct_equity_risk]
		cpl = ['Gain / Loss', result.CumPL.iloc[-1]]
		ret = ['% Return', result.CumPL.iloc[-1] / float(initial_equity)]
		rm = ['R-Mult', result['Cum-R'].iloc[-1]]
		exp = ['Expectancy', result['R-Mult'].mean()]
		stdv = ['Std. Dev', result['R-Mult'].std()]
		trades = result.loc[result['R-Mult'] !=0, 'R-Mult'].count()
		tc = ['Trades', trades]
		sqn = sqn_calc(result['R-Mult'], result)
		wins = result.loc[result['PvL'] > 0, 'PvL'].count()
		w = ['Wins', wins]
		wp = ['Win %', float(wins)/float(trades)]
		losses = result.loc[result['PvL'] < 0, 'PvL'].count()
		l = ['+R : -R', float(wins) / losses]
		max_dd_r = max_drawdown(result['Cum-R'], 'R')
		mdr = ['Max Drawdown: R', max_dd_r]
		max_dd_p = max_drawdown(result['CumPL'], '%')
		mdp = ['Max Drawdown: %', max_dd_p]
		sma1 = ['sma1', sma1_period]
		sma2 = ['sma2', sma2_period]
		atrp = ['atr_per', atr_periods]
		atrf = ['atr_factor', atr_factor]
		df_stats = pd.DataFrame([tix, sd, ed, ie, rsk, cpl, ret, rm, exp, stdv, w, tc, wp, l, mdr, mdp, sma1, sma2, atrp, atrf, sqn], columns=('Stat', 'Value'))
		# print df_stats
		if (os.path.isfile(output_stats_file)==True):
			with open(output_stats_file, 'a') as f:
				df_stats.to_csv(f, header=False)
		else:
			df_stats.to_csv(output_stats_file)
	
	if show_stats==2:
		t = ticker
		ed = get_date(prices, 'end')
		ret = result.CumPL.iloc[-1] / float(initial_equity)
		sqn = result['R-Mult'].mean() / result['R-Mult'].std() * np.sqrt(result.loc[result['R-Mult'] !=0, 'R-Mult'].count())
		trades = result.loc[result['R-Mult'] !=0, 'R-Mult'].count()
		wins = result.loc[result['PvL'] > 0, 'PvL'].count()
		losses = result.loc[result['PvL'] < 0, 'PvL'].count()		
		wtl = div_by_zero(float(wins), losses)
		wp = div_by_zero(float(wins), float(trades))
		d = {'Ticker' : [t], 'EndDate' : [ed], 'SMA1' : [sma1_period], 'ATR Factor' : [atr_factor], 'Return%' : [ret], 'SQN' : [sqn], 'Trades' : [trades], 'Win %' : [wp], '+R : -R' : [wtl]}
		sv2 = pd.DataFrame(data=d)
		if (os.path.isfile(output_stats_file)==True):
			with open(output_stats_file, 'a') as f:
				sv2.to_csv(f, header=False)
		else:
			sv2.to_csv(output_stats_file)		
	
if __name__ == "__main__":
	sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8], sys.argv[9], sys.argv[10], sys.argv[11], sys.argv[12], sys.argv[13], sys.argv[14]))
