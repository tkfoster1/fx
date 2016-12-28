# ==============================================================
# %run "/Users/tfoster/Documents/p/fin/Data/System/MA/MA_xover_Backtest.py"
# TKF
# Created: 12.01.2016
# ==============================================================
import get_MA_xover_Backtest
import time

# ==============================================================================
# ASSUMPTIONS
# ==============================================================================
show_stats = 2

ticker = 'EURUSD'

source_data_file_path = 'Documents/p/fin/Data/System/FX/current/'
destination_file_path = 'C:/Users/tfoster/Documents/p/fin/Data/System/MA/'
# output_stats_file = destination_file_path+ticker#+'fx_ma_cross_stat_total.csv'
# output_stats_file = destination_file_path+ticker+'fx_ma_cross_stat_annual_20.csv'
output_stats_file = destination_file_path+ticker+'fx_ma_cross_stat_allyears_20.csv'
# output_data_file_total = destination_file_path+ticker+'fx_ma_cross_prices_total.csv'
# output_data_file_total = destination_file_path+ticker+'fx_ma_cross_prices_annual_20.csv'
output_data_file_total = destination_file_path+ticker+'fx_ma_cross_prices_allyears_20.csv'

initial_equity = 26000.00
pct_equity_risk = .015
sma1_period = 20
sma2_period = 50
atr_periods = 15 
# atr_factor = 2

# # MULTIPLE INDIVIDUAL YEARS
# start_time = time.time()
# for i in range (1, 17): # year loop
	
	# print 'i: '+ str(i)
	
	# if len(str(i))==1:
		# yr = '0'+str(i)
	# else: 
		# yr = str(i)
		
	# start_date = '20'+yr+'-1-1'
	# end_date = '20'+yr+'-12-31'
	
	# for j in xrange(1,5,1): # atr_factor loop
		# print 'atr_factor:' + str(j)
		# atr_factor = j
		# get_MA_xover_Backtest.main(show_stats, source_data_file_path, destination_file_path, initial_equity, pct_equity_risk, sma1_period, sma2_period, atr_periods, atr_factor, start_date, end_date, ticker, output_stats_file, output_data_file_total)
	
# elapsed_time = time.time() - start_time
# print elapsed_time / 60.0, 'minutes'

###	
start_date = '2001-1-1'
end_date = '2016-12-31'

# ATR FACTOR
for i in xrange(1, 5, 1):
	start_time = time.time()
	print 'atr_factor: '+ str(i)
	
	atr_factor = i
	get_MA_xover_Backtest.main(show_stats, source_data_file_path, destination_file_path, initial_equity, pct_equity_risk, sma1_period, sma2_period, atr_periods, atr_factor, start_date, end_date, ticker, output_stats_file, output_data_file_total)
	
	elapsed_time = time.time() - start_time
	print elapsed_time / 60.0, 'minutes'

get_MA_xover_Backtest.main(show_stats, source_data_file_path, destination_file_path, initial_equity, pct_equity_risk, sma1_period, sma2_period, atr_periods, atr_factor, start_date, end_date, ticker, output_stats_file, output_data_file_total)	
