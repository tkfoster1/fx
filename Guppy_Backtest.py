# ==============================================================
# %run "/Users/tfoster/Documents/p/fin/Data/System/Guppy/Guppy_Backtest.py"
# TKF
# Created: 12.28.2016
# ==============================================================
import get_Guppy_Backtest
import time

# ==============================================================================
# ASSUMPTIONS
# ==============================================================================
show_stats = 2

ticker = 'EURUSD'

source_data_file_path = 'Documents/p/fin/Data/System/FX/current/'
destination_file_path = 'C:/Users/tfoster/Documents/p/fin/Data/System/Guppy/'

output_stats_file = destination_file_path+ticker+'fx_stat_guppy_allyears.csv'
output_data_file_total = destination_file_path+ticker+'fx_prices_guppy_allyears.csv'

initial_equity = 26000.00
pct_equity_risk = .015
sma1_period = 20
sma2_period = 60
atr_periods = 20 
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
		# get_Guppy_Backtest.main(show_stats, source_data_file_path, destination_file_path, initial_equity, pct_equity_risk, sma1_period, sma2_period, atr_periods, atr_factor, start_date, end_date, ticker, output_stats_file, output_data_file_total)
	
# elapsed_time = time.time() - start_time
# print elapsed_time / 60.0, 'minutes'

###	
start_date = '2001-1-1'
end_date = '2011-4-30'

# ATR FACTOR
for i in xrange(2, 3, 1):
	start_time = time.time()
	print 'atr_factor: '+ str(i)
	
	atr_factor = i
	get_Guppy_Backtest.main(show_stats, source_data_file_path, destination_file_path, initial_equity, pct_equity_risk, sma1_period, sma2_period, atr_periods, atr_factor, start_date, end_date, ticker, output_stats_file, output_data_file_total)
	
	elapsed_time = time.time() - start_time
	print elapsed_time / 60.0, 'minutes'

# # # get_Guppy_Backtest.main(show_stats, source_data_file_path, destination_file_path, initial_equity, pct_equity_risk, sma1_period, sma2_period, atr_periods, atr_factor, start_date, end_date, ticker, output_stats_file, output_data_file_total)	
