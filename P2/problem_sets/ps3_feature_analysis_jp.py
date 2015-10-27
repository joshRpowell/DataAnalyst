import numpy as np
import scipy
import scipy.stats
import pandas
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from datetime import datetime

file_path = '/Users/jpowell/Dropbox/Udacity/DataAnalyst/P2/data/turnstile_data_master_with_weather.csv'

def mann_whitney_plus_means(turnstile_weather):

    dataframe = pandas.read_csv(turnstile_weather)

    dataframe['weekday'] = pandas.to_datetime(dataframe['DATEn']).apply(lambda x: x.weekday())
    dataframe['bizday'] = np.where(dataframe['weekday'] == (0 or 6),0,1)
    dataframe['bizday'] = np.where(pandas.to_datetime(dataframe['DATEn']) == datetime(2011, 5, 30), 0, dataframe['bizday'])

    with_rain = dataframe['ENTRIESn_hourly'][dataframe['bizday'] == 1]
    without_rain = dataframe['ENTRIESn_hourly'][dataframe['bizday'] == 0]

    with_rain_count = np.count_nonzero(with_rain)
    with_rain_mean = np.mean(with_rain)
    # with_rain_mean = np.median(with_rain)
    without_rain_count = np.count_nonzero(without_rain)
    without_rain_mean = np.mean(without_rain)
    # without_rain_mean = np.median(without_rain)

    mw_test = scipy.stats.mannwhitneyu(with_rain, without_rain)

    U = mw_test[0]
    p = mw_test[1]

    print with_rain_count
    print without_rain_count
    print with_rain_mean, without_rain_mean, U, p # leave this line for the grader

mann_whitney_plus_means(file_path)
