import numpy as np
import pandas
import matplotlib.pyplot as plt

def entries_histogram(turnstile_weather):
    '''
    Before we perform any analysis, it might be useful to take a
    look at the data we're hoping to analyze. More specifically, let's
    examine the hourly entries in our NYC subway data and determine what
    distribution the data follows. This data is stored in a dataframe
    called turnstile_weather under the ['ENTRIESn_hourly'] column.

    Let's plot two histograms on the same axes to show hourly
    entries when raining vs. when not raining. Here's an example on how
    to plot histograms with pandas and matplotlib:
    turnstile_weather['column_to_graph'].hist()

    Your histogram may look similar to bar graph in the instructor notes below.

    You can read a bit about using matplotlib and pandas to plot histograms here:
    http://pandas.pydata.org/pandas-docs/stable/visualization.html#histograms

    You can see the information contained within the turnstile weather data here:
    https://www.dropbox.com/s/meyki2wl9xfa7yk/turnstile_data_master_with_weather.csv
    '''

    df = pandas.read_csv(turnstile_weather)

    plt.figure()
    plt.xlabel('ENTRIESn_hourly')
    plt.ylabel('frequency')
    plt.title('Histogram of ENTRIESn_hourly')

    df['ENTRIESn_hourly'][df['rain'] == 0].hist(bins=100, range=[0, 6000], label='no rain')
    # ^your code here to plot a historgram for hourly entries when it is not raining
    df['ENTRIESn_hourly'][df['rain'] == 1].hist(bins=100, range=[0, 6000], label='rain')
    # ^your code here to plot a historgram for hourly entries when it is raining
    plt.legend(loc='upper right')
    x1,x2,y1,y2 = plt.axis()
    print x1,x2,y1,y2
    plt.axis((x1,x2,y1,30000))

    plt.show()

file_path = '/Users/jpowell/Dropbox/Udacity/DataAnalyst/P2/data/turnstile_data_master_with_weather.csv'

entries_histogram(file_path)

import numpy as np
import scipy
import scipy.stats
import pandas

def mann_whitney_plus_means(turnstile_weather):
    '''
    This function will consume the turnstile_weather dataframe containing
    our final turnstile weather data.

    You will want to take the means and run the Mann Whitney U-test on the
    ENTRIESn_hourly column in the turnstile_weather dataframe.

    This function should return:
        1) the mean of entries with rain
        2) the mean of entries without rain
        3) the Mann-Whitney U-statistic and p-value comparing the number of entries
           with rain and the number of entries without rain

    You should feel free to use scipy's Mann-Whitney implementation, and you
    might also find it useful to use numpy's mean function.

    Here are the functions' documentation:
    http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html
    http://docs.scipy.org/doc/numpy/reference/generated/numpy.mean.html

    You can look at the final turnstile weather data at the link below:
    https://www.dropbox.com/s/meyki2wl9xfa7yk/turnstile_data_master_with_weather.csv
    '''
    df = pandas.read_csv(turnstile_weather)
    with_rain = df['ENTRIESn_hourly'][df['rain'] == 1]
    without_rain = df['ENTRIESn_hourly'][df['rain'] == 0]

    with_rain_count = np.count_nonzero(with_rain)
    with_rain_mean = np.mean(with_rain)
    without_rain_count = np.count_nonzero(without_rain)
    without_rain_mean = np.mean(without_rain)

    mw_test = scipy.stats.mannwhitneyu(with_rain, without_rain)

    U = mw_test[0]
    p = mw_test[1]

    print with_rain_count
    print without_rain_count
    print with_rain_mean, without_rain_mean, U, p # leave this line for the grader

# mann_whitney_plus_means(file_path)

import numpy as np
import pandas
import statsmodels.api as sm
from datetime import datetime

"""
In this question, you need to:
1) implement the linear_regression() procedure
2) Select features (in the predictions procedure) and make predictions.

"""

def linear_regression(features, values):
    """
    Perform linear regression given a data set with an arbitrary number of features.

    This can be the same code as in the lesson #3 exercise.
    """
    features = sm.add_constant(features)

    model = sm.OLS(values, features)
    results = model.fit()
    intercept = results.params[0]
    params = results.params[1:]

    print intercept, params
    return intercept, params

def predictions(weather_turnstile):
    '''
    The NYC turnstile data is stored in a pandas dataframe called weather_turnstile.
    Using the information stored in the dataframe, let's predict the ridership of
    the NYC subway using linear regression with gradient descent.

    You can download the complete turnstile weather dataframe here:
    https://www.dropbox.com/s/meyki2wl9xfa7yk/turnstile_data_master_with_weather.csv

    Your prediction should have a R^2 value of 0.40 or better.
    You need to experiment using various input features contained in the dataframe.
    We recommend that you don't use the EXITSn_hourly feature as an input to the
    linear model because we cannot use it as a predictor: we cannot use exits
    counts as a way to predict entry counts.

    Note: Due to the memory and CPU limitation of our Amazon EC2 instance, we will
    give you a random subet (~10%) of the data contained in
    turnstile_data_master_with_weather.csv. You are encouraged to experiment with
    this exercise on your own computer, locally. If you do, you may want to complete Exercise
    8 using gradient descent, or limit your number of features to 10 or so, since ordinary
    least squares can be very slow for a large number of features.

    If you receive a "server has encountered an error" message, that means you are
    hitting the 30-second limit that's placed on running your program. Try using a
    smaller number of features.
    '''
    ################################ MODIFY THIS SECTION #####################################
    # Select features. You should modify this section to try different features!             #
    # We've selected rain, precipi, Hour, meantempi, and UNIT (as a dummy) to start you off. #
    # See this page for more info about dummy variables:                                     #
    # http://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html          #
    ##########################################################################################
    dataframe = pandas.read_csv(weather_turnstile)

    dataframe['weekday'] = pandas.to_datetime(dataframe['DATEn']).apply(lambda x: x.weekday())
    dataframe['bizday'] = np.where(dataframe['weekday'] == (0 or 6),0,1)
    dataframe['bizday'] = np.where(pandas.to_datetime(dataframe['DATEn']) == datetime(2011, 5, 30), 0, dataframe['bizday'])

    features = dataframe[['rain', 'bizday']]
    dummy_units = pandas.get_dummies(dataframe['UNIT'], prefix='unit')
    features = features.join(dummy_units)
    dummy_units = pandas.get_dummies(dataframe['Hour'], prefix='hour')
    features = features.join(dummy_units)

    # Values
    values = dataframe['ENTRIESn_hourly']

    # Perform linear regression
    intercept, params = linear_regression(features, values)

    predictions = intercept + np.dot(features, params)
    return predictions

# predictions = predictions(file_path)

import pandas
import numpy as np
import scipy
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

def plot_residuals(turnstile_weather, predictions):
    '''
    Using the same methods that we used to plot a histogram of entries
    per hour for our data, why don't you make a histogram of the residuals
    (that is, the difference between the original hourly entry data and the predicted values).
    Try different binwidths for your histogram.

    Based on this residual histogram, do you have any insight into how our model
    performed?  Reading a bit on this webpage might be useful:

    http://www.itl.nist.gov/div898/handbook/pri/section2/pri24.htm
    '''
    dataframe = pandas.read_csv(turnstile_weather)
    x = (dataframe['ENTRIESn_hourly'] - predictions)
    # the histogram of the data
    n, bins, patches = plt.hist(x, bins=100, normed=1, facecolor='green', alpha=0.5)
    # add a 'best fit' line
    mu = 0 # mean of distribution
    sigma = np.std(x) # standard deviation of distribution
    y = mlab.normpdf(bins, mu, sigma)
    plt.plot(bins, y, 'r--')
    plt.xlabel('Residuals')
    plt.ylabel('Probability')
    plt.title(r'Histogram of Residuals:') #$\mu=100$, $\sigma=15$')

    x1,x2,y1,y2 = plt.axis()
    print x1,x2,y1,y2
    plt.axis((x1,x2,y1,0.0007))
    # Tweak spacing to prevent clipping of ylabel
    plt.subplots_adjust(left=0.15)

    kurtosis = scipy.stats.kurtosis(x)
    print kurtosis
    plt.show()

    # return plt
# plot_residuals(file_path, predictions)

def plot_scatter_residuals(turnstile_weather, predictions):
    dataframe = pandas.read_csv(turnstile_weather)

    # x = dataframe['ENTRIESn_hourly']
    x = predictions
    # y = predictions
    y = dataframe['ENTRIESn_hourly'] - predictions
    # colors = np.random.rand(N)
    # area = np.pi * (15 * np.random.rand(N))**2 # 0 to 15 point radiuses

    plt.scatter(x, y, alpha=0.5)
    plt.show()

# plot_scatter_residuals(file_path, predictions(file_path))


import numpy as np
import scipy
import matplotlib.pyplot as plt
import sys

def compute_r_squared(turnstile_weather, predictions):
    '''
    In exercise 5, we calculated the R^2 value for you. But why don't you try and
    and calculate the R^2 value yourself.

    Given a list of original data points, and also a list of predicted data points,
    write a function that will compute and return the coefficient of determination (R^2)
    for this data.  numpy.mean() and numpy.sum() might both be useful here, but
    not necessary.

    Documentation about numpy.mean() and numpy.sum() below:
    http://docs.scipy.org/doc/numpy/reference/generated/numpy.mean.html
    http://docs.scipy.org/doc/numpy/reference/generated/numpy.sum.html
    '''
    dataframe = pandas.read_csv(turnstile_weather)
    data = dataframe['ENTRIESn_hourly']

    SST = np.square(data - np.mean(data)).sum()
    SSReg = np.square(data - predictions).sum()
    r_squared = 1 - SSReg / SST

    print r_squared

# compute_r_squared(file_path, predictions)

import numpy as np
import pandas
from sklearn.linear_model import SGDRegressor

"""
In this question, you need to:
1) Implement the linear_regression() procedure using gradient descent.
   You can use the SGDRegressor class from sklearn, since this class uses gradient descent.
2) Select features (in the predictions procedure) and make predictions.

"""

def normalize_features(features):
    '''
    Returns the means and standard deviations of the given features, along with a normalized feature
    matrix.
    '''
    means = np.mean(features, axis=0)
    std_devs = np.std(features, axis=0)
    normalized_features = (features - means) / std_devs
    return means, std_devs, normalized_features

def recover_params(means, std_devs, norm_intercept, norm_params):
    '''
    Recovers the weights for a linear model given parameters that were fitted using
    normalized features. Takes the means and standard deviations of the original
    features, along with the intercept and parameters computed using the normalized
    features, and returns the intercept and parameters that correspond to the original
    features.
    '''
    intercept = norm_intercept - np.sum(means * norm_params / std_devs)
    params = norm_params / std_devs
    return intercept, params

def linear_regression(features, values):
    """
    Perform linear regression given a data set with an arbitrary number of features.
    """

    # print features.shape
    # print values.shape
    clf = SGDRegressor(n_iter=10)
    results = clf.fit(features, values)
    # print results
    norm_intercept = results.intercept_
    norm_params = results.coef_
    # print norm_intercept
    # print norm_params
    return norm_intercept, norm_params

def predictions(dataframe):
    '''
    The NYC turnstile data is stored in a pandas dataframe called weather_turnstile.
    Using the information stored in the dataframe, let's predict the ridership of
    the NYC subway using linear regression with gradient descent.

    You can download the complete turnstile weather dataframe here:
    https://www.dropbox.com/s/meyki2wl9xfa7yk/turnstile_data_master_with_weather.csv

    Your prediction should have a R^2 value of 0.40 or better.
    You need to experiment using various input features contained in the dataframe.
    We recommend that you don't use the EXITSn_hourly feature as an input to the
    linear model because we cannot use it as a predictor: we cannot use exits
    counts as a way to predict entry counts.

    Note: Due to the memory and CPU limitation of our Amazon EC2 instance, we will
    give you a random subset (~50%) of the data contained in
    turnstile_data_master_with_weather.csv. You are encouraged to experiment with
    this exercise on your own computer, locally.

    If you receive a "server has encountered an error" message, that means you are
    hitting the 30-second limit that's placed on running your program. Try using a
    smaller number of features or fewer iterations.
    '''
    ################################ MODIFY THIS SECTION #####################################
    # Select features. You should modify this section to try different features!             #
    # We've selected rain, precipi, Hour, meantempi, and UNIT (as a dummy) to start you off. #
    # See this page for more info about dummy variables:                                     #
    # http://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html          #
    ##########################################################################################
    # features = dataframe[['rain', 'precipi', 'Hour', 'meantempi']]
    features = dataframe[['Hour', 'meandewpti', 'meanpressurei', 'fog', 'rain', 'meanwindspdi', 'meantempi', 'precipi']]

    dummy_units = pandas.get_dummies(dataframe['UNIT'], prefix='unit')
    features = features.join(dummy_units)

    # Values
    values = dataframe['ENTRIESn_hourly']

    # Get numpy arrays
    features_array = features.values
    values_array = values.values

    means, std_devs, normalized_features_array = normalize_features(features_array)

    # Perform gradient descent
    norm_intercept, norm_params = linear_regression(normalized_features_array, values_array)

    intercept, params = recover_params(means, std_devs, norm_intercept, norm_params)

    predictions = intercept + np.dot(features_array, params)
    # The following line would be equivalent:
    # predictions = norm_intercept + np.dot(normalized_features_array, norm_params)

    return predictions