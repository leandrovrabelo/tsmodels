import numpy as np
import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt

def plot_ccf(x, y, lags=20, figsize=(14,6)):
    '''
    This function plots the cross correlation between two time series.
    '''
    
    xname = pd.DataFrame(x).columns.values[0]
    yname = pd.DataFrame(y).columns.values[0]
    
    corr = np.array([x.corr(y.shift(i)) for i in range(-lags, lags)])

    # CALCULATING THE STANDARD ERROR AND THE CONFIDENCE INTERVALS
    right = np.array([sqrt((1/len(x)) * (1 + 2 * sum(corr[:i]**2))) * 1.96 for i in range(1,lags+1)])
    lower_right = -right
    upper_right = right

    left = np.array([sqrt((1/len(x)) * (1 + 2 * sum(corr[-i::]**2))) * 1.96 for i in range(1,lags+1)])
    lower_left = -left[::-1]
    upper_left = left[::-1]
    
    # PLOTING THE CORRELATION
    plt.figure(figsize=figsize)
    plt.stem(corr,linefmt='cornflowerblue', markerfmt='bo', basefmt='cornflowerblue', label='Corr. Cruzada', use_line_collection=True)
    plt.vlines(len(corr)/2, ymax=corr.max(), ymin=corr.min(), color='black', lw=3, alpha=1, label='Base Zero')
    plt.plot(corr.argmax(),corr.max(), 'o', markersize=8, color='red', label=f'Max Lag {int(len(corr)/2)-corr.argmax()}')
    plt.fill_between(range(lags,lags*2), lower_right, 
                 upper_right, alpha=0.25, color='cornflowerblue')
    plt.fill_between(range(1,lags+1), lower_left, 
                 upper_left, alpha=0.25, color='cornflowerblue')
    plt.title(f'Correlation Between {xname} and {yname} with {lags} lags')
    plt.legend()
    plt.show()