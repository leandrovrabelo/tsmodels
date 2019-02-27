import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_ccr(x, y, nlags=20, figsize=(14,6)):
    
    xname = pd.DataFrame(x).columns.values[0]
    yname = pd.DataFrame(y).columns.values[0]
    
    x = np.array(x)
    y = np.array(y)
    
    corr = np.array([])
    for lag in np.arange(nlags):

        corr = np.append(corr, np.corrcoef(x[:-1-lag],y[lag+1:])[0][1])

    # CALCULATING THE STANDARD ERROR AND THE CONFIDENCE INTERVALS
    from math import sqrt

    upper_ci = np.array([])
    lower_ci = np.array([])
    
    for i in np.arange(nlags):
        
        std_err = sqrt((1/len(x)) * (1 + 2 * sum(corr[:i]**2))) * 1.96
                         
        upper_ci = np.append(upper_ci, std_err)
        lower_ci = np.append(lower_ci, -std_err)

    # PLOTING THE CORRELATION
    plt.figure(figsize=figsize)
    plt.plot(upper_ci, c='b', alpha=0)
    plt.plot(lower_ci, c='b', alpha=0)
    plt.stem(corr,linefmt='black', markerfmt='bo', basefmt='cornflowerblue')
    plt.fill_between(np.arange(nlags), lower_ci, upper_ci, alpha=0.25)
    plt.title(f'Correlation Between {xname} and {yname} with {nlags} lags')
    plt.show()
