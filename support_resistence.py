# Support and Resistance Levels
# This module provides functions to calculate support and resistance levels for financial time series data
# based on volume-weighted price data.
# It uses Gaussian kernel density estimation to identify significant price levels and forecasts a future price based on these levels.
# This code is designed to be run as a script, taking a CSV file as input with 'Date', 'Close' and 'Volume'.
# The output includes a DataFrame of resistance and support levels along with a forecasted price.

## Usage:
# python support_resistence.py <path_to_csv_file>

import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
from scipy import signal
import sys


def profile_gaussiankde(price, df_params, factor):
    kde_factor = float(factor) # standard 0.05
    num_samples = 500
    kde = gaussian_kde(price,weights=df_params,bw_method=kde_factor)
    xr = np.linspace(price.min(),price.max(),num_samples)
    kdy = kde(xr)
    ticks_per_sample = (xr.max() - xr.min()) / num_samples
    min_prom = kdy.max() * 0.1
    width_range=0.1
    peaks, peak_props = signal.find_peaks(kdy, prominence=min_prom, width=width_range)
    return kde, kdy, xr, peaks, peak_props, ticks_per_sample

def kdevolumeprice(df,factor):
    kde, kdy, xr, peaks, peak_props, ticks_per_sample = profile_gaussiankde(df['Close'], df['Volume'], factor)

    pkx = xr[peaks]
    # Dataframe of the key levels
    left_base = peak_props['left_bases']
    right_base = peak_props['right_bases']
    int_from = xr.min() + (left_base * ticks_per_sample)
    int_to = xr.min() + (right_base * ticks_per_sample)

    density = [kde.integrate_box_1d(x0, x1) for x0, x1 in zip(int_from, int_to)]

    # Table of the key levels
    res_sup = pd.DataFrame ({
        'max price':pkx.tolist(),
        'density':density,
    })
    res_sup = res_sup.sort_values(by='density', ascending=False).reset_index(drop=True)
    # To forecast the future price we can use a weighted average approach. 
    # Each price is assigned a weight based on its density percentage and weight values.
    weighted_sum = (res_sup['max price'] * res_sup['density']).sum() 
    sum_of_weights = res_sup['density'].sum() 
    forecasted_price = weighted_sum / sum_of_weights
    forecasted_price = "{:.2f}".format(forecasted_price)
    return res_sup, forecasted_price 
'''
if __name__ == "__main__":
    try:
        df = pd.read_csv(sys.argv[1], index_col=0)
    except Exception as e:
       print(f"Error reading CSV file: {e}")
       sys.exit(1)

    if df.empty:
        print("DataFrame is empty. Please provide a valid CSV file.")
        sys.exit(1)
    resistence_support, forecasted_price = kdevolumeprice(df=df,factor=0.05)
    print(f"Resistance and Support Levels:\n{resistence_support}\nLast price: {df['Close'].iloc[-1]}\nForecasted Price: {forecasted_price}")
    sys.exit(0)
'''
