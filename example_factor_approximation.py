#!/usr/bin/env python

from matplotlib import pyplot as plt
#from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np
import pandas as pd
import os
import argparse

home_dir = os.path.expanduser("~")
paper_path = os.path.join(home_dir, 'Documents/github/neoval-project')
data_path = os.path.join(paper_path, 'data')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--region', help = "Region to plot.", default = "SYDNEY - BLACKTOWN", required=False)


    df_coefs = pd.read_csv(os.path.join(data_path, 'df_reg_coefs.csv'))
    df = pd.read_csv(os.path.join(data_path, 'df_factor_trends.csv'))
    df_indexes = pd.read_csv(os.path.join(data_path, 'indexes_city_and_sa4.csv'))


    df_coefs = df_coefs.set_index('region')
    df_indexes['month_date'] = pd.to_datetime(df_indexes['month_date'])


    region=parser.parse_args().region

    S_market=df_coefs.loc[region, 'market'] * df['market']
    S = S_market  + df_coefs.loc[region, 'mining'] * df['mining']

    plt.close()

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(df_indexes['month_date'], S_market, label='market')
    ax1.plot(df_indexes['month_date'], S, label='market + factor 2')
    ax1.plot(df_indexes['month_date'], df_indexes[region], label='actual index')
    ax1.legend()
    ax1.set_title(region)


    fig.tight_layout()
    plt.savefig(os.path.join(paper_path, 'figures', f'example_factor_approx.pdf'))

