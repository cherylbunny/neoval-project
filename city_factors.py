#!/usr/bin/env python

from matplotlib import pyplot as plt
#from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np
import pandas as pd
import os

home_dir = os.path.expanduser("~")
paper_path = os.path.join(home_dir, 'Documents/github/neoval-project')
data_path = os.path.join(paper_path, 'data')

if __name__ == '__main__':

    df = pd.read_csv(os.path.join(data_path, 'city_indexes.csv'))
    df_pcs = pd.read_csv(os.path.join(data_path, 'df_pcs.csv'))
    df_p = df_pcs.pivot(columns='mode', index='period', values='pc_value')



    df = df.set_index('month_date')
    df.index = pd.to_datetime(df.index)
    df_p.index = pd.to_datetime(df_p.index)
    # df_p already has 'period' as index from your example

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Left y-axis: Brisbane minus Sydney
    series1 = df['GREATER BRISBANE'] - df['GREATER SYDNEY']
    ax1.plot(series1.index, series1, color='tab:blue', label='Greater Brisbane − Greater Sydney')
    ax1.set_ylabel('House Price Index Difference\n(Brisbane − Sydney, % growth)', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Right y-axis: PC1 from df_p
    series2 = df_p[1]
    ax2 = ax1.twinx()
    ax2.plot(series2.index, series2, color='tab:red', label='PC1: [Insert interpretation]')
    ax2.set_ylabel('Principal Component 1 (standardized units)', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # Title and legends
    #fig.suptitle('Greater Brisbane vs Greater Sydney Price Growth\nand PC1 Time Series', fontsize=14)

    # Combined legend
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    fig.tight_layout()
    plt.savefig(os.path.join(paper_path, 'figures', 'brisbane_sydney_pc1_comparison.pdf'))

