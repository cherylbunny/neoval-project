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
    parser.add_argument('--factor', help = "Factor to plot.", type=int, default = 1, required=False)

    show_factor = parser.parse_args().factor - 1

    df = pd.read_csv(os.path.join(data_path, 'df_factor_trends.csv'))
    df_pcs = pd.read_csv(os.path.join(data_path, 'df_pcs.csv'))
    df_p = df_pcs.pivot(columns='mode', index='period', values='pc_value')

    factors = ["market", "mining", "lifestyle"]

    df = df.set_index('month_date')
    df.index = pd.to_datetime(df.index)
    df_p.index = pd.to_datetime(df_p.index)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Left y-axis: Brisbane - Sydney spread
    series1 = df[factors[show_factor]]
    ax1.plot(series1.index, series1, color='tab:blue', label=f'Factor {show_factor+1}')
    ax1.set_ylabel(f'Factor {show_factor+1}', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Right y-axis: PC1 from df_p
    series2 = df_p[show_factor]
    ax2 = ax1.twinx()
    ax2.plot(series2.index, series2, color='tab:red', label=f'PC{show_factor+1}')
    ax2.set_ylabel(f'Principal Component {show_factor+1} (standardized units)', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # Title and legends
    #fig.suptitle('Greater Brisbane vs Greater Sydney Price Growth\nand PC1 Time Series', fontsize=14)

    # Combined legend
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    fig.tight_layout()
    plt.savefig(os.path.join(paper_path, 'figures', f'factor{show_factor+1}.pdf'))

