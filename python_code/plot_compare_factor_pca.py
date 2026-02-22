#!/usr/bin/env python

from matplotlib import pyplot as plt
#from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np
import pandas as pd
import datetime

import argparse
import os
from pathlib import Path

from dateutil.relativedelta import relativedelta

#NAMESPACE = 'predictions'
#NAMESPACE = 'predictions_testing'

"""
yo
"""

def resolve_env_path(var, default):
    p = os.environ.get(var, "").strip()
    return Path(os.path.expanduser(p)) if p else Path(default).expanduser()



HOME = Path.home()
PAPER_PATH = resolve_env_path("PAPER_PATH", HOME / "Documents/github/neoval-project")

IN       = resolve_env_path("IN",         PAPER_PATH / "data")
OUT      = resolve_env_path("OUT",        PAPER_PATH / "data_out")
OUT.mkdir(parents=True, exist_ok=True)

pc_file = IN / "df_pcs.csv"
factor_file = IN / "df_factor_trends.csv"

def make_file_path(file, base_path = OUT):
    if len(file) < 1:
        raise Exception('Empty string provided for filename')
    elif file[0] == '/':
        return file

    return os.path.join(base_path, file)

def insert_subname(file, subname):

    if not subname:
        return file
    L = file.split('.')
    if len(L) < 2:
        raise Exception('Provide file format via dot')

    L[-2] = f"{L[-2]}_{subname}"
    return '.'.join(L)


def show_or_save_fig(args, handle=None, subname=''):

    handle = handle or plt

    if args.file == 'show':
        handle.show()
    else:
        file = insert_subname(args.file, subname)
        file_path = make_file_path(file)
        print('Saving figure to {0}'.format(file_path))

        kwargs = {}
        if args.fine_dpi:
            kwargs['dpi'] = 300
        handle.savefig(file_path, bbox_inches='tight', pad_inches=0.01, **kwargs)


def date2i_month(date_str, start_date = datetime.datetime(1990, 1, 1)):
    date = datetime.datetime.strptime(date_str, '%Y-%m-%d')
    delta = (date.year - start_date.year) * 12 + (date.month - start_date.month)
    return delta

def i_month2date(i_month, start_date=datetime.datetime(1990, 1, 1)):
    target_date = start_date + relativedelta(months=i_month)
    return target_date.strftime('%Y-%m-%d')



def region2label(region, remove_greater=True, max_components=3, max_len=50):

    region = region.replace('AUSTRALIAN CAPITAL TERRITORY', 'ACT').replace('SYDNEY - ','')

    region = region.strip()
    if remove_greater:
        region = region.replace('GREATER ', '')

    if ',' in region:
        return ",".join([region2label(e, max_components=max_components, max_len=max_len) for e in region.split(',')])

    def word2lower(string):

        if string in ['NSW', 'QLD', 'WA', 'SA', 'ACT', 'TAS', 'VIC', 'NT']:
            return string

        return string[0].upper() + string[1:].lower()


    def to_lower(string):
        comps = [word2lower(e) for e in string.split(" ")]
        return " ".join(comps)

    def abbrev(string):
        if len(string) > max_len:
            return ''.join(e[0] for e in string.split(' '))
        return string

    components = [ abbrev(to_lower(e.strip())) for e in region.split('-')][:max_components]

    return " - ".join(components).strip()




def normalize(df, start_year=None, end_year=2022):

    mask = np.ones(len(df), dtype=bool)
    if start_year is not None:
        mask &= (df.index >= f"01-01-{start_year}")
    if end_year is not None:
        mask &= (df.index <= f"12-31-{end_year}")
    df_win = df.loc[mask]

    df = (df - df_win.mean())/df_win.std()

    return df




if __name__ == '__main__':


    parser = argparse.ArgumentParser()

    parser.add_argument('-f', '--file', help = "Filename to save to. To show on screen 'show'.", default = 'show' )
    parser.add_argument('--fine_dpi', action='store_true', help="Finer dpi.", default = False)

    parser.add_argument('--level', help = "Regional level.", default = 'major_city', required=False)
    parser.add_argument('--other_cat', help = "PC components including and above this level are of category other.", default = 3, type=int, required=False)
    parser.add_argument('--user_suffix', help = "user defined suffix to use in output table.", default = '_tm')

    parser.add_argument('--omit_labels', action='store_true', help = "Omit region labels as text within graph.", default = False)

    parser.add_argument('--x_label', help = "Additional text for x-label, e.g. Sydney.", default = '')

    parser.add_argument('--omit_nt', action='store_true', help = "Omit the Northern Territory.", default = False)

    parser.add_argument('--no_labels', action='store_true', help = "Omit region labels within graph: if there are too many.", default = False)

    parser.add_argument('--all_labels', action='store_true', help = "Label every single region.", default = False)

    parser.add_argument('--end_date', help = "End date of the PCA fit window.", default = '2024-07-01')

    parser.add_argument('--window_start_date', help = "Start date of the growth window.", default = '1998-04-01')

    parser.add_argument('--window_end_date', help = "End date of the growth window.", default = '2024-07-01')

    parser.add_argument('--test_mode', help = "Test mode.", default = False, required=False, action='store_true')


    args = parser.parse_args()

    # Get the default Matplotlib color cycle
    #default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    #color_cycle = itertools.cycle(default_colors)

    default_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:gray']


    if not pc_file.exists():
        raise FileNotFoundError(f"Missing {pc_file}")
    df_pcs = pd.read_csv(pc_file)

    df_pcs = df_pcs.pivot(index='period', columns='mode', values='pc_value')
    df_pcs.index = pd.DatetimeIndex(df_pcs.index)

    if not factor_file.exists():
        raise FileNotFoundError(f"Missing {factor_file}")
    df_factors = pd.read_csv(factor_file)

    df_factors = df_factors.set_index('month_date')
    df_factors.index = pd.DatetimeIndex(df_factors.index)

    descriptions = { 0 : 'Market', 1 : 'Mining vs Sydney', 2 : 'Lifestyle' }
    factors = { 0 : 'National Index', 1 : 'Perth-Sydney Spread', 2 : 'Lifestyle Spread' }
    df_corr = pd.DataFrame(index=range(1,4), columns=[ 'Factor', 'Corr', 'Description'])
    df_corr.index.name = 'PC series'
    for i, row in enumerate(df_corr.index):
        df_corr.loc[row, 'Corr']=df_factors[df_factors.columns[i]].corr(df_pcs[i])
        df_corr.loc[row, 'Description'] = descriptions[i]
        df_corr.loc[row, 'Factor'] = factors[i]

    print(df_corr.reset_index().to_latex(index=False,float_format='%0.2f'))

    i = 1
    scale = max(df_factors[df_factors.columns[i]])/ max(df_pcs[i])
    df_pcs[i] = df_pcs[i]*scale

    plt.close()
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10, 4))

    ax1.plot(df_pcs[i], linewidth=2, label=f'PC{i+1} (scaled)')
    ax1.plot(df_factors[df_factors.columns[i]], linewidth=2, label=f"Factor {i+1}")


    ax1.legend(
        loc='upper left',
        fontsize=7
    )

    ax1.set_ylabel('PC')

    ax1.set_title(f'a) Comparison of PC{i+1} and Factor {i+1} over time')
    ax1.set_xticks([])

    i += 1
    scale = max(df_factors[df_factors.columns[i]])/ max(df_pcs[i])
    df_pcs[i] = df_pcs[i]*scale


    ax2.plot(df_pcs[i], linewidth=2, label=f'PC{i+1} (scaled)')
    ax2.plot(df_factors[df_factors.columns[i]], linewidth=2, label=f"Factor {i+1}")


    ax2.legend(
        loc='upper left',
        fontsize=7
    )

    ax2.set_ylabel('PC')

    ax2.set_title(f'b) Comparison of PC{i+1} and Factor {i+1} over time')
    #ax2.set_xticks([])


    plt.tight_layout()



    if args.file == 'show':
        plt.show()
    else:
        for h, subname in [ (fig, f'')]:
            show_or_save_fig(args, handle=h, subname=subname)

