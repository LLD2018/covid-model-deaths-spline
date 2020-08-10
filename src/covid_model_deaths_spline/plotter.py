from typing import List, Tuple
from pathlib import Path
from functools import reduce

import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import warnings
warnings.simplefilter('ignore')

DOY_COLORS = {
    'Sunday':'grey',
    'Monday':'indianred',
    'Tuesday':'peru',
    'Wednesday':'olive',
    'Thursday':'seagreen',
    'Friday':'royalblue',
    'Saturday':'darkmagenta'
}


def get_plot_idx(i: int, n_vars: int):
    if n_vars > 1:
        top_idx = 0, i
        bottom_idx = 1, i
    else:
        top_idx = 0
        bottom_idx = 1
    return top_idx, bottom_idx


def plotter(df: pd.DataFrame, plot_vars: List[str], draw_df: pd.DataFrame,
            model_labels: List[str], draw_ranges: List[Tuple[int, int]],
            plot_file: str = None):
    # set up plot
    sns.set_style('whitegrid')
    n_cols = max(len(plot_vars), 1)
    n_rows = 3
    widths = [1] * n_cols
    if n_cols < 3:
        heights = [1, 1, 1]
    else:
        heights = [1, 1, 1.5]
    fig = plt.figure(figsize=(n_cols*11, 24), constrained_layout=True)
    gs = fig.add_gridspec(n_rows, n_cols, width_ratios=widths, height_ratios=heights)

    # aesthetic features
    raw_points = {'c':'dodgerblue', 'edgecolors':'navy', 's':100, 'alpha':0.75}
    raw_lines = {'color':'navy', 'alpha':0.5, 'linewidth':3}
    cfr_lines = {'color':'forestgreen', 'alpha':0.5, 'linewidth':3}
    hfr_lines = {'color':'darkorchid', 'alpha':0.5, 'linewidth':3}
    smoothed_pred_lines = {'color':'firebrick', 'alpha':0.75, 'linewidth':3}
    smoothed_pred_area = {'color':'firebrick', 'alpha':0.25}

    # cases
    indep_idx = 1
    if 'Confirmed case rate' in plot_vars:
        ax_cfr = fig.add_subplot(gs[0, indep_idx])
        ax_cfr.scatter(df['Confirmed case rate'],
                       df['Death rate'],
                       **raw_points)
        ax_cfr.plot(df['Confirmed case rate'],
                    df['Predicted death rate (CFR)'],
                    **cfr_lines)
        ax_cfr.plot(df['Confirmed case rate'],
                    df['Smoothed predicted death rate'],
                    **smoothed_pred_lines)    
        ax_cfr.set_xlabel('Cumulative case rate', fontsize=14)
        ax_cfr.set_ylabel('Cumulative death rate', fontsize=14)
        indep_idx += 1

    # hospitalizations
    if 'Hospitalization rate' in plot_vars:
        ax_hfr = fig.add_subplot(gs[0, indep_idx])
        ax_hfr.scatter(df['Hospitalization rate'],
                       df['Death rate'],
                       **raw_points)
        ax_hfr.plot(df['Hospitalization rate'],
                    df['Predicted death rate (HFR)'],
                    **hfr_lines)
        ax_hfr.plot(df['Hospitalization rate'],
                    df['Smoothed predicted death rate'],
                    **smoothed_pred_lines)
        ax_hfr.set_xlabel('Cumulative hospitalization rate', fontsize=14)
        ax_hfr.set_ylabel('Cumulative death rate', fontsize=14)
        
    for i, smooth_variable in enumerate(plot_vars):
        # get index
        top_idx, bottom_idx = get_plot_idx(i, n_cols)
        
        # cumulative
        raw_variable = smooth_variable.replace('Smoothed ', '').capitalize()
        plot_label = raw_variable.lower().replace(' rate', 's')
        if ~df[raw_variable].isnull().all():
            if 'death' in smooth_variable.lower():
                ax_cumul = fig.add_subplot(gs[top_idx])
                ax_cumul.plot(df['Date'], df[raw_variable] * df['population'], **raw_lines)
                ax_cumul.scatter(df['Date'], df[raw_variable] * df['population'], **raw_points)
                ax_cumul.set_ylabel(f'Cumulative {plot_label}', fontsize=14)
                ax_cumul.set_xlabel('Date', fontsize=14)


            # daily
            ax_daily = fig.add_subplot(gs[bottom_idx])
            ax_daily.plot(df['Date'],
                          np.diff(df[raw_variable], prepend=0) * df['population'],
                          **raw_lines)
            ax_daily.scatter(df['Date'],
                             np.diff(df[raw_variable], prepend=0) * df['population'],
                             **raw_points)
            ax_daily.axhline(0, color='black', alpha=0.5)
            if 'death' in smooth_variable.lower():
                ax_daily.set_xlabel('Date', fontsize=14)
            else:
                ax_daily.set_xlabel('Date (+8 days)', fontsize=14)
            ax_daily.set_ylabel(f'Daily {plot_label}', fontsize=14)
    
    # predictions - cumul
    ax_cumul = fig.add_subplot(gs[0, 0])
    ax_cumul.plot(df['Date'], df['Predicted death rate (CFR)'] * df['population'],
                  **cfr_lines)
    ax_cumul.plot(df['Date'], df['Predicted death rate (HFR)'] * df['population'],
                  **hfr_lines)
    ax_cumul.plot(df['Date'],
                  df['Smoothed predicted death rate'] * df['population'],
                  **smoothed_pred_lines)
    ax_cumul.fill_between(
        df['Date'],
        df['Smoothed predicted death rate lower'] * df['population'],
        df['Smoothed predicted death rate upper'] * df['population'],
        **smoothed_pred_area
    )
    
    # predictions - daily
    ax_daily = fig.add_subplot(gs[1, 0])
    ax_daily.plot(df['Date'],
                  np.diff(df['Predicted death rate (CFR)'], prepend=0) * df['population'],
                  **cfr_lines)
    ax_daily.plot(df['Date'],
                  np.diff(df['Predicted death rate (HFR)'], prepend=0) * df['population'],
                  **hfr_lines)
    ax_daily.plot(df['Date'],
                  df['Smoothed predicted daily death rate'] * df['population'],
                  **smoothed_pred_lines)
    ax_daily.fill_between(
        df['Date'],
        df['Smoothed predicted daily death rate lower'] * df['population'],
        df['Smoothed predicted daily death rate upper'] * df['population'],
        **smoothed_pred_area
    )
    
    ## smoothed draws - ln(rate)
    # floor
    floor = 0.05 / df['population'].values[0]
    
    # format draws
    draw_df = draw_df.copy()
    draw_cols = [col for col in draw_df.columns if col.startswith('draw_')]
    draw_data = np.diff(draw_df[draw_cols], axis=0, prepend=0)
    draw_df[draw_cols] = draw_data
    
    # format model inputs
    df = df.copy()
    
    for input_var in ['Death rate', 'Predicted death rate (CFR)', 'Predicted death rate (HFR)']:
        df[input_var][1:] = np.diff(df[input_var])
        df.loc[df[input_var] < floor, input_var] = floor
    
    # plot draws
    show_draws = np.arange(0, len(draw_cols), 10).tolist()
    show_draws += [len(draw_cols) - 1]
    ax_draws = fig.add_subplot(gs[2:, 0:])
    for model_label, draw_range in zip(model_labels, draw_ranges):
        # which day
        if len(model_labels) > 1:
            doy = model_label[model_label.find('(') + 1:model_label.find(')')]
            color = DOY_COLORS[doy]
        else:
            color = 'firebrick'
        # submodel draws
        ax_draws.plot(draw_df['Date'],
                      np.log(draw_df[[f'draw_{d}' for d in range(*draw_range) if d in show_draws]]),
                      color=color, alpha=0.1)
        # submodel means
        ax_draws.plot(draw_df['Date'],
                      np.log(draw_df[[f'draw_{d}' for d in range(*draw_range)]]).mean(axis=1),
                      color=color, linewidth=3, label=model_label)
    # overall mean
    ax_draws.plot(draw_df['Date'],
                  np.log(draw_df[draw_cols]).mean(axis=1),
                  color='black', linestyle='--', linewidth=3)
    ax_draws.set_ylabel('ln(daily death rate)', fontsize=18)
    ax_draws.legend(loc=2, ncol=1, fontsize=16)
    ax_draws.set_xlabel('Date', fontsize=14)
    
    # plot data
    if any(~df['Death rate'].isnull()):
        ax_draws.plot(df['Date'],
                      np.log(df['Death rate']),
                      **raw_lines)
        ax_draws.scatter(df['Date'],
                         np.log(df['Death rate']),
                         **raw_points)
    ax_draws.plot(df['Date'],
                  np.log(df['Predicted death rate (CFR)']),
                  **cfr_lines)
    ax_draws.plot(df['Date'],
                  np.log(df['Predicted death rate (HFR)']),
                  **hfr_lines)
    ##
    
    location_name = df.loc[~df['location_name'].isnull(), 'location_name'].values
    location_id = int(df.loc[~df['location_id'].isnull(), 'location_id'].values[0])
    if location_name.size > 0:
        location_name = location_name[0]
        plot_label = f'{location_name} [{location_id}]'
    else:
        plot_label = str(location_id)
    fig.suptitle(plot_label, y=1.0025, fontsize=24)
    fig.tight_layout()
    if plot_file:
        fig.savefig(plot_file, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
        
        
def calc_change(data: pd.DataFrame, plot_cols: List[str]) -> pd.DataFrame:
    prior_week = data['Date'].astype(str).values[-14] + ' to ' + data['Date'].astype(str).values[-8]
    last_week = data['Date'].astype(str).values[-7] + ' to ' + data['Date'].astype(str).values[-1]
    location_name = data['location_name'].unique().item()

    prior_week_data = data.iloc[-14:-7][plot_cols].sum()  # skipna=False
    last_week_data = data.iloc[-7:][plot_cols].sum()  # skipna=False
    chng_data = ((last_week_data - prior_week_data) / prior_week_data) * 100
    chng_data = chng_data.replace([np.inf, -np.inf], np.nan)
    chng_data = chng_data.fillna(0)
    location_id = data['location_id'].unique().item()
    location_name = data['location_name'].unique().item()
    
    data = pd.DataFrame(chng_data).T
    data['location_id'] = location_id
    data['location_name'] = location_name
    
    return data
        
        
def ratio_plot_helper(data: pd.DataFrame, plot_cols: List[str], pdf):
    if len(data) >= 14:
        level_3 = data['level_3'].unique().item()
        data = data.groupby('location_id').apply(lambda x: calc_change(x, plot_cols)).reset_index(drop=True)
        data = data.rename(index=str, columns={'Death rate': 'Deaths',
                                               'Confirmed case rate': 'Cases', 
                                               'Hospitalization rate': 'Hosp.'})
        names_dict = dict(zip(data['location_id'], data['location_name']))
        del data['location_name']
        data = data.set_index('location_id')
        location_ids = data.index.values
        n_locations = location_ids.size
        
        if level_3 == 102:
            map_data = pd.read_csv('../../data/US_graph_map.csv', header=None)
            plot_labels = pd.read_csv('../../data/US_state_labels.csv')
            label_dict = dict(zip(plot_labels['location_id'], plot_labels['local_id']))
            n_rows, n_cols = map_data.shape
            figsize = (16, 10)
        else:
            if n_locations > 2:
                if n_locations >= 30:
                    n_cols = 6
                elif n_locations >= 20:
                    n_cols = 5
                elif n_locations >= 12:
                    n_cols = 4
                else:
                    n_cols = 3
                n_rows = int(np.ceil(n_locations / n_cols))
                figsize = (16, 10)
            else:
                n_cols = n_locations
                n_rows = 1
                figsize = (11, 8.5)
                
        y_min_max = np.percentile(data.values, (0, 100))
        y_min = min(0, y_min_max[0] * 1.025)
        y_max = max(0, y_min_max[1] * 1.025)

        fig = plt.figure(figsize=figsize, constrained_layout=True)
        gs = fig.add_gridspec(n_rows, n_cols, width_ratios=[1] * n_cols, height_ratios=[1] * n_rows)
        
        i = 0
        for location_id in location_ids:
            plot_data = data.loc[location_id]
            plot_name = names_dict[location_id]
            if level_3 == 102:
                plot_row, plot_col = np.where(map_data.values == plot_name)
                plot_name = label_dict[location_id]
                plot_row = plot_row.item()
                plot_col = plot_col.item()
            else:
                plot_row = int(i / n_cols)
                plot_col = i % n_cols
            loc_ax = fig.add_subplot(gs[plot_row, plot_col])
            loc_ax.bar(plot_data.index, 0, color='black', alpha=0)
            loc_ax.bar(plot_data[plot_data >= 0].index, 
                       plot_data[plot_data >= 0].values,
                       color='indianred', edgecolor='maroon', alpha=0.75)
            loc_ax.bar(plot_data[plot_data < 0].index, 
                       plot_data[plot_data < 0].values,
                       color='dodgerblue', edgecolor='navy', alpha=0.75)
            loc_ax.axhline(0, linestyle='--', alpha=0.75, color='black')
            loc_ax.set_ylim(y_min, y_max)
            if plot_col == 0:
                loc_ax.set_ylabel('% change', fontsize=12)
            else:
                loc_ax.get_yaxis().set_visible(False)
            if plot_row == n_rows - 1:
                loc_ax.tick_params(axis='x', labelsize=12, labelrotation=60)
            else:
                loc_ax.get_xaxis().set_visible(False)
            if np.max(np.abs(plot_data)) < 10:
                loc_ax.set_ylim(-10, 10)
            if level_3 == 102:
                loc_ax.text(0.875, 0.875, plot_name, ha='center', va='center', 
                            transform=loc_ax.transAxes)
            else:
                loc_ax.set_title(plot_name)
            i += 1
        
        fig.tight_layout()
        pdf.savefig()
        plt.close(fig)
        
        return data
        
        
def agg_to_level_3(data: pd.DataFrame, agg_hierarchy: pd.DataFrame, plot_cols: List[str]) -> pd.DataFrame:
    agg_out = (data['level_4'].notnull()) & (data['location_id'] != data['level_4'])
    agg_data = data.loc[agg_out]
    agg_data['location_id'] = agg_data['level_4'].astype(int)
    del agg_data['location_name']
    agg_data = agg_data.merge(agg_hierarchy[['location_id', 'location_name']])
    
    agg_data[plot_cols] = agg_data[plot_cols].values * agg_data[['population']].values
    agg_cols = plot_cols + ['population']
    id_cols = [col for col in data.columns if col not in agg_cols]
    agg_data = agg_data.groupby(id_cols, as_index=False)[agg_cols].sum()
    agg_data['population'] = agg_data.groupby('location_id')['population'].transform(max)
    agg_data[plot_cols] = agg_data[plot_cols].values / agg_data[['population']].values
    
    data = data.loc[~agg_out].append(agg_data)
    
    return data

        
def ratio_plot(data_list: List[pd.DataFrame], hierarchy: pd.DataFrame, agg_hierarchy: pd.DataFrame, output_root: Path):
    data = reduce(lambda x, y: pd.merge(x,
                                        y.rename(index=str, columns={'True date':'Date'}),
                                        how='left'),
                  data_list)
    plot_cols = [col for col in data.columns if col not in ['location_id', 'Date', 'population']]
    hierarchy = hierarchy.copy()
    hierarchy['level_3'] = hierarchy['path_to_top_parent'].apply(lambda x: int(x.split(',')[3]))
    hierarchy['level_4'] = hierarchy['path_to_top_parent'].apply(lambda x: int(x.split(',')[4]) if len(x.split(',')) >= 5 else np.nan)
    hierarchy['sort_order'] = hierarchy.groupby('level_3')['sort_order'].transform(min)
    data = data.merge(hierarchy[['location_id', 'location_name', 'level_3', 'level_4', 'sort_order']])
    data = data.sort_values(['location_id', 'Date']).reset_index(drop=True)
    data[plot_cols] = np.vstack(data.groupby('location_id')
                                .apply(lambda x: np.diff(x[plot_cols], axis=0, prepend=0)).values)
    data = agg_to_level_3(data, agg_hierarchy, plot_cols)
    data = data.sort_values(['sort_order', 'location_id', 'Date']).reset_index(drop=True)
    
    with PdfPages(output_root / 'last_week_change.pdf') as pdf:
        data = data.groupby('sort_order').apply(lambda x: ratio_plot_helper(x, plot_cols, pdf)).reset_index()
    data.to_csv(output_root / 'last_week_change.csv', index=False)
    