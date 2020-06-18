from pathlib import Path
from typing import Callable, List
import sys

from covid_shared import shell_tools
import dill as pickle
import numpy as np
import pandas as pd
import tqdm
import yaml

from covid_model_deaths_spline.mr_spline import SplineFit


def cfr_model(location_id: int,
              data: pd.DataFrame,
              daily: bool,
              log: bool,
              dep_var: str, spline_var: str, indep_vars: List[str],
              model_dir: str, 
              model_type: str, **_) -> pd.DataFrame:
    # set up model
    df = data[data.location_id == location_id]

    # add intercept
    df['intercept'] = 1

    # log transform, setting floor of 0.01 per population
    df = df.sort_values('Date').reset_index(drop=True)
    floor = 0.01 / df['population'].values[0]
    adj_vars = {}
    for orig_var in [dep_var, spline_var] + indep_vars:
        mod_var = f'Model {orig_var.lower()}'
        df[mod_var] = df[orig_var]
        if daily:
            start_idx = df.loc[~df[mod_var].isnull()].index.values[0]
            df[mod_var][start_idx+1:] = np.diff(df[mod_var].values[start_idx:])
        if log:
            df.loc[df[mod_var] < floor, mod_var] = floor
            df[mod_var] = np.log(df[mod_var])
        adj_vars.update({orig_var:mod_var})
    df['Model log'] = log
    df['Model daily'] = daily

    # keep what we can use to predict (subset further to fitting dataset below)
    non_na = ~df[list(adj_vars.values())[1:]].isnull().any(axis=1)
    df = df.loc[non_na].reset_index(drop=True)

    # lose NAs in deaths as well for modeling
    mod_df = df.copy()
    non_na = ~mod_df[adj_vars[dep_var]].isnull()
    max_1week_of_zeros_spline = (mod_df[spline_var][::-1] == 0).cumsum()[::-1] <= 7
    mod_df = mod_df.loc[non_na & max_1week_of_zeros_spline,
                        ['intercept'] + list(adj_vars.values())].reset_index(drop=True)

    # run model and predict
    spline_options = {
        'spline_knots_type': 'frequency',
        'spline_degree': 3,
        'spline_r_linear':True,
        'spline_l_linear':True,
    }
    if not daily:
        spline_options.update({'prior_spline_monotonicity':'increasing'})
        
    # run model
    prediction_pending = True
    n_i_knots = 6
    last_days_pctile = min(0.05, 5 / len(mod_df))
    while prediction_pending:
        try:
            mr_model = SplineFit(
                data=mod_df,
                dep_var=adj_vars[dep_var],
                spline_var=adj_vars[spline_var],
                indep_vars=['intercept'] + list(map(adj_vars.get, indep_vars)),
                n_i_knots=n_i_knots,
                spline_options=spline_options,
                scale_se=True,
                scale_se_floor_pctile=last_days_pctile
            )
            mr_model.fit_model()
            prediction = mr_model.predict(df)
            if not np.isnan(prediction).any():
                prediction_pending = False
            else:
                print(f'Elasticity model failed with {n_i_knots} knots (nans).')
        except:
            print(f'Elasticity model failed with {n_i_knots} knots (error).')
        if n_i_knots == 1:
            prediction_pending = False
        else:
            n_i_knots -= 1
    
    # attach prediction
    df['Predicted model death rate'] = prediction
    df[f'Predicted death rate ({model_type})'] = df['Predicted model death rate']
    if log:
        df[f'Predicted death rate ({model_type})'] = np.exp(df[f'Predicted death rate ({model_type})'])
    if daily:
        df[f'Predicted death rate ({model_type})'] = df[f'Predicted death rate ({model_type})'].cumsum()

    with open(f"{model_dir}/{df['location_id'][0]}_{model_type}.pkl", 'wb') as fwrite:
        pickle.dump(mr_model, fwrite, -1)

    return df
