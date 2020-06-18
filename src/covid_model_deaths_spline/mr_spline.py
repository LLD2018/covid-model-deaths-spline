import numpy as np
import pandas as pd
from mrtool import MRData, LinearCovModel, MRBeRT, utils
from typing import List, Dict


class SplineFit:
    """Spline fit class
    """
    def __init__(self, 
                 data: pd.DataFrame, 
                 dep_var: str,
                 spline_var: str,
                 indep_vars: List[str], 
                 n_i_knots: int,
                 ensemble_knots: np.array = None,
                 spline_options: Dict = dict(),
                 scale_se: bool = True,
                 scale_se_power: float = 0.2,
                 scale_se_floor_pctile: float = 0.05,
                 observed_var: str = None, 
                 pseudo_se_multiplier: float = 1.):
        # set up model data
        data = data.copy()
        if scale_se:
            data['obs_se'] = 1./np.exp(data[dep_var])**scale_se_power
            se_floor = np.percentile(data['obs_se'], scale_se_floor_pctile)
            data.loc[data['obs_se'] < se_floor, 'obs_se'] = se_floor
        else:
            data['obs_se'] = 1
        if observed_var:
            if not data[observed_var].dtype == 'bool':
                raise ValueError(f'Observed variable ({observed_var}) is not boolean.')
            data.loc[~data[observed_var], 'obs_se'] *= pseudo_se_multiplier
        else:
            observed_var = 'observed'
            data[observed_var] = True

        # create mrbrt object
        data['study_id'] = 1
        mr_data = MRData(
            df=data,
            col_obs=dep_var,
            col_obs_se='obs_se',
            col_covs=indep_vars + [spline_var],
            col_study_id='study_id'
        )
        self.data = data
        
        # cov models
        cov_models = []
        if 'intercept' in indep_vars:
            cov_models += [LinearCovModel(
                alt_cov='intercept',
                use_re=True,
                prior_gamma_uniform=np.array([0.0, 0.0]),
                name='intercept'
            )]
        if 'Model testing rate' in indep_vars:
            cov_models += [LinearCovModel(
                alt_cov='Model testing rate',
                use_re=False,
                prior_beta_uniform=np.array([-np.inf, 0.]),
                name='Model testing rate'
            )]
        if any([i not in ['intercept', 'Model testing rate'] for i in indep_vars]):
            bad_vars = [i for i in indep_vars if i not in ['intercept', 'Model testing rate']]
            raise ValueError(f"Unsupported independent variable(s) entered: {'; '.join(bad_vars)}")
            
        # get random knot placement
        if 'spline_knots' in list(spline_options.keys()):
            raise ValueError('Using random spline, do not manually specify knots.')
        if ensemble_knots is None:
            ensemble_knots = self.get_ensemble_knots(n_i_knots, data[spline_var].values, 
                                                     data[observed_var].values, spline_options)
        
        # spline cov model
        spline_model = LinearCovModel(
            alt_cov=spline_var,
            use_re=False,
            use_spline=True,
            **spline_options,
            prior_spline_num_constraint_points=100,
            spline_knots=ensemble_knots[0],
            name=spline_var
        )
        
        # var names
        self.indep_vars = [i for i in indep_vars if i != 'intercept']
        self.spline_var = spline_var
        
        # model
        self.mr_model = MRBeRT(mr_data, 
                               ensemble_cov_model=spline_model, 
                               ensemble_knots=ensemble_knots,
                               cov_models=cov_models)
        self.submodel_fits = None
        self.coef_dicts = None
        
    def get_ensemble_knots(self, n_i_knots: int, spline_data: np.array, observed: np.array, 
                           spline_options: Dict, N: int = 50) -> List[np.array]:
        # sample
        n_intervals = n_i_knots + 1
        k_start = 0.
        k_end = 1.
        if n_i_knots >= 3:
            if np.diff([spline_data.min(), np.quantile(spline_data[observed], 0.05)]) > 1e-10:
                n_intervals -= 1
                k_start = 0.15
            if np.diff([np.quantile(spline_data[observed], 0.95), spline_data.max()]) > 1e-10:
                n_intervals -= 1
                k_end = 0.85
        ensemble_knots = utils.sample_knots(n_intervals, 
                                            b=np.array([[k_start, k_end]]*(n_intervals-1)),
                                            d=np.array([[0.1, 1]]*n_intervals),
                                            N=N)
        if k_start > 0.:
            ensemble_knots = np.insert(ensemble_knots, 1, 0.05, 1)
        if k_end < 1.:
            ensemble_knots = np.insert(ensemble_knots, -1, 0.95, 1)
            
        # rescale to observed
        if (~observed).any():
            if spline_options['spline_knots_type'] != 'domain':
                raise ValueError('Expecting `spline_knots_type` domain for knot rescaling (stage 2 model).')
            ensemble_knots = rescale_k(spline_data[observed], spline_data, ensemble_knots)
        
        # make sure we have unique knots
        _ensemble_knots = []
        for knots in ensemble_knots:
            if np.unique(np.quantile(spline_data, knots)).size == knots.size:
                _ensemble_knots.append(knots)
        ensemble_knots = np.vstack(_ensemble_knots)
        
        return ensemble_knots

    def fit_model(self):
        self.mr_model.fit_model(inner_max_iter=30)
        self.mr_model.score_model()
        self.coef_dicts = [self.get_submodel_coefficients(sm) for sm in self.mr_model.sub_models]

    def get_submodel_coefficients(self, sub_model):
        coef_dict = {}
        for variable in self.indep_vars:
            coef_dict.update({
                variable: sub_model.beta_soln[sub_model.x_vars_idx[variable]]
            })
        spline_coefs = sub_model.beta_soln[sub_model.x_vars_idx[self.spline_var]]
        if 'intercept' in sub_model.linear_cov_model_names:
            intercept_coef = sub_model.beta_soln[sub_model.x_vars_idx['intercept']]
            spline_coefs = np.hstack([intercept_coef, intercept_coef + spline_coefs])
        coef_dict.update({
            self.spline_var:spline_coefs
        })
        
        return coef_dict
    
    def predict(self, pred_data: pd.DataFrame):
        # get individual curves
        submodel_fits = [self.predict_submodel(sub_model, coef_dict, pred_data) for sub_model, coef_dict in
                         zip(self.mr_model.sub_models, self.coef_dicts)]
        submodel_fits = np.array(submodel_fits)
        weights = np.array([self.mr_model.weights]).T
            
        return (submodel_fits * weights).sum(axis=0)
        
    def predict_submodel(self, sub_model, coef_dict: dict, pred_data: pd.DataFrame):
        spline_model_idx = sub_model.linear_cov_model_names.index(self.spline_var)
        spline_model = sub_model.linear_cov_models[spline_model_idx]
        spline_model = spline_model.create_spline(self.mr_model.data)
        preds = []
        for variable, coef in coef_dict.items():
            if variable == self.spline_var:
                mat = spline_model.design_mat(pred_data[variable].values,
                                              l_extra=True, r_extra=True)
            else:
                mat = pred_data[[variable]].values
            preds += [mat.dot(coef)]

        return np.sum(preds, axis=0)

    
def rescale_k(x_from: np.array, x_to: np.array, ensemble_knots: np.array) -> np.array:
    ensemble_knots = ensemble_knots.copy()

    def _rescale_k(x1, k, x2):
        k1_n = x1.min() + k * x1.ptp()
        k2 = (k1_n - x2.min()) / x2.ptp()
        return k2

    ensemble_knots = [_rescale_k(x_from, ek, x_to) for ek in ensemble_knots]
    ensemble_knots = np.vstack(ensemble_knots)

    ensemble_knots[:,0] = 0
    ensemble_knots[:,-1] = 1

    return ensemble_knots