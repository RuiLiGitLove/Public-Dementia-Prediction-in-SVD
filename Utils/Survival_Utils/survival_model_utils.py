import numpy as np
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest

def get_survival_hyperparam_search_setting(model_name, hyperparam_search):
    if model_name=='CoxPH':
        hyperparam_dict = [{
            'alpha': [0.0]
        }]
    elif model_name=='CoxPH_l2':
        hyperparam_dict = [{
            'alpha': np.linspace(1e-3, 15, 50)#10**np.linspace(-3, 1.5, 50)
        }]
    elif model_name=='CoxPH_l1':
        hyperparam_dict = [{
            'alpha': np.linspace(1e-3, 5, 50)#10**np.linspace(-3, 1.5, 50)
        }]
    elif model_name=='Reg_Cox':
        if hyperparam_search:
            hyperparam_dict = [{
                'alphas': np.logspace(-3,0,10), #np.linspace(0.001, 0.01, 0.001),
                'l1_ratio': [0.2, 0.4, 0.6, 0.8, 1.0]
            }]
        else:
            hyperparam_dict = [{
                'alphas': [0.001],
                'l1_ratio': [0.5]
            }]
    elif model_name=='RSF':
        hyperparam_dict = [
            # {'n_estimators': [int(x) for x in np.arange(20,220,20)] ,
            # 'max_depth': [int(x) for x in np.arange(1,6,1)],
            # 'max_features': ['sqrt']}, #'sqrt is default
            {'n_estimators': [50, 100, 150],
            'max_depth': [5,7,9] ,
            'max_features': ['sqrt']}
        ]
    elif model_name=='GBT':
        hyperparam_dict = [
            {'n_estimators': [150, 175, 200, 225, 250],
            'max_depth': [1],
            'learning_rate': [0.1],
            'max_features': ['sqrt'], #If “auto”, then max_features=n_features.
            'subsample': [0.2]#np.linspace(0.1, 0.5, num=3) #The fraction of samples to be used for fitting the individual regression trees.
            }
        ]
    else:
        print("Unrecognised survival model!", model_name)
        hyperparam_dict = None
    
    return hyperparam_dict

def get_initialized_survival_model(model_name, param_dict):
    if model_name=='CoxPH':
        model = CoxPHSurvivalAnalysis(alpha=0)
    elif model_name=='CoxPH_l2':
        model = CoxPHSurvivalAnalysis(alpha=param_dict['param_alpha'])
    elif model_name=='CoxPH_l1':
        model = CoxnetSurvivalAnalysis(
            l1_ratio=1.0, 
            alphas=[param_dict['param_alphas']], 
            fit_baseline_model=True)
    elif model_name=='Reg_Cox':
        try:
            model = CoxnetSurvivalAnalysis(
                l1_ratio=param_dict['param_l1_ratio'], 
                alphas=[param_dict['param_alphas']], 
                fit_baseline_model=True)
        except:
            model = CoxnetSurvivalAnalysis(
                l1_ratio=param_dict['l1_ratio'], 
                alphas=[param_dict['alphas']], 
                fit_baseline_model=True)
    elif model_name=='GBT':
        model = GradientBoostingSurvivalAnalysis(
            loss='coxph', #default='coxph'
            n_estimators=param_dict['param_n_estimators'],
            max_depth=param_dict['param_max_depth'],
            learning_rate=param_dict['param_learning_rate'],
            max_features=param_dict['param_max_features'], #The number of features to consider when looking for the best split. If “sqrt”, then max_features=sqrt(n_features).
            subsample=param_dict['param_subsample']
            )
    elif model_name=='RSF':
        model = RandomSurvivalForest(
            n_estimators=param_dict['param_n_estimators'],
            max_depth=param_dict['param_max_depth'],
            max_features=param_dict['param_max_features'], #The number of features to consider when looking for the best split. If “sqrt”, then max_features=sqrt(n_features).
            n_jobs=-1, #-1 means using all processors
            max_samples=1.0 #If bootstrap is True (default), the number of samples to draw from X to train each base estimator. If 1.0 or None (default), then draw X.shape[0] samples
            )
    else:
        print("Unrecognised survival model!", model_name)
        model=None
    
    return model

if __name__ == '__main__':
    print("Running survival_model_utils.py")