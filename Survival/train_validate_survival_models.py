import sys
import os
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_DIR)

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)
import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.model_selection import StratifiedKFold, ParameterGrid

import sklearn.neighbors._base
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base

from Utils.data_preparation import read_in_data, get_input_variables, get_feature_set
from Utils.validation_utils import get_summary_best_param_cv_results
from Utils.Survival_Utils.survival_model_utils import get_survival_hyperparam_search_setting
from Utils.Survival_Utils.survival_validation_utils import do_nested_cv_survival, do_single_cv_survival, do_external_validation_survival, get_best_model_with_cv_survival



def train_validate_survival_models(config):
    model_name = config['model_name']

    ############ Specify variables #################
    input_variables_to_print, FS_name, var_description, cat_feature_indices = get_feature_set(config['featureset_name'])
    input_variables = get_input_variables(input_variables_to_print, config['input_variable_type'])
    print("Input variables:", input_variables, len(input_variables), var_description)

    boolean_variable = config['boolean_variable']
    stratify_variable = config['stratify_variable']
    survival_time_variable = config['survival_time_variable']
    all_required_variables = input_variables + [boolean_variable, stratify_variable, survival_time_variable]

    ############ Create save directory #################
    savepath_base = '{}/{}/{}/{}'.format(config['folder_to_save'], config['evaluation_strategy'], model_name, FS_name)

    ########### Define model and get hyperparameters ##############
    print("Model:", model_name)
    hyperparam_dict_lst = get_survival_hyperparam_search_setting(model_name, config['hyperparam_search'])

    for param_dict in hyperparam_dict_lst:
        for param in list(param_dict.keys()):
            print(param, param_dict[param], "N={}".format(len(param_dict[param])))
    n_hyperparam_combinations = len(list(ParameterGrid(hyperparam_dict_lst)))
    print("{} combinations in total.".format(n_hyperparam_combinations))

    ########## Read in cohorts data ########################
    run_dmc_csv_path = '/Users/lirui/Downloads/Cohort_Dementia_Prediction/Cohort_Data/Selected_Data/Data_6.0/augmented_data/augmented_complete_RUN_DMC_503_subjects.csv'
    scans_csv_path = '/Users/lirui/Downloads/Cohort_Dementia_Prediction/Cohort_Data/Selected_Data/Data_6.0/augmented_data/augmented_complete_SCANS_121_subjects.csv'
    harmo_csv_path = '/Users/lirui/Downloads/Cohort_Dementia_Prediction/Cohort_Data/Selected_Data/Data_6.0/augmented_data/augmented_complete_HARMONISATION_265_subjects.csv'

    

    # TODO: Check if this is what you want!
    if config['impute_data_choice'] == True:
        vars_to_dropNA_for_cohorts = {
            'RUN DMC': [boolean_variable, survival_time_variable, 'PSMD'],
            'SCANS':[boolean_variable, survival_time_variable],
            'HARMO':[boolean_variable, survival_time_variable, 'SVDp']
        }
    else:
        if config['featureset_name'] in ['Multi_BMI', 'Multi_Stroke', 'Multi_BMI_Stroke']:
            multi_input_variables_to_print, multi_FS_name, multi_var_description, multi_cat_feature_indices = get_feature_set('Multi_BMI_Stroke')
            all_multimodal_variables = get_input_variables(multi_input_variables_to_print, config['input_variable_type'])
        else:
            multi_input_variables_to_print, multi_FS_name, multi_var_description, multi_cat_feature_indices = get_feature_set('Multi')
            all_multimodal_variables = get_input_variables(multi_input_variables_to_print, config['input_variable_type'])

        vars_to_dropNA_for_cohorts = {
            'RUN DMC': [boolean_variable, survival_time_variable]+all_multimodal_variables,
            'SCANS':[boolean_variable, survival_time_variable]+all_multimodal_variables,
            'HARMO':[boolean_variable, survival_time_variable]+all_multimodal_variables
        }

    # run_dmc_data contains columns input_variables+[boolean_variable, stratify_variable, survival_time_variable]
    print("RUN DMC:")
    run_dmc_data_all_columns, run_dmc_data= read_in_data(run_dmc_csv_path, all_required_variables, vars_to_dropNA=vars_to_dropNA_for_cohorts['RUN DMC'], get_data_xy=False, survival_data=True, survival_time_variable=survival_time_variable, print_size=True)
    print("SCANS:")
    scans_data_all_columns, scans_data = read_in_data(scans_csv_path, all_required_variables, vars_to_dropNA=vars_to_dropNA_for_cohorts['SCANS'], get_data_xy=False, survival_data=True, survival_time_variable=survival_time_variable, print_size=True)

    if config['evaluation_strategy'] != 'RUN_DMC_SCANS_only':
        print("HARMO:")
        harmo_data_all_columns, harmo_data = read_in_data(harmo_csv_path, all_required_variables, vars_to_dropNA=vars_to_dropNA_for_cohorts['HARMO'], get_data_xy=False, survival_data=True, survival_time_variable=survival_time_variable, print_size=True)

    ############## Organise internal and external datasets #############
    # Specify internal data 
    if config['evaluation_strategy'] == 'all_pooled':
        data = pd.concat([run_dmc_data, scans_data, harmo_data])
    else:
        data = pd.concat([run_dmc_data, scans_data])
        

    data = (data.sample(frac=1, random_state=0)).reset_index(drop=True)
    assert max(data[boolean_variable])==1
    inner_dataset_names = ['train', 'valid']
    outer_dataset_names = ['train_valid', 'test']# Must be set to ['train_valid', 'test']. They are the dataset names for outer loop in nested CV for internal validation
    
    
    if config['evaluation_strategy'] == 'external_harmo':
        external_datasets = {
            'harmo': {'data': harmo_data}
        }
        external_dataset_names = list(external_datasets.keys())

    elif config['evaluation_strategy'] == 'external_harmo_split':
        # Split HARMONISATION into high and low SVD
        external_datasets = {
            'harmo': {'data': harmo_data},
            'harmo_high_SVD': {'data': harmo_data_all_columns.loc[harmo_data_all_columns['SVDp']>0.239][all_required_variables]},
            'harmo_low_SVD': {'data': harmo_data_all_columns.loc[harmo_data_all_columns['SVDp']<=0.239][all_required_variables]}
        }
        external_dataset_names = list(external_datasets.keys())

    elif config['evaluation_strategy'] in ['RUN_DMC_SCANS_only', 'all_pooled']:
        external_datasets = {}
        external_dataset_names = []
    
    else:
        print("Unrecognised evaluation strategy!")
    all_dataset_names = outer_dataset_names+external_dataset_names

    ######################### Define Metric Criteria ########################
    cv_metrics_rank_criteria = {
        "Harrell_C": "larger_better", 
        "Uno_C": "larger_better", 
        "IBS": "smaller_better"
    }
    ############# Save things needed into config ####################
    config['input_variables'] = input_variables
    config['input_variables_to_print'] = input_variables_to_print
    config['cat_var_idx_arr'] = np.array(cat_feature_indices)
    config['cont_var_idx_arr'] = np.array([i for i in np.arange(len(input_variables)) if i not in cat_feature_indices])
    config['hyperparam_dict_lst'] = hyperparam_dict_lst
    config['inner_dataset_names'] = inner_dataset_names
    config['cv_metrics_rank_criteria'] = cv_metrics_rank_criteria
    config['outer_dataset_names'] = outer_dataset_names

    ############################## Hyperparameter Optimization #########################################
    # opt_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=6)

    # best_model_from_cv, final_input_scaler_from_cv, final_imputer_from_cv, final_imputer_scaler_from_cv, best_setting_results_from_cv, standardized_imputed_data_train_valid_x, data_train_valid_y, cv_results = get_best_model_with_cv_survival(opt_cv, model_name, data, config['outer_cv_metrics'], config['inner_rank_metric'], config['inner_dataset_names'], hyperparam_dict_lst, config, rank_dataset_name='valid')

    # filepath = '{}/{}/{}/{}_hyperparam_optimization_results.pkl'.format(config['folder_to_save'], config['evaluation_strategy'], model_name, config['featureset_name'])
    # pkl.dump(cv_results,  open(filepath, "wb" ))
    ############################# Nested CV #########################################
    n_splits=5
    cv_outer = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0) #if random state is set, the indices will be deterministic

    if config['hyperparam_search'] and n_hyperparam_combinations>1:
        print("Doing nested CV...")
        #-------------- Specify CV ---------------#
        cv_inner = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)
        cv_final = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2)

        config['cv_outer'] = cv_outer
        config['cv_inner'] = cv_inner
        config['cv_final'] = cv_final
        #-----------------------------------------#

        #outer_cv_results, outer_cv_results_df, final_model, final_input_scaler, final_imputer, final_imputer_scaler, best_setting_results, data_y = do_nested_cv_survival(data, model_name, config, return_final_model=True)

        # Ablation Study
        outer_cv_results, outer_cv_results_df = do_nested_cv_survival(data, model_name, config, return_final_model=False)

    else:
        print("Only 1 hyperparameter setting. Doing unnested k-fold cross validation...")
        hyperparam_dict = list(ParameterGrid(hyperparam_dict_lst))[0] # Must be in the type of dict
        config['hyperparam_dict'] = hyperparam_dict
        config['cv_outer'] = cv_outer

        outer_cv_results, outer_cv_results_df, final_model, final_input_scaler, final_imputer, final_imputer_scaler, data_y = do_single_cv_survival(data, model_name, config, return_final_model=True)
            
    #################### External Validation -- Train a final model and test on external cohorts ################################
    if config['evaluation_strategy'] in ['RUN_DMC_SCANS_only', 'all_pooled']:
        print('No external validation.')
        full_validation_results = outer_cv_results_df
        full_validation_results_summary_df = get_summary_best_param_cv_results(full_validation_results, model_name, config['outer_cv_metrics'], all_dataset_names)
    else:
        print("Doing external validation...")
        external_val_results, external_validation_results_df = do_external_validation_survival(final_model, final_input_scaler, final_imputer, final_imputer_scaler, external_datasets, data_y, config, BT_sample_size=None)

        full_validation_results = pd.concat([outer_cv_results_df, external_validation_results_df], axis=1)
        full_validation_results_summary_df = get_summary_best_param_cv_results(full_validation_results, model_name, config['outer_cv_metrics'], all_dataset_names)    

    print('Done')
    print('------------------------------------------')

    ####################### Save Results ######################
    try:
        os.mkdir(savepath_base)
    except:
        os.rmdir(savepath_base)
        os.mkdir(savepath_base)

    with open(savepath_base+'/{}_{}.pkl'.format(model_name, FS_name), 'wb') as f:
        pkl.dump({
            'full_validation_results': full_validation_results, 
            'final_model': final_model,
            'final_input_scaler': final_input_scaler,
            'final_imputer': final_imputer,
            'final_imputer_scaler': final_imputer_scaler,
            'config': config}, 
            f)

    full_validation_results_summary_df.to_csv(savepath_base+'/{}_{}_summary.csv'.format(model_name, FS_name), index=True)

    return None

######################################### Main Program  ###########################################################

if __name__ == "__main__":
    model_name = input('Model name [CoxPH/Reg_Cox/RSF/GBT]:')
    hyperparam_search = (input('Do you want to search through hyperparameters, i.e. use nested CV? [Y/N]') == 'Y')
    folder_to_save = input("Path to folder for saving results: ([input]/eval_strategy/models)")
    evaluation_strategies = {
        '1': 'external_harmo',
        '2': 'external_harmo_split', # This includes validating on the whole of harmo as well.
        '3': 'RUN_DMC_SCANS_only',
        '4': 'all_pooled'
    }
    evaluation_strategy = evaluation_strategies[input("Evaluation strategy (1: external_harmo; 2: external_harmo_split; 3: RUN_DMC_SCANS_only; 4: all_pooled):")]

    impute_data_choice = (input('Do you want to impute missing data with KNN? [Y/N]') == 'Y')
    if impute_data_choice:
        impute_with_outcome = (input('Do you want to impute with outcome? [Y/N]') == 'Y')
    else:
        imputer_type = None
        impute_with_outcome = False

    # Ablation study
    # multi_variables = ['WMLL', 'Lacunes', 'CMB_bin', 'TBV', 'PSMD', 
    #                     'Global_cog', 'EF', 'PS', 
    #                     'Age', 'Edu_yrs', 'Sex', 'HTN', 'HC', 'Diabetes', 'Smoking']
    
    #for featureset_name in ['No_'+i for i in multi_variables]:
    for featureset_name in ['Multi', 'Image_Demo', 'Cog_Demo', 'Demo']:
        print('Feature set:', featureset_name)
        config = {
            'model_name': model_name,
            'featureset_name' : featureset_name,
            'impute_data_choice': impute_data_choice, # True or False
            'imputer_type': imputer_type,
            'impute_with_outcome': impute_with_outcome,

            'hyperparam_search': hyperparam_search,
            'input_variable_type' : 'transformed',
            'evaluation_strategy' : evaluation_strategy,
            'boolean_variable' : 'dementia_final',
            'stratify_variable' : 'cohort_dementia_final',
            'survival_time_variable' : 'T_survival',
            'folder_to_save': folder_to_save,

            'inner_cv_metrics' : ['IBS'],
            'inner_rank_metric' : "IBS",
            'outer_cv_metrics' : ["Harrell_C", "Uno_C", "IBS"],
            'N_bootstrapping' : 100
        }

        train_validate_survival_models(config)