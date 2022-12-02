#%% Import libraries
# Append the repository path to os.path.
import sys
import os
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_DIR)

import pickle as pkl
import numpy as np
import pandas as pd
import warnings
warnings.simplefilter("ignore", UserWarning)
# from IPython.display import display
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, ParameterGrid

from Utils.data_preparation import read_in_data, get_input_variables, get_feature_set
from Utils.validation_utils import initialize_cv_results_single_hyperparam, get_summary_best_param_cv_results
from Utils.Classification_Utils.classification_model_utils import get_classification_hyperparam_search_setting, get_LVQ_prototypes_in_dataframe
from Utils.Classification_Utils.classification_validation_utils import do_nested_cv_classification, do_single_cv_classification, do_external_validation_classification, get_best_model_with_cv_classification
from Utils.plot_utils import plot_diag_lambda_mat, plot_full_relevance_matrix



def train_validate_classification_models(config):
    model_name = config['model_name']

    ############ Specify variables #################
    input_variables_to_print, FS_name, var_description, cat_feature_indices = get_feature_set(config['featureset_name'])
    input_variables = get_input_variables(input_variables_to_print, config['input_variable_type'])
    print("Input variables:", input_variables, len(input_variables), var_description)
    boolean_variable = config['boolean_variable']
    stratify_variable = config['stratify_variable']
    all_required_variables = input_variables + [boolean_variable, stratify_variable]

    ############ Create save directory #################
    savepath_base = '{}/{}/{}/{}'.format(config['folder_to_save'], config['evaluation_strategy'], model_name, FS_name)
    
    ############ Define model and get hypersettings ############
    print("Model:", model_name)
    hyperparam_dict_lst = get_classification_hyperparam_search_setting(model_name, config['hyperparam_search'], config['oversampler_lst']) 
    for param_dict in hyperparam_dict_lst:
        for param in list(param_dict.keys()):
            print(param, param_dict[param], "N={}".format(len(param_dict[param])))
    n_hyperparam_combinations = len(list(ParameterGrid(hyperparam_dict_lst)))
    print("{} combinations in total.".format(n_hyperparam_combinations))

    ############ Read in cohorts data ############################
    run_dmc_csv_path = '/Users/lirui/Downloads/Cohort_Dementia_Prediction/Cohort_Data/Selected_Data/Data_6.0/augmented_data/augmented_complete_RUN_DMC_503_subjects.csv'
    scans_csv_path = '/Users/lirui/Downloads/Cohort_Dementia_Prediction/Cohort_Data/Selected_Data/Data_6.0/augmented_data/augmented_complete_SCANS_121_subjects.csv'
    harmo_csv_path = '/Users/lirui/Downloads/Cohort_Dementia_Prediction/Cohort_Data/Selected_Data/Data_6.0/augmented_data/augmented_complete_HARMONISATION_265_subjects.csv'


    ################ Prepare to read in cohort data #####################
    # TODO: Check if this is what you want!
    if config['impute_data_choice'] == True:
        vars_to_dropNA_for_cohorts = {
            'RUN DMC': [boolean_variable, 'PSMD'],
            'SCANS':[boolean_variable],
            'HARMO':[boolean_variable, 'SVDp']
        }
    else:
        multi_input_variables_to_print, multi_FS_name, multi_var_description, multi_cat_feature_indices = get_feature_set('Multi')
        all_multimodal_variables = get_input_variables(multi_input_variables_to_print, config['input_variable_type'])

        vars_to_dropNA_for_cohorts = {
            'RUN DMC': [boolean_variable] + all_multimodal_variables,
            'SCANS':[boolean_variable] + all_multimodal_variables,
            'HARMO':[boolean_variable] + all_multimodal_variables
        }

    # run_dmc_data contains columns input_variables+[boolean_variable, stratify_variable, survival_time_variable]
    print("RUN DMC:")
    run_dmc_data_all_columns, run_dmc_data= read_in_data(run_dmc_csv_path, all_required_variables, vars_to_dropNA=vars_to_dropNA_for_cohorts['RUN DMC'], get_data_xy=False, survival_data=False, print_size=True)
    print("SCANS:")
    scans_data_all_columns, scans_data = read_in_data(scans_csv_path, all_required_variables, vars_to_dropNA=vars_to_dropNA_for_cohorts['SCANS'], get_data_xy=False, survival_data=False, print_size=True)

    if config['evaluation_strategy'] != 'RUN_DMC_SCANS_only':
        print("HARMO:")
        harmo_data_all_columns, harmo_data = read_in_data(harmo_csv_path, all_required_variables, vars_to_dropNA=vars_to_dropNA_for_cohorts['HARMO'], get_data_xy=False, survival_data=False, print_size=True)
    

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

    ############# Save things needed into config ####################
    config['input_variables'] = input_variables
    config['input_variables_to_print'] = input_variables_to_print
    config['cat_var_idx_arr'] = np.array(cat_feature_indices)
    config['cont_var_idx_arr'] = np.array([i for i in np.arange(len(input_variables)) if i not in cat_feature_indices])
    config['hyperparam_dict_lst'] = hyperparam_dict_lst
    config['inner_dataset_names'] = inner_dataset_names
    config['outer_dataset_names'] = outer_dataset_names

    #%% ######### Hyperparameter Optimization ######################
    # opt_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    # best_model_from_cv, final_input_scaler_from_cv, final_imputer_from_cv, final_imputer_scaler_from_cv, best_setting_results_from_cv, imputed_data_train_valid_x, data_train_valid_y, cv_results = get_best_model_with_cv_classification(opt_cv, model_name, data, config['outer_cv_metrics'], config['inner_rank_metric'], config['inner_dataset_names'], config['hyperparam_dict_lst'], config, rank_dataset_name='valid')

    # filepath = savepath_base+'/hyperparam_optimization_results.pkl'
    # pkl.dump(cv_results,  open(filepath, "wb" ))
    #%% ############################# Nested CV #########################################
    n_splits=5
    cv_outer = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    if config['hyperparam_search'] and n_hyperparam_combinations>1:
        print("Doing nested CV...")
        #-------------- Specify CV ---------------#
        cv_inner = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)
        cv_final = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2)

        config['cv_outer'] = cv_outer
        config['cv_inner'] = cv_inner
        config['cv_final'] = cv_final
        #-----------------------------------------#

        #outer_cv_results, outer_cv_results_df, final_model, final_input_scaler, final_imputer, final_imputer_scaler, best_setting_results = do_nested_cv_classification(data, model_name, config, return_final_model=True)
        # Ablation Study
        outer_cv_results, outer_cv_results_df = do_nested_cv_classification(data, model_name, config, return_final_model=False)

    else:
        print("Only 1 hyperparameter setting. Doing unnested k-fold cross validation.")
        hyperparam_dict = list(ParameterGrid(hyperparam_dict_lst))[0] # Must be in the type of dict
        config['hyperparam_dict'] = hyperparam_dict
        config['cv_outer'] = cv_outer

        outer_cv_results, outer_cv_results_df, final_model, final_input_scaler, final_imputer, final_imputer_scaler = do_single_cv_classification(data, model_name, config, return_final_model=True)

    #%% ########## External Validation -- Train a final model and test on external cohorts ############
    if config['evaluation_strategy'] in ['RUN_DMC_SCANS_only', 'all_pooled']:
        print('No external validation.')
        full_validation_results = outer_cv_results_df
        full_validation_results_summary_df = get_summary_best_param_cv_results(full_validation_results, model_name, config['outer_cv_metrics'], all_dataset_names)
    else:
        print("Doing external validation...")
        external_val_results, external_validation_results_df = do_external_validation_classification(final_model, final_input_scaler, final_imputer, final_imputer_scaler, external_datasets, config, BT_sample_size=None)

        full_validation_results = pd.concat([outer_cv_results_df, external_validation_results_df], axis=1)
        full_validation_results_summary_df = get_summary_best_param_cv_results(full_validation_results, model_name, config['outer_cv_metrics'], all_dataset_names)
    
    print('Done')
    print('------------------------------------------')

    ####################### Save Results ######################
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

    ## Plot feature importance from LVQ models
    if model_name == 'GMLVQ':
        # Plot diagonal elements of the relevance matrix
        plot_diag_lambda_mat(final_model.lambda_, input_variables_to_print, model_name, savepath=savepath_base+'/diag_lambda.png', vertical_plot=False)

        # Plot full relevance matrix
        plot_full_relevance_matrix(final_model.lambda_, input_variables_to_print, savepath=savepath_base+'/full_relevance_matrix.png')
        
        # Save prototypes
        prototypes_df = get_LVQ_prototypes_in_dataframe(final_model.prototypes_, final_model.prototypes_labels_, final_input_scaler, input_variables, boolean_variable, config['cont_var_idx_arr'])
        prototypes_df.to_csv(savepath_base+"/{}_prototypes.csv".format(model_name), index=False)

    elif model_name == 'GRLVQ':
        # Plot diagonal elements of the relevance matrix
        plot_diag_lambda_mat(final_model.lambda_, input_variables_to_print, model_name, savepath=savepath_base+'/diag_lambda.png', vertical_plot=False)

        # Save prototypes
        prototypes_df = get_LVQ_prototypes_in_dataframe(final_model.w_, final_model.c_w_, final_input_scaler, input_variables, boolean_variable, config['cont_var_idx_arr'])
        prototypes_df.to_csv(savepath_base+"/{}_prototypes.csv".format(model_name), index=False)


if __name__ == "__main__":
    model_name = input('Model name [GMLVQ/GRLVQ/SVM/Reg_Logistic/Logistic]:')
    hyperparam_search = (input('Do you want to search through hyperparameters, i.e. use nested CV? [Y/N]') == 'Y')
    folder_to_save = input("Path to folder for saving results: ([input]/eval_strategy/models)")
    evaluation_strategies = {
        '1': 'external_harmo',
        '2': 'external_harmo_split', # This includes validating on the whole of harmo as well.
        '3': 'RUN_DMC_SCANS_only',
        '4': 'all_pooled'
    }
    evaluation_strategy = evaluation_strategies[input("Evaluation strategy (1: external_harmo; 2: external_harmo_split; 3: RUN_DMC_SCANS_only; 4: all_pooled):")] # Always used option 4 in the end.
    
    # No imputation was done in the end.
    impute_data_choice = (input('Do you want to impute missing data with KNN? [Y/N]') == 'Y')
    if impute_data_choice:
        impute_with_outcome = (input('Do you want to impute with outcome? [Y/N]') == 'Y')
    else:
        imputer_type = None
        impute_with_outcome = False

    # Get list of oversamplers
    # if len(cat_feature_indices)==0: # all numerical
    #     oversampler_lst = ['None', 'RandomOverSampler', 'SMOTE', 'SVMSMOTE', 'ADASYN']
    # else: # numerical with categorical
    #     oversampler_lst = ['None', 'RandomOverSampler', 'SMOTENC']
    oversampler_lst = ['RandomOverSampler']
    
    # Ablation Study
    # multi_variables = ['WMLL', 'Lacunes', 'CMB_bin', 'TBV', 'PSMD', 
    #                     'Global_cog', 'EF', 'PS', 
    #                     'Age', 'Edu_yrs', 'Sex', 'HTN', 'HC', 'Diabetes', 'Smoking']
    
    # for featureset_name in ['No_'+i for i in multi_variables]: 
    for featureset_name in ['Multi', 'Image_Demo', 'Cog_Demo', 'Demo']:
        print('Feature set:', featureset_name)
        config = {
            'model_name': model_name,
            'featureset_name' : featureset_name,
            'impute_data_choice': impute_data_choice, # True or False
            'imputer_type': 'KNN',
            'impute_with_outcome': impute_with_outcome,

            'hyperparam_search': hyperparam_search,
            'oversampler_lst': oversampler_lst,
            'input_variable_type' : 'transformed',
            'evaluation_strategy' : evaluation_strategy,
            'boolean_variable' : 'dementia_3yr',
            'stratify_variable' : 'cohort_dementia_3yr',
            'folder_to_save': folder_to_save,

            'inner_cv_metrics' : ['ROC-AUC'],
            'inner_rank_metric' : 'ROC-AUC',
            'outer_cv_metrics' : ['ROC-AUC', 'G-mean', 'Precision', 'Sensitivity', 'Specificity', 'MAA', 'Accuracy'],
            'N_bootstrapping' : 100
        }
    

        train_validate_classification_models(config)






    
  


