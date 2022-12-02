import numpy as np
from tqdm import tqdm
import pickle as pkl
from IPython.display import display
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

from ..data_preparation import get_data_xy_arrays_from_classification_data, get_shuffled_balanced_train_data, fit_imputer_and_get_imputed_data_train_valid_xy, impute_data_with_imputer, standardise_cont_vars_in_data, impute_new_dataset
from ..validation_utils import initialize_cv_results_gridSearch,initialize_cv_results_single_hyperparam, get_complete_inner_cv_results_df, get_best_setting_from_gridSearch_cv_results_df, get_complete_cv_no_gridSearch_results_df, get_summary_best_param_cv_results 
from .classification_model_utils import get_initialized_classification_model

# ################################## Metric Evaluation #############################
def evaluate_single_metric_from_confusion_mat(metric, tn, fp, fn, tp):
    if metric=='Sensitivity':
        try:
            value = tp/(tp+fn)
        except:
            value = None
    elif metric=='Specificity':
        try:
            value = tn/(tn+fp)
        except:
            value = None
    elif metric=='Accuracy':
        try:
            value = (tp+tn)/(tn+fp+fn+tp)
        except:
            value = None
    elif metric=='Precision':
        try:
            value = tp/(tp+fp)
        except:
            value = None
    elif metric=='MAA': #Macro averaged Accuracy
        try:
            value = 0.5*( tp/(tp+fn) + tn/(tn+fp)) 
        except:
            value = None
    elif metric=='G-mean':
        try:
            value = np.sqrt((tp/(tp+fn))* (tn/(tn+fp)))
        except:
            value = None
    else:
        print("Unrecognized metric type: {} to compute from confusion mat!".format(metric))
        value = None
    return value

def evaluate_single_classification_metric(model, metric, standardized_test_data_x, test_data_y):
    if metric in ['Sensitivity', 'Specificity', 'Accuracy', 'Precision', 'MAA', 'G-mean']:
        predicted_y = model.predict(standardized_test_data_x)
        tn, fp, fn, tp = confusion_matrix(y_true=test_data_y, y_pred=predicted_y).ravel()
        return evaluate_single_metric_from_confusion_mat(metric, tn, fp, fn, tp)
    elif metric == 'ROC-AUC':
        return roc_auc_score(y_true=test_data_y, y_score=model.decision_function(standardized_test_data_x))
    else:
        print("Unrecognised metric!", metric)
        return None

def evaluate_binary_classification_metrics_in_cv(model, model_cv_results, metrics, data_x_lst, data_y_lst, dataset_names, this_param_combination=None):
    # Evaluate a model on multiple datasets (data_x_lst) on chosen metrics, and append results to model_cv_results
    assert len(data_x_lst)==len(data_y_lst)==len(dataset_names)
    num_datasets = len(dataset_names)
  
    for idx in range(num_datasets):
        overall_score = 0
        for metric in metrics:
            if metric == 'Overall':
                continue
            else:
                score_for_dataset_metric = evaluate_single_classification_metric(model, metric, data_x_lst[idx], data_y_lst[idx])
                overall_score += score_for_dataset_metric
                try:
                    model_cv_results[this_param_combination][dataset_names[idx]+'_'+metric+'_values'].append(score_for_dataset_metric)
                except:
                    model_cv_results[dataset_names[idx]+'_'+metric+'_values'].append(score_for_dataset_metric)
    
        if 'Overall' in metrics:
            overall_score = overall_score/len(metrics)
            try:
                model_cv_results[this_param_combination][dataset_names[idx]+'_Overall_values'].append(overall_score)
            except:
                model_cv_results[dataset_names[idx]+'_Overall_values'].append(overall_score)

    return model_cv_results


# ################################## Cross Validation #############################
def get_best_model_with_cv_classification(cv, model_name, data_train_valid, cv_metrics, rank_metric, inner_dataset_names, hyperparam_dict_lst, config, rank_dataset_name='valid'):
    ''' Performs cross validation and find the hyperparameter setting that gives the best average performance, then return a model trained with this best hyperparameter setting.'''
    input_variables = config['input_variables'] 
    boolean_variable = config['boolean_variable'] 
    stratify_variable = config['stratify_variable']
    impute_data_choice = config['impute_data_choice']
    all_input_output_variables = input_variables + [boolean_variable]

    # Initialize results dict for inner loop hyperparameter optimization
    cv_results = initialize_cv_results_gridSearch(cv_metrics, inner_dataset_names, hyperparam_dict_lst) # cv_results still a dict.
        
    for train_index, valid_index in tqdm(cv.split(
        data_train_valid.values, 
        data_train_valid[stratify_variable].values), total=cv.get_n_splits(), desc="Inner Loop", leave=False, position=1):
        
        # data_train/valid_cv does not contain stratify variable
        data_train_cv, data_valid_cv = data_train_valid[all_input_output_variables].iloc[train_index], data_train_valid[all_input_output_variables].iloc[valid_index]

        # Do imputation as required
        imputed_data_train_cv_x, data_train_cv_y, imputed_data_valid_cv_x, data_valid_cv_y, imputer, imputer_scaler = fit_imputer_and_get_imputed_data_train_valid_xy(data_train_cv, config, impute_data_valid=True, data_valid=data_valid_cv, is_survival_data=False)

        # Standardise imputed data x
        standardised_imputed_data_train_cv_x, cv_input_scaler = standardise_cont_vars_in_data(imputed_data_train_cv_x, config['cont_var_idx_arr'], fit_scaler=True, provided_scaler=None)

        standardised_imputed_data_valid_cv_x = standardise_cont_vars_in_data(imputed_data_valid_cv_x, config['cont_var_idx_arr'], fit_scaler=False, provided_scaler=cv_input_scaler)

        # Prepare datasets for evaluation
        standardised_imputed_data_cv_x_lst = [standardised_imputed_data_train_cv_x, standardised_imputed_data_valid_cv_x]
        data_cv_y_lst = [data_train_cv_y, data_valid_cv_y]


        # Loop through parameter grid (all combinations of parameters) and get cv results
        for this_param_combination in tqdm(list(cv_results.keys()), desc="Looping through hyperparameter settings", leave=False, position=2):
            # Class balancing
            this_oversampler = cv_results[this_param_combination]['param_oversampler']
            # except:
            #     this_oversampler = cv_results[this_param_combination]['oversampler']

            balanced_standardised_imputed_data_train_cv_x, balanced_data_train_cv_y = get_shuffled_balanced_train_data(standardised_imputed_data_train_cv_x, data_train_cv_y, this_oversampler, config['cat_var_idx_arr'])

            # Build and train a model
            model = get_initialized_classification_model(model_name, cv_results[this_param_combination])
            model.fit(balanced_standardised_imputed_data_train_cv_x, balanced_data_train_cv_y)

            # Evaluate model on validation data
            cv_results = evaluate_binary_classification_metrics_in_cv(model, cv_results, cv_metrics, standardised_imputed_data_cv_x_lst, data_cv_y_lst, inner_dataset_names, this_param_combination)
        ############# Finish looping through all parameter combinations ##########
    
    # Select best hyperparameter combination from cv results
    cv_results_df = get_complete_inner_cv_results_df(cv_results, cv_metrics, rank_dataset_name=rank_dataset_name, rank_metric=rank_metric, rank_criterion='larger_better')
    best_setting_results = get_best_setting_from_gridSearch_cv_results_df(cv_results_df, rank_dataset_name, rank_metric)
    
   
    ### Gather data to retrain a model on the entire dataset with the best hyperparameter setting
    imputed_data_train_valid_x, data_train_valid_y, final_imputer, final_imputer_scaler = fit_imputer_and_get_imputed_data_train_valid_xy(data_train_valid[all_input_output_variables], config, impute_data_valid=False, is_survival_data=False)

    # Standardising data x
    standardised_imputed_data_train_valid_x, final_input_scaler = standardise_cont_vars_in_data(imputed_data_train_valid_x.copy(), config['cont_var_idx_arr'], fit_scaler=True, provided_scaler=None)

    # Class balancing
    balanced_standardised_imputed_data_train_valid_x, balanced_data_train_valid_y = get_shuffled_balanced_train_data(standardised_imputed_data_train_valid_x, data_train_valid_y, best_setting_results['param_oversampler'], config['cat_var_idx_arr'])

    # Retrain model
    best_model = get_initialized_classification_model(model_name, best_setting_results) 
    best_model.fit(balanced_standardised_imputed_data_train_valid_x, balanced_data_train_valid_y)

    return best_model, final_input_scaler, final_imputer, final_imputer_scaler, best_setting_results, standardised_imputed_data_train_valid_x, data_train_valid_y, cv_results

def do_nested_cv_classification(data, model_name, config, return_final_model=True):
    ### Read in config ###
    input_variables = config['input_variables']
    stratify_variable = config['stratify_variable']
    boolean_variable = config['boolean_variable'] 
    cat_var_idx_arr = config['cat_var_idx_arr']

    cv_outer = config['cv_outer']
    cv_inner = config['cv_inner']
    cv_final = config['cv_final']


    outer_cv_results = initialize_cv_results_single_hyperparam(config['outer_cv_metrics'], config['outer_dataset_names'])

    # Outer Loop CV -- testing this pipeline/overall model
    for train_valid_index, test_index in tqdm(cv_outer.split(data.values, data[stratify_variable].values), total=cv_outer.get_n_splits(), desc="Outer Loop", position=0):
        data_train_valid, data_test = data.iloc[train_valid_index], data.iloc[test_index]

        ## Inner Loop CV -- hyperparameter optimization
        best_model_from_inner_cv, final_input_scaler_from_inner_cv, final_imputer_from_inner_cv, final_imputer_scaler_from_inner_cv, best_setting_results_from_inner_cv, standardised_imputed_data_train_valid_x, data_train_valid_y, inner_cv_results = get_best_model_with_cv_classification(cv_inner, model_name, data_train_valid, config['inner_cv_metrics'], config['inner_rank_metric'], config['inner_dataset_names'], config['hyperparam_dict_lst'], config, rank_dataset_name='valid')

        ## Prepare data for outer cv test
        if config['impute_data_choice']:
            imputed_data_test = impute_new_dataset(data_test, config, final_imputer_from_inner_cv, final_imputer_scaler_from_inner_cv, is_survival_data=False)
        else:
            imputed_data_test = data_test

        imputed_data_test_x, data_test_y = get_data_xy_arrays_from_classification_data(imputed_data_test, input_variables, boolean_variable)
        
        # Standardise test data x
        standardised_imputed_data_test_x = standardise_cont_vars_in_data(imputed_data_test_x.copy(), config['cont_var_idx_arr'], fit_scaler=False, provided_scaler=final_input_scaler_from_inner_cv)

        # Store all datasets for evaluation in a list
        outer_standardised_imputed_data_x_lst = [standardised_imputed_data_train_valid_x, standardised_imputed_data_test_x]
        outer_data_y_lst = [data_train_valid_y, data_test_y]
    
        # Test model and add to outer cv results
        outer_cv_results = evaluate_binary_classification_metrics_in_cv(best_model_from_inner_cv, outer_cv_results, config['outer_cv_metrics'], outer_standardised_imputed_data_x_lst, outer_data_y_lst, config['outer_dataset_names'], this_param_combination=None)

        ### Finish outer cv loop ###
        
    outer_cv_results_df = get_complete_cv_no_gridSearch_results_df(outer_cv_results, config['outer_dataset_names'], config['outer_cv_metrics'])
 
    if return_final_model:
        ## Train final model on all data
        final_model, final_input_scaler, final_imputer, final_imputer_scaler, best_setting_results, standardised_imputed_data_x, data_y, final_cv_results = get_best_model_with_cv_classification(cv_final, model_name, data, config['inner_cv_metrics'], config['inner_rank_metric'], config['inner_dataset_names'], config['hyperparam_dict_lst'], config, rank_dataset_name='valid')
          
        return outer_cv_results, outer_cv_results_df, final_model, final_input_scaler, final_imputer, final_imputer_scaler, best_setting_results
    else:
        return outer_cv_results, outer_cv_results_df

def do_single_cv_classification(data, model_name, config, return_final_model=True): 
    '''Use this function to do 1-loop cross validation, when you don't need to search for hyperparameter setting, thus no need for nested cv.'''

    ### Read in config ###
    input_variables = config['input_variables']
    stratify_variable = config['stratify_variable']
    boolean_variable = config['boolean_variable'] 
    hyperparam_dict = config['hyperparam_dict']
    cv_outer = config['cv_outer']

    outer_cv_results = initialize_cv_results_single_hyperparam(config['outer_cv_metrics'], config['outer_dataset_names'])

    for train_valid_index, test_index in tqdm(cv_outer.split(data.values, data[stratify_variable].values), total=cv_outer.get_n_splits(), desc="Outer Loop", position=0):
        data_train_valid, data_test = data.iloc[train_valid_index], data.iloc[test_index]
        
        # Impute data
        imputed_data_train_valid_x, data_train_valid_y, imputed_data_test_x, data_test_y, imputer, imputer_scaler = fit_imputer_and_get_imputed_data_train_valid_xy(data_train_valid, config, impute_data_valid=True, data_valid=data_test, is_survival_data=False)

        # Standardise data x
        standardised_imputed_data_train_valid_x, cv_input_scaler = standardise_cont_vars_in_data(imputed_data_train_valid_x, config['cont_var_idx_arr'], fit_scaler = True, provided_scaler=None)
        standardised_imputed_data_test_x = standardise_cont_vars_in_data(imputed_data_test_x, config['cont_var_idx_arr'], fit_scaler=False, provided_scaler=cv_input_scaler)

        # Prepare datasets for evaluation
        outer_standardised_imputed_data_x_lst = [standardised_imputed_data_train_valid_x, standardised_imputed_data_test_x]
        outer_data_y_lst = [data_train_valid_y, data_test_y]

        # Data balancing
        balanced_standardised_imputed_data_train_valid_x, balanced_data_train_valid_y = get_shuffled_balanced_train_data(standardised_imputed_data_train_valid_x, data_train_valid_y, hyperparam_dict['oversampler'], config['cat_var_idx_arr'])

        # Build model
        model = get_initialized_classification_model(model_name, hyperparam_dict)
        model.fit(balanced_standardised_imputed_data_train_valid_x, balanced_data_train_valid_y)
        
        # Test model pipe on training and validation data
        outer_cv_results = evaluate_binary_classification_metrics_in_cv(model, outer_cv_results, config['outer_cv_metrics'], outer_standardised_imputed_data_x_lst, outer_data_y_lst, config['outer_dataset_names'], this_param_combination=None)


    outer_cv_results_df = get_complete_cv_no_gridSearch_results_df(outer_cv_results, config['outer_dataset_names'], config['outer_cv_metrics'])
 

    if return_final_model:
        # Impute data
        imputed_data_x, data_y, final_imputer, final_imputer_scaler = fit_imputer_and_get_imputed_data_train_valid_xy(data, config, impute_data_valid=False, is_survival_data=False)

        # Standardise data x
        standardised_imputed_data_x, final_input_scaler = standardise_cont_vars_in_data(imputed_data_x, config['cont_var_idx_arr'], fit_scaler=True, provided_scaler=None)

        # Data balancing
        balanced_standardised_imputed_data_x, balanced_data_y = get_shuffled_balanced_train_data(standardised_imputed_data_x, data_y, hyperparam_dict['oversampler'], config['cat_var_idx_arr'])

        # Retrain model
        final_model = get_initialized_classification_model(model_name, hyperparam_dict)
        final_model.fit(balanced_standardised_imputed_data_x, balanced_data_y)
        return outer_cv_results, outer_cv_results_df, final_model, final_input_scaler, final_imputer, final_imputer_scaler
    else:
        return outer_cv_results, outer_cv_results_df

def do_external_validation_classification(model, input_scaler, imputer, imputer_scaler, external_datasets, config, BT_sample_size=None):
    '''This function performs external validation with bootstrapping.'''
    # Read in config
    input_variables = config['input_variables']
    boolean_variable = config['boolean_variable'] 

    external_ds_names = list(external_datasets.keys())

    for external_ds_name in external_ds_names:
        # Impute dataset
        imputed_external_dataset = impute_new_dataset(external_datasets[external_ds_name]['data'], config, imputer, imputer_scaler, is_survival_data=False)
        imputed_data_x, data_y = get_data_xy_arrays_from_classification_data(imputed_external_dataset, input_variables, boolean_variable)
        standardised_imputed_data_x =  standardise_cont_vars_in_data(imputed_data_x, config['cont_var_idx_arr'], fit_scaler=False, provided_scaler=input_scaler)
        external_datasets[external_ds_name]['data_y'] = data_y
        external_datasets[external_ds_name]['standardised_imputed_data_x'] = standardised_imputed_data_x

    external_val_results = initialize_cv_results_single_hyperparam(config['outer_cv_metrics'], external_ds_names)

    # Bootstrapping external datasets
    np.random.seed(0)
    for i in range(config['N_bootstrapping']):
        standardised_imputed_external_data_x_lst = []
        external_data_y_lst = []
        for external_ds_name in external_ds_names:
            standardised_imputed_external_data_x = external_datasets[external_ds_name]['standardised_imputed_data_x']
            external_data_y = external_datasets[external_ds_name]['data_y']
            N = standardised_imputed_external_data_x.shape[0]
            if BT_sample_size == None:
                bootstrapped_indices = np.random.choice(np.arange(N), size=N, replace=True)
            else:
                bootstrapped_indices = np.random.choice(np.arange(N), size=BT_sample_size, replace=True )
            
            standardised_imputed_external_data_x_lst.append(standardised_imputed_external_data_x[bootstrapped_indices])
            external_data_y_lst.append(external_data_y[bootstrapped_indices])

        external_val_results = evaluate_binary_classification_metrics_in_cv(model, external_val_results, config['outer_cv_metrics'], standardised_imputed_external_data_x_lst, external_data_y_lst, external_ds_names, this_param_combination=None)

    external_validation_results_df = get_complete_cv_no_gridSearch_results_df(external_val_results, external_ds_names, config['outer_cv_metrics'])
    
    return external_val_results, external_validation_results_df

if __name__ == '__main__':
    print("Running classification_validation_utils.py")