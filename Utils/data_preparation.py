from pyexpat import features
import sys
import math
from xmlrpc.client import boolean
import numpy as np
import pandas as pd
import pickle
import matplotlib
import matplotlib.pyplot as plt
from sksurv.util import Surv
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from imblearn.over_sampling import RandomOverSampler, SMOTENC, SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE, KMeansSMOTE

import sklearn.neighbors._base
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
from missingpy import MissForest
from sklearn.impute import KNNImputer

#*************************************************************************************************************************************************************
### Functions for reading in data ###

def read_in_data(csv_path, all_vars, vars_to_dropNA=None, get_data_xy=False, input_variable_names=None, boolean_variable_name=None, survival_data=False, survival_time_variable=None, print_size=True, index_col=None): # vars_to_dropNA needs to be a list
    # Simplest version is to just read in the data with all required variables
    data_all_columns = pd.read_csv(csv_path, header=0, index_col=index_col)
    if vars_to_dropNA != None:
        data_all_columns = data_all_columns.dropna(axis='index', how='any', subset=vars_to_dropNA)
    
    data = data_all_columns[all_vars]
    
    if get_data_xy:
        if survival_data:
            data_x = data[input_variable_names]
            data_y = Surv().from_dataframe(boolean_variable_name, survival_time_variable, data)
        else:
            data_x, data_y = data[input_variable_names], data[boolean_variable_name] 

        return data_all_columns, data, data_x, data_y #, data_train, data_train_x, data_train_y, data_test, data_test_x, data_test_y
    else:
        if print_size:
            print("N = {}".format(data.shape[0]))
        return data_all_columns, data

def get_input_variables(required_variable_full_names, var_type):
    # var_type='original' or 'transformed'

    input_feature_name_dict = {
        'WMLL':{'transformed': 'Trans_SVDp', 'original': 'SVDp'}, # *** transformed
        'WMH':{'transformed': 'WMH_vol_ml', 'original': 'WMH_vol_ml'}, 
        'Lacunes':{'transformed': 'num_lacunes', 'original': 'num_lacunes'},
        'CMB':{'transformed': 'Trans_num_mb', 'original': 'num_mb'}, # *** transformed
        'CMB_bin':{'transformed': 'mb_bin', 'original': 'mb_bin'},
        'TBV':{'transformed': 'TBV_ml', 'original': 'TBV_ml'},
        'WM':{'transformed': 'WM_vol_ml', 'original': 'WM_vol_ml'},
        'GM':{'transformed': 'GM_vol_ml', 'original': 'GM_vol_ml'},
        'GMF':{'transformed': 'GMF', 'original': 'GMF'},
        'PSMD':{'transformed': 'Trans_PSMD', 'original': 'PSMD'}, # *** transformed
        'Global_cog':{'transformed': 'global_cog', 'original': 'global_cog'},
        'EF':{'transformed': 'EF', 'original': 'EF'},
        'PS':{'transformed': 'PS', 'original': 'PS'},

        'Age':{'transformed': 'age', 'original': 'age'},
        'Edu_yrs':{'transformed': 'edu_yrs', 'original': 'edu_yrs'},
        'Sex':{'transformed': 'sex', 'original': 'sex'},
        'HTN':{'transformed': 'HTN', 'original': 'HTN'},
        'HC':{'transformed': 'HC', 'original': 'HC'},
        'Diabetes':{'transformed': 'diabetes', 'original': 'diabetes'},
        'Smoking':{'transformed': 'smoking', 'original': 'smoking'},

        'BMI':{'transformed': 'BMI', 'original': 'BMI'},
        'stroke_history': {'transformed': 'stroke_history', 'original': 'stroke_history'}
    }

    if var_type == 'trans_psmd_only':
        input_variable_names = []
        for var in required_variable_full_names:
            if var == 'PSMD':
                input_variable_names.append(input_feature_name_dict[var]['transformed'])
            else:
                input_variable_names.append(input_feature_name_dict[var]['original']) 
    elif var_type in ['transformed', 'original']:
        input_variable_names = [input_feature_name_dict[var][var_type] for var in required_variable_full_names]
    else:
        print('Unrecognised variable type!')
        input_variable_names = None
    return input_variable_names

def get_data_xy_arrays_from_classification_data(data_df, input_variable_names, output_variable_name): # To get classification data_x and data_y
    data_x = data_df[input_variable_names].to_numpy()
    data_y = data_df[output_variable_name].to_numpy()
    assert data_x.shape[0]==data_y.shape[0]
    assert data_x.shape[1]==len(input_variable_names)
    return data_x, data_y

def get_data_xy_arrays_from_survival_data(data, input_variable_names, boolean_variable_name, survival_time_name): # To get survival data_x and data_y
    data_x = data[input_variable_names]
    data_y = Surv().from_dataframe(boolean_variable_name, survival_time_name, data)
    # Sanity check
    assert data_x.shape[0]==data.shape[0]==data_y.shape[0]
    assert data_x.shape[1]==len(input_variable_names)
    return data_x, data_y

def get_shuffled_balanced_train_data(data_train_x, data_train_y, oversampler='RandomOverSampler', cat_feature_indices=None): 
    # By oversampling the minority class
    # train_indices is an array.
    # Both data_x and data_y should be arrays.
    data_train_x = np.array(data_train_x)
    data_train_y = np.array(data_train_y)
    
    assert data_train_x.shape[0] == len(data_train_y)
    # Note that you can only use RandomOverSampler or SMOTENC if you have categorical features in your featureset.
    if oversampler == 'RandomOverSampler':
        balanced_data_train_x, balanced_data_train_y = RandomOverSampler(sampling_strategy="minority", random_state=6).fit_resample(data_train_x, data_train_y)
    elif oversampler == 'SMOTENC':
        balanced_data_train_x, balanced_data_train_y = SMOTENC(categorical_features=cat_feature_indices, random_state=6).fit_resample(data_train_x, data_train_y)
    elif oversampler == 'SMOTE':
        balanced_data_train_x, balanced_data_train_y = SMOTE(random_state=6).fit_resample(data_train_x, data_train_y)
    elif oversampler == 'BorderlineSMOTE':
        balanced_data_train_x, balanced_data_train_y = BorderlineSMOTE(random_state=6).fit_resample(data_train_x, data_train_y)
    elif oversampler == 'SVMSMOTE':
        balanced_data_train_x, balanced_data_train_y = SVMSMOTE(random_state=6).fit_resample(data_train_x, data_train_y)
    elif oversampler == 'KMeansSMOTE':
        balanced_data_train_x, balanced_data_train_y = KMeansSMOTE(random_state=6).fit_resample(data_train_x, data_train_y)
    elif oversampler == 'ADASYN':
        balanced_data_train_x, balanced_data_train_y = ADASYN(random_state=6).fit_resample(data_train_x, data_train_y)
    elif oversampler == 'None':
        balanced_data_train_x, balanced_data_train_y = data_train_x, data_train_y
    else:
        print("Unrecognised oversampling strategy. No oversampling is done.")
        balanced_data_train_x, balanced_data_train_y = data_train_x, data_train_y
        
    p = np.random.RandomState(seed=0).permutation(balanced_data_train_x.shape[0])
    return balanced_data_train_x[p], balanced_data_train_y[p] # shuffle

def calculate_feature(operation, features, patient_data, NA_features, contains_NA, desired_feature):
        # Check if all features are available, i.e. not NA
        for thisFeature in features:
            if pd.isna(patient_data[thisFeature]):
                contains_NA=True
                NA_features.append(thisFeature)
                feature_value = None
        
        if contains_NA:
            NA_features.insert(0, desired_feature)
            return None, contains_NA, NA_features
        else:
            if operation=='sum':  #sum up all features
                feature_value = 0
                for thisFeature in features:
                    feature_value += patient_data[thisFeature]
            
            elif operation=='divSumPerc':  #feature 1 * 100/ sum(feature 2 onwards)
                feature_value = 0
                for i in range(1,len(features)): #sum up from feature 2 onwards
                    thisFeature = features[i]
                    feature_value += patient_data[thisFeature]

                first_feature_value = patient_data[features[0]]
                feature_value = first_feature_value*100/feature_value
            elif operation=='average': #average all features   
                feature_value = 0
                for thisFeature in features:
                    feature_value += patient_data[thisFeature]
                feature_value = feature_value/len(features)
            else:
                print("Unrecognisable operation type for feature {}!".format(desired_feature))
            return feature_value, contains_NA, NA_features

def adjust_skewness(array, feature):
    if feature == 'SVDp':
        return np.log(array+0.005), True
    # elif feature == 'WMH_vol_ml':
    #     return np.log(array+0.1)
    elif feature == 'num_mb':
        return np.log(array+1), True
    # elif feature in ['MMSE','MOCA']:
    #     return np.power(array,2)
    elif feature == 'PSMD':
        return np.log(array), True
    else:
        # Do not need adjust skewness
        return array, False

def adjust_skewness_for_dataframe(data):
    features = list(data.columns)
    for feature in features:
        transformed_values, transformed = adjust_skewness(data[feature], feature)
        if transformed:
            data['Trans_'+feature] = transformed_values
    return data

def get_cat_feature_indices(input_variables_to_print):
    cat_features = ['CMB_bin', 'Sex', 'HTN', 'HC', 'Diabetes', 'Smoking']
    cat_feature_indices = []
    for i in np.arange(len(input_variables_to_print)):
        feature = input_variables_to_print[i]
        if feature in cat_features:
            cat_feature_indices.append(i)
    return cat_feature_indices

    
def get_feature_set(featureset_name):
    # cat_feature_indices must be [] if there is no categorical feature!

    if featureset_name in ['Multimodal', 'Multi']:
        input_variables_to_print = ['WMLL', 'Lacunes', 'CMB_bin', 'TBV', 'PSMD', 
                        'Global_cog', 'EF', 'PS', 
                        'Age', 'Edu_yrs', 'Sex', 'HTN', 'HC', 'Diabetes', 'Smoking']
        FS_name = 'Multi'
        var_description = '{}_var'.format(len(input_variables_to_print))
        cat_feature_indices = get_cat_feature_indices(input_variables_to_print)

    elif featureset_name in ['Multi_BMI']:
        input_variables_to_print = ['WMLL', 'Lacunes', 'CMB_bin', 'TBV', 'PSMD', 
                        'Global_cog', 'EF', 'PS', 
                        'Age', 'Edu_yrs', 'BMI', 'Sex', 'HTN', 'HC', 'Diabetes', 'Smoking']
        FS_name = 'Multi_BMI'
        var_description = '{}_var'.format(len(input_variables_to_print))
        cat_feature_indices = get_cat_feature_indices(input_variables_to_print)
        
    elif featureset_name in ['Multi_Stroke']:
        input_variables_to_print = ['WMLL', 'Lacunes', 'CMB_bin', 'TBV', 'PSMD', 
                        'Global_cog', 'EF', 'PS', 
                        'Age', 'Edu_yrs', 'stroke_history', 'Sex', 'HTN', 'HC', 'Diabetes', 'Smoking']
        FS_name = 'Multi_Stroke'
        var_description = '{}_var'.format(len(input_variables_to_print))
        cat_feature_indices = get_cat_feature_indices(input_variables_to_print)

    elif featureset_name in ['Multi_BMI_Stroke']:
        input_variables_to_print = ['WMLL', 'Lacunes', 'CMB_bin', 'TBV', 'PSMD', 
                        'Global_cog', 'EF', 'PS', 
                        'Age', 'Edu_yrs', 'BMI', 'stroke_history', 'Sex', 'HTN', 'HC', 'Diabetes', 'Smoking']
        FS_name = 'Multi_BMI_Stroke'
        var_description = '{}_var'.format(len(input_variables_to_print))
        cat_feature_indices = get_cat_feature_indices(input_variables_to_print)

    elif featureset_name in ['Imaging and Demographic', 'Image_Demo']:
        input_variables_to_print = ['WMLL', 'Lacunes', 'CMB_bin', 'TBV', 'PSMD', 
                        'Age', 'Edu_yrs', 'Sex', 'HTN', 'HC', 'Diabetes', 'Smoking']
        FS_name = 'Image_Demo'
        var_description = '{}_var'.format(len(input_variables_to_print))
        cat_feature_indices = get_cat_feature_indices(input_variables_to_print)

    elif featureset_name in ['Image_Demo_Num_CMB']:
        input_variables_to_print = ['WMLL', 'Lacunes', 'CMB', 'TBV', 'PSMD', 
                        'Age', 'Edu_yrs', 'Sex', 'HTN', 'HC', 'Diabetes', 'Smoking']
        FS_name = 'Image_Demo_Num_CMB'
        var_description = '{}_var'.format(len(input_variables_to_print))
        cat_feature_indices = get_cat_feature_indices(input_variables_to_print)

    elif featureset_name in ['Image_Demo_no_PSMD']:
        input_variables_to_print = ['WMLL', 'Lacunes', 'CMB_bin', 'TBV', 
                        'Age', 'Edu_yrs', 'Sex', 'HTN', 'HC', 'Diabetes', 'Smoking']
        FS_name = 'Image_Demo_no_PSMD'
        var_description = '{}_var'.format(len(input_variables_to_print))
        cat_feature_indices = get_cat_feature_indices(input_variables_to_print)

    elif featureset_name in ['Cognitive and Demographic', 'Cog_Demo']:
        input_variables_to_print = ['Global_cog', 'EF', 'PS', 
                        'Age', 'Edu_yrs', 'Sex', 'HTN', 'HC', 'Diabetes', 'Smoking']
        FS_name = 'Cog_Demo'
        var_description = '{}_var'.format(len(input_variables_to_print))
        cat_feature_indices = get_cat_feature_indices(input_variables_to_print)

    elif featureset_name in ['Imaging', 'Image']:
        input_variables_to_print = ['WMLL', 'Lacunes', 'CMB_bin', 'TBV', 'PSMD']
        FS_name = 'Image'
        var_description = '{}_var'.format(len(input_variables_to_print))
        cat_feature_indices = get_cat_feature_indices(input_variables_to_print)


    elif featureset_name in ['Cognitive', 'Cog']:
        input_variables_to_print = ['Global_cog', 'EF', 'PS']
        FS_name = 'Cog'
        var_description = '{}_var'.format(len(input_variables_to_print))
        cat_feature_indices = get_cat_feature_indices(input_variables_to_print)

    elif featureset_name in ['Demographic', 'Demo']:
        input_variables_to_print = ['Age', 'Edu_yrs', 'Sex', 'HTN', 'HC', 'Diabetes', 'Smoking']
        FS_name = 'Demo'
        var_description = '{}_var'.format(len(input_variables_to_print))
        cat_feature_indices = get_cat_feature_indices(input_variables_to_print) 
    
    elif featureset_name.startswith("No_"):
        excluded_feature = featureset_name[3:]
        input_variables_to_print = ['WMLL', 'Lacunes', 'CMB_bin', 'TBV', 'PSMD', 
                        'Global_cog', 'EF', 'PS', 
                        'Age', 'Edu_yrs', 'Sex', 'HTN', 'HC', 'Diabetes', 'Smoking']
        input_variables_to_print.remove(excluded_feature)
        FS_name = featureset_name
        var_description = '{}_var'.format(len(input_variables_to_print))
        cat_feature_indices = get_cat_feature_indices(input_variables_to_print)

    else:
        print('Unrecognised feature set name!')
        input_variables_to_print=None
        FS_name=None
        var_description=None
        cat_feature_indices=None

    return input_variables_to_print, FS_name, var_description, cat_feature_indices

#*************************************************************************************************************************************************************
### Functions for imputation ###
def correct_imputed_data_with_cat_var(orig_data, imputed_data, cat_var_idx_arr):
    ''' Make imputed categorical variables integer '''
    if len(cat_var_idx_arr)!=0:
        if type(imputed_data) == pd.core.frame.DataFrame:
            imputed_data.iloc[:,cat_var_idx_arr] = (imputed_data.iloc[:,cat_var_idx_arr]>=0.5).astype(np.int64)
        else:
            imputed_data[:,cat_var_idx_arr] = (imputed_data[:,cat_var_idx_arr]>=0.5).astype(np.int64)
        return imputed_data
    else:
        return imputed_data

def impute_data_with_imputer(data, imputer_type, imputer, imputer_scaler, cat_var_idx_arr):
    cont_var_idx_arr = np.array([i for i in np.arange(data.shape[1]) if i not in cat_var_idx_arr])
    cont_var_standardised_data = standardise_cont_vars_in_data(data, cont_var_idx_arr, fit_scaler=False, provided_scaler=imputer_scaler)

    if imputer_type == 'MF':
        imputed_cont_var_standardised_data = imputer.transform(cont_var_standardised_data)
        imputed_data = imputed_cont_var_standardised_data.copy()
        imputed_data[:, cont_var_idx_arr] = imputer_scaler.inverse_transform(imputed_data[:, cont_var_idx_arr])

    elif imputer_type == 'KNN':
        imputed_cont_var_standardised_data = imputer.transform(cont_var_standardised_data)
        imputed_data = imputed_cont_var_standardised_data.copy()
        imputed_data[:, cont_var_idx_arr] = imputer_scaler.inverse_transform(imputed_data[:, cont_var_idx_arr])
        
    else:
        print('Unrecognised imputer type:', imputer)

    # Make sure that imputed categorical variables are integer 
    imputed_data = correct_imputed_data_with_cat_var(data, imputed_data, cat_var_idx_arr)
    assert data.shape == imputed_data.shape
    return imputed_data # dataframe or array, same type as data


def fit_imputer_and_impute_data(data, cat_var_idx_arr, imputer_type, return_imputed_data=True):
    '''This function fits an imputer to the provided data and imputes it'''
    cont_var_idx_arr = np.array([i for i in np.arange(data.shape[1]) if i not in cat_var_idx_arr])
    cont_var_standardised_data, imputer_scaler = standardise_cont_vars_in_data(data, cont_var_idx_arr, fit_scaler=True, provided_scaler=None)

    if imputer_type == 'MF':
        imputer = MissForest(criterion=('squared_error', 'gini'), max_features='sqrt').fit(cont_var_standardised_data, cat_vars = cat_var_idx_arr)
        
    elif imputer_type == 'KNN':
        imputer = KNNImputer(n_neighbors=5, weights='distance').fit(cont_var_standardised_data)

    else:
        print('Unrecognised imputer type:', imputer)

    if return_imputed_data:
        imputed_data = impute_data_with_imputer(data, imputer_type, imputer, imputer_scaler, cat_var_idx_arr)
    else:
        imputed_data = None
    return imputed_data, imputer, imputer_scaler # Note that the imputer scaler only scales cont variables!


def fit_imputer_and_get_imputed_data_train_valid_xy(data_train, config, impute_data_valid=False, data_valid=None, is_survival_data=True):
    '''This function prepares the training data for fitting an imputer depending on whether the outcome variables are needed, and then imputes the training (and validation) data as required.'''
    '''data_train can contain irrelevant columns such as stratifying variable.'''
    
    input_variables = config['input_variables'] 
    boolean_variable = config['boolean_variable'] 
    if is_survival_data:
        survival_time_variable = config['survival_time_variable']
        all_variables = input_variables+[boolean_variable, survival_time_variable]
    else:
        all_variables = input_variables+[boolean_variable]


    if config['impute_data_choice']: # If to impute missing data
        if config['impute_with_outcome']: # Impute training data but not valid data
            cat_var_idx_arr_for_imputation = np.append(config['cat_var_idx_arr'], len(input_variables))
            # Fit and impute training data
            imputed_data_train, imputer, imputer_scaler = fit_imputer_and_impute_data(
                data_train[all_variables].copy(), 
                cat_var_idx_arr_for_imputation,
                config['imputer_type']
                )
            imputed_data_train = pd.DataFrame(imputed_data_train, columns=all_variables)
            # Get data x y
            if is_survival_data:
                imputed_data_train_x, data_train_y = get_data_xy_arrays_from_survival_data(imputed_data_train, input_variables, boolean_variable, survival_time_variable)
            else:
                imputed_data_train_x, data_train_y = get_data_xy_arrays_from_classification_data(imputed_data_train, input_variables, boolean_variable)

        else: # Impute without considering outcome variable
            # Split data x y
            if is_survival_data:
                data_train_x, data_train_y = get_data_xy_arrays_from_survival_data(data_train, input_variables, boolean_variable, survival_time_variable)
            else:
                data_train_x, data_train_y = get_data_xy_arrays_from_classification_data(data_train, input_variables, boolean_variable)
            # Impute data train x 
            imputed_data_train_x, imputer, imputer_scaler = fit_imputer_and_impute_data(data_train_x, config['cat_var_idx_arr'], config['imputer_type'])

        # Impute valid data as required
        if impute_data_valid:
            imputed_data_valid = impute_new_dataset(data_valid, config, imputer, imputer_scaler, is_survival_data)

    else: # No imputation. Data provided should not have NA.
        if is_survival_data:
            data_train_x, data_train_y = get_data_xy_arrays_from_survival_data(data_train, input_variables, boolean_variable, survival_time_variable)
        else:
            data_train_x, data_train_y = get_data_xy_arrays_from_classification_data(data_train, input_variables, boolean_variable)

        imputer=None
        imputer_scaler=None
        imputed_data_train_x = data_train_x
        
        if impute_data_valid:
            imputed_data_valid = data_valid

        
    if impute_data_valid:
        if is_survival_data:
            imputed_data_valid_x, data_valid_y = get_data_xy_arrays_from_survival_data(imputed_data_valid, input_variables, boolean_variable, survival_time_variable)
        else:
            imputed_data_valid_x, data_valid_y = get_data_xy_arrays_from_classification_data(imputed_data_valid, input_variables, boolean_variable)

        return imputed_data_train_x, data_train_y, imputed_data_valid_x, data_valid_y, imputer, imputer_scaler
    else:
        return imputed_data_train_x, data_train_y, imputer, imputer_scaler

def impute_new_dataset(new_dataset, config, imputer, imputer_scaler, is_survival_data):
    # new_dataset is a dataframe
    imputed_new_dataset = new_dataset.copy()
    input_variables = config['input_variables'] 
    boolean_variable = config['boolean_variable'] 

    if is_survival_data:
        survival_time_variable = config['survival_time_variable']
        all_variables = input_variables+[boolean_variable, survival_time_variable]
    else:
        all_variables = input_variables+[boolean_variable]

    if config['impute_with_outcome']: 
        cat_var_idx_arr_for_imputation = np.append(config['cat_var_idx_arr'], len(input_variables))
        selected_variables = all_variables
    else:
        cat_var_idx_arr_for_imputation = config['cat_var_idx_arr']
        selected_variables = input_variables

    
    imputed_new_dataset[selected_variables] = impute_data_with_imputer(new_dataset[selected_variables].copy().to_numpy(), config['imputer_type'], imputer, imputer_scaler, cat_var_idx_arr_for_imputation)
    
    return imputed_new_dataset

#*************************************************************************************************************************************************************
### Functions for standardising input data ###
def standardise_cont_vars_in_data(data, cont_var_idx_arr, fit_scaler=True, provided_scaler=None):
    '''This function returns data in its original data type -- dataframe or array'''
    standardised_data = data.copy()
    if type(data) == pd.core.frame.DataFrame:
        if fit_scaler:
            new_scaler = StandardScaler()
            standardised_data.iloc[:,cont_var_idx_arr] = new_scaler.fit_transform(standardised_data.iloc[:,cont_var_idx_arr])
            return standardised_data, new_scaler
        else:
            if provided_scaler == None:
                print('No scaler provided!')
                return None
            else:
                standardised_data.iloc[:,cont_var_idx_arr] = provided_scaler.transform(standardised_data.iloc[:,cont_var_idx_arr])
                return standardised_data

    else: #standardised_data has to be a numpy array
        if fit_scaler:
            data_selected_columns = standardised_data[:,cont_var_idx_arr].copy()
            new_scaler = StandardScaler()
            standardised_data[:,cont_var_idx_arr] = new_scaler.fit_transform(data_selected_columns)
            return standardised_data, new_scaler
        else:
            if provided_scaler == None:
                print('No scaler provided!')
                return None
            else:
                standardised_data[:,cont_var_idx_arr] = provided_scaler.transform(standardised_data[:,cont_var_idx_arr])
                return standardised_data



if __name__ == '__main__':
    print("Running utils.py")