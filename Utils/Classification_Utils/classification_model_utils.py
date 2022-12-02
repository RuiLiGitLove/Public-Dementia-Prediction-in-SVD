import sys
import os
# Append the repository path to os.path.
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_DIR)
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklvq import GMLVQ, LGMLVQ
from sklearn_lvq import GrlvqModel
from Data_Preprocessing.preprocessing_utils import reverse_adjust_skewness_for_dataframe


def get_initialized_classification_model(model_name, param_dict):
    if model_name=='GMLVQ':
        try:
            model = GMLVQ(
                    distance_type="adaptive-squared-euclidean",
                    activation_type=param_dict['param_activation_type'],
                    activation_params=param_dict['param_activation_params'],
                    solver_type=param_dict['param_solver_type'],
                    solver_params=param_dict['param_solver_params'],
                    prototype_n_per_class=param_dict['param_prototype_n_per_class'],
                    random_state=666
                    )
        except:
            try:
                model = GMLVQ(
                        distance_type="adaptive-squared-euclidean",
                        activation_type=param_dict['activation_type'],
                        activation_params=param_dict['activation_params'],
                        solver_type=param_dict['solver_type'],
                        solver_params=param_dict['solver_params'],
                        prototype_n_per_class=param_dict['prototype_n_per_class'],
                        random_state=666
                        )
            except:
                print("Default parameter used for", model_name)
                model = GMLVQ(random_state=666)
    elif model_name=='GRLVQ':
        try:
            model = GrlvqModel(
                    prototypes_per_class=param_dict['param_prototypes_per_class'],
                    # regularization=0.5,
                    random_state=666
                    )
        except:
            try:
                model = GrlvqModel(
                        prototypes_per_class=param_dict['prototypes_per_class'],
                        # regularization=0.5,
                        random_state=666
                        )
            except:
                print("Default parameter used for", model_name)
                model = GrlvqModel(random_state=666)
    elif model_name=='LGMLVQ':
        try:
            model = LGMLVQ(
                    distance_type="local-adaptive-squared-euclidean",
                    activation_type=param_dict['param_activation_type'],
                    activation_params=param_dict['param_activation_params'],
                    solver_type=param_dict['param_solver_type'],
                    solver_params=param_dict['param_solver_params'],
                    random_state=666
                    )
        except:
            try:
                model = LGMLVQ(
                    distance_type="local-adaptive-squared-euclidean",
                    activation_type=param_dict['activation_type'],
                    activation_params=param_dict['activation_params'],
                    solver_type=param_dict['solver_type'],
                    solver_params=param_dict['solver_params'],
                    random_state=666
                    )
            except:
                print("Default parameter used for", model_name)
                model = LGMLVQ(random_state=666)

    elif model_name == 'SVM':
        try:
            model = svm.SVC(
                C=param_dict['param_C'],
                kernel=param_dict['param_kernel'],
                gamma=param_dict['param_gamma'],
                class_weight=param_dict['param_class_weight'],
                random_state=666
            )
        except:
            try:
                model = svm.SVC(
                    C=param_dict['C'],
                    kernel=param_dict['kernel'],
                    gamma=param_dict['gamma'],
                    class_weight=param_dict['class_weight'],
                    random_state=666
                )
            except:
                print("Default parameter used for", model_name)
                model = svm.SVC(random_state=666)

    elif model_name in ['Logistic', 'Reg_Logistic']:
        try:
            model = LogisticRegression(
                    penalty=param_dict['param_penalty'],
                    C=param_dict['param_C'],
                    solver=param_dict['param_solver'],
                    l1_ratio= param_dict['param_l1_ratio'],
                    random_state=666 # Used when solver == ‘sag’, ‘saga’ or ‘liblinear’ to shuffle the data
            )
        except:
            try:
                model = LogisticRegression(
                    penalty=param_dict['penalty'],
                    C=param_dict['C'],
                    solver=param_dict['solver'],
                    l1_ratio= param_dict['l1_ratio'],
                    random_state=666
                    )
            except:
                print("Default parameter used for", model_name)
                model = LogisticRegression(random_state=666)
    else:
        print("Unrecognised model type!")
        model=None
    return model

def get_classification_hyperparam_search_setting(model_name, hyperparam_search, oversampler_lst):
    '''If hyperparam_search=True, this function returns a list of hyperparameter dictionaries encoding the space to search through in hyperparameter optimization. 
    If hyperparam_search=False, this function returns one hyperparameter setting that you can use for the model.'''

    hyperparam_dict_lst = None
    if model_name in ['GMLVQ', 'LGMLVQ']:
        if hyperparam_search:
            num_prototypes_class_0 = [1,3,5,7]
            num_prototypes_class_1 = [1,3,5,7]
            hyperparam_dict_lst = [
                {
                    "solver_type": ["bfgs"],
                    "solver_params": [None],
                    "activation_type": ["soft+"],
                    "activation_params": [{"beta": 1}],#[{"beta": beta} for beta in [1,2,3,4]], # varying the shape of activation function. 
                    "prototype_n_per_class": [np.array([x,y]) for x in num_prototypes_class_0 for y in num_prototypes_class_1],
                    "oversampler": oversampler_lst
                }
            ]
        else:
            hyperparam_dict_lst = [
                {
                    "solver_type": ["bfgs"],
                    "solver_params": [None],
                    "activation_type": ["soft+"],
                    "activation_params": [None],
                    "prototype_n_per_class": [1],
                    "oversampler": oversampler_lst
                }
            ]
    elif model_name == 'GRLVQ':
        if hyperparam_search:
            num_prototypes_class_0 = [1,3,5,7]
            num_prototypes_class_1 = [1,3,5,7]
            hyperparam_dict_lst = [
                {
                    "prototypes_per_class": [[x,y] for x in num_prototypes_class_0 for y in num_prototypes_class_1], #Number of prototypes per class. Use list to specify different numbers per class.
                    "oversampler": oversampler_lst
                }
            ]
        else:
            hyperparam_dict_lst = [
                {
                    "prototypes_per_class": [1],
                    "oversampler": oversampler_lst
                }
            ]
    elif model_name == 'SVM':
        if hyperparam_search:
            hyperparam_dict_lst = [
                {
                    "C": [0.2, 0.4, 0.6, 0.8, 1.0], # a larger C means larger penalty to misclassification, which results in smaller margin but less misclassification
                    "kernel": ['linear'],
                    "gamma": [0.01], # a smaller gamma means larger radius of rbf, and larger influence of samples
                    "class_weight": ['balanced'],
                    "oversampler": oversampler_lst
                },
                {   
                    "C": [0.2, 0.4, 0.6, 0.8, 1.0], # a larger C means larger penalty to misclassification, which results in smaller margin but less misclassification
                    "kernel": ['rbf'],
                    "gamma": np.logspace(-7, 0, 8), # a smaller gamma means larger radius of rbf, and larger influence of samples
                    "class_weight": ['balanced'],
                    "oversampler": oversampler_lst
                }
            ]
        else:
            hyperparam_dict_lst = [
                {
                    "oversampler": oversampler_lst
                }
            ]
    elif model_name == 'Reg_Logistic':
        if hyperparam_search:
            hyperparam_dict_lst = [
                # {
                #     "penalty": ['l1'],
                #     "C": [0.5, 1, 2],
                #     "solver": ["liblinear"],
                #     "l1_ratio": [None]
                # },
                # {
                #     "penalty": ['l2', 'none'],
                #     "C": [0.5, 1, 2],
                #     "solver": ["lbfgs"],
                #     "l1_ratio": [None]
                # },
                {
                    "penalty": ['elasticnet'],
                    "C": [0.1, 0.5, 1, 2], # C is inverse of regularization strength; must be a positive float. Smaller values specify stronger regularization.
                    "solver": ["saga"],
                    "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
                    "oversampler": oversampler_lst
                }
            ]
        else: #unregularised logistic regression
            hyperparam_dict_lst = [
                {
                    "penalty": ['none'],
                    "C": [1.0],
                    "solver": ["lbfgs"],
                    "l1_ratio": [None],
                    "oversampler": oversampler_lst
                }
            ]
    elif model_name == 'Logistic':
        if hyperparam_search:
            hyperparam_dict_lst = [
                {
                "penalty": ['none'],
                "C": [1.0], # default=1.0
                "solver": ["lbfgs"],
                "l1_ratio": [None],
                "oversampler": oversampler_lst
                }
            ]
        else: 
            hyperparam_dict_lst = [{
                "penalty": ['none'],
                "C": [1.0],
                "solver": ["lbfgs"],
                "l1_ratio": [None],
                "oversampler": oversampler_lst
            }]
    else:
        print("Unrecognised model name!")
    
    return hyperparam_dict_lst

def get_LVQ_prototypes_in_dataframe(prototypes, labels, scaler, input_variables, outcome_variable_name, cont_var_idx_arr):
    (n_prototypes, n_features) = prototypes.shape
    assert n_features == len(input_variables)

    original_input_variables = []
    for input_variable in input_variables:
        if input_variable != 'Trans_PSMD':
            original_input_variables.append(input_variable.removeprefix('Trans_'))
        else:
            original_input_variables.append(input_variable)

    prototypes[:, cont_var_idx_arr] = scaler.inverse_transform(prototypes[:, cont_var_idx_arr])
    proto_df = pd.DataFrame(prototypes, columns=input_variables)
    proto_df[outcome_variable_name] = labels
    proto_df = reverse_adjust_skewness_for_dataframe(proto_df)
    return proto_df[original_input_variables+[outcome_variable_name]]

if __name__ == '__main__':
    print("Running classification_model_utils.py")