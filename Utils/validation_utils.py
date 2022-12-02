import numpy as np
import pandas as pd
import scipy.stats as ss
from sklearn.model_selection import ParameterGrid

def initialize_cv_results_single_hyperparam(metrics, dataset_names): 
    ''' Initializes a dict with keys as all combinations of 'dataset_metric_values'.'''
    results_dict = {}
    for dataset in dataset_names:
        for metric in metrics:
            results_dict[dataset+'_'+metric+'_values'] = []
    return results_dict

def initialize_cv_results_gridSearch(metrics, dataset_names, hyperparam_dict_lst):
    ''' Initializes a dict of this structure: hyperparam setting -> dataset_metric_values'''
    cv_results = {}
    hyperparam_grid = list(ParameterGrid(hyperparam_dict_lst)) # returns a list of combinations of all hyperparameter settings
    for this_hyperparam_dict in hyperparam_grid:
        cv_results[str(this_hyperparam_dict)] = initialize_cv_results_single_hyperparam(metrics, dataset_names)
        for param in list(this_hyperparam_dict.keys()):
            cv_results[str(this_hyperparam_dict)]['param_'+param] = this_hyperparam_dict[param]
    return cv_results

def get_complete_cv_no_gridSearch_results_df(orig_cv_results, dataset_names, metrics):
    # Computes means, stds and rankings
    cv_results = orig_cv_results.copy()
    n_repeats = len(cv_results[dataset_names[0]+'_'+metrics[0]+'_values'])
    
    for dataset in dataset_names:
        for metric in metrics:
            cv_values_key = dataset+'_'+metric+'_values'
            cv_values = cv_results[cv_values_key]
            cv_results[cv_values_key] = [cv_values]
            assert len(cv_values)==n_repeats
            cv_results[dataset+'_'+metric+'_mean'] = np.mean(cv_values)
            cv_results[dataset+'_'+metric+'_std'] = np.std(cv_values)

    # Convert dict to dataframe
    cv_results_df = pd.DataFrame.from_dict(cv_results, orient='columns')
        
    return cv_results_df

def get_complete_cv_gridSearch_results_df(orig_cv_results, dataset_names, metrics, dataset_to_rank, metrics_rank_criteria):
    # Computes means, stds and rankings
    cv_results = orig_cv_results.copy()
    param_grid = list(cv_results.keys())
    n_repeats = len(cv_results[param_grid[0]][dataset_names[0]+'_'+metrics[0]+'_values'])
    
    for this_param_combination in param_grid:
        for dataset in dataset_names:
            for metric in metrics:
                cv_values_key = dataset+'_'+metric+'_values'
                cv_values = cv_results[this_param_combination][cv_values_key]
                assert len(cv_values)==n_repeats
                cv_results[this_param_combination][dataset+'_'+metric+'_mean'] = np.mean(cv_values)
                cv_results[this_param_combination][dataset+'_'+metric+'_std'] = np.std(cv_values)

    # Convert dict to dataframe
    cv_results_df = pd.DataFrame.from_dict(cv_results, orient='index')
    cv_results_df.index.name='params'
    cv_results_df = cv_results_df.reset_index(drop=True)

    # Compute rankings
    for metric in metrics:
        if metrics_rank_criteria[metric]=='smaller_better':
            rank_for_metric = ss.rankdata(cv_results_df[dataset_to_rank+'_'+metric+'_mean']).astype('int')
        elif metrics_rank_criteria[metric]=='larger_better':
            rank_for_metric = ss.rankdata(-1*cv_results_df[dataset_to_rank+'_'+metric+'_mean']).astype('int')
        else:
            print("Unrecognised metric rank criterion:", metrics_rank_criteria[metric])
        cv_results_df['rank_'+dataset_to_rank+'_'+metric] = rank_for_metric
        
    return cv_results_df


def get_complete_inner_cv_results_df(inner_cv_results, inner_cv_metrics, rank_dataset_name='valid', rank_metric='overall', rank_criterion='larger_better'):
    # Add ranking
    for this_param_combination in list(inner_cv_results.keys()):
        for metric in inner_cv_metrics:
            inner_cv_results[this_param_combination][rank_dataset_name+'_'+metric+'_mean'] = np.mean(inner_cv_results[this_param_combination][rank_dataset_name+'_'+metric+'_values'])
        inner_cv_results[this_param_combination][rank_dataset_name+'_overall_mean'] = np.mean([inner_cv_results[this_param_combination][rank_dataset_name+'_'+metric+'_mean'] for metric in inner_cv_metrics])
    inner_cv_results_df = pd.DataFrame.from_dict(inner_cv_results, orient='index')
    inner_cv_results_df.index.name='params'
    inner_cv_results_df = inner_cv_results_df.reset_index()
    # Compute rankings
    if rank_criterion=='larger_better':
        mean_score_column = inner_cv_results_df[rank_dataset_name+'_'+rank_metric+'_mean']
        inner_cv_results_df[rank_dataset_name+'_'+rank_metric+'_rank'] = ss.rankdata(-1*mean_score_column).astype('int')
        rank_column = inner_cv_results_df[rank_dataset_name+'_'+rank_metric+'_rank']

        #assert np.where(rank_column==min(rank_column))[0][0] in np.where(abs(mean_score_column-max(mean_score_column))<1e-6)[0]
    elif rank_criterion=='smaller_better':
        mean_score_column = inner_cv_results_df[rank_dataset_name+'_'+rank_metric+'_mean']
        inner_cv_results_df[rank_dataset_name+'_'+rank_metric+'_rank'] = ss.rankdata(mean_score_column).astype('int')
        rank_column = inner_cv_results_df[rank_dataset_name+'_'+rank_metric+'_rank']
        #assert np.where(rank_column==min(rank_column))[0][0] == np.where(mean_score_column== min(mean_score_column))[0][0]

    return inner_cv_results_df

def get_best_setting_from_gridSearch_cv_results_df(inner_cv_results_df, inner_rank_dataset_name, inner_rank_metric):
    '''This function randomly choose one hyperparameter setting when there are multiple best ones.'''
    rank_column_name = inner_rank_dataset_name+'_'+inner_rank_metric+'_rank'
    all_best_setting_results = inner_cv_results_df.loc[inner_cv_results_df[rank_column_name]==min(inner_cv_results_df[rank_column_name])]
    # if all_best_setting_results.shape[0] > 1:
    #     np.random.seed(0)
    #     best_setting_results = all_best_setting_results.iloc[ np.random.choice(np.arange(all_best_setting_results.shape[0]), 1)[0] ]
    # else:
    #     best_setting_results = all_best_setting_results.iloc[0]
    best_setting_results = all_best_setting_results.iloc[0]
    return best_setting_results


def get_summary_best_param_cv_results(best_param_cv_results, model_name, metrics, dataset_names):
    # best_param_cv_results should be one row of the dataframe from cv_results_df
    best_param_cv_results = best_param_cv_results.iloc[0]
    summary_df = {
        'index': ['Mean', 'SD'],
        'columns': [],
        'data': [[], []],
        'index_names': [model_name],
        'column_names': []
        }

    for metric in metrics:
        summary_df['column_names'] = ['Metric','Dataset']
        for dataset in dataset_names:
            summary_df['columns'].append((str(metric), str(dataset)))
            summary_df['data'][0].append(best_param_cv_results[dataset+'_'+metric+'_mean'])
            summary_df['data'][1].append(best_param_cv_results[dataset+'_'+metric+'_std'])
    
    summary_df = pd.DataFrame.from_dict(summary_df, orient='tight').round(3)
    return summary_df

if __name__ == '__main__':
    print("Running validation_utils.py")