#%%
import numpy as np
import pandas as pd

singapore_complete_data = pd.read_csv("/Users/lirui/Downloads/Cohort_Dementia_Prediction/Cohort_Data/HARMONISATION/Handover_HARMO_all_available/Singapore_data_set_29_10.csv", header=0, index_col=0)
data_in_analysis = singapore_complete_data.dropna(subset=['Age']).copy() # Exclude those people with most variables missing.
assert data_in_analysis.shape[0] == 242

cognitive_tests = {
    'executive_function':['V0edfabscore'],
    'attention':[
        'V0attentiondsforward',
        'V0attentiondsbackward',
        'V0attentionvmsf',
        'V0attentionvmsb',	
        'V0attentionadt'
    ],
    'language':[
        'V0langmbnscore',
        'V0langvfanimscore',	
        'V0langvffoodscore'
    ],
    'verbal_memory':[
        'V0vbmwlrimdtrecal',	
        'V0vbmwlrdelyrecal',	
        'V0vbmwlrdelyrecog',	
        'V0vbmsraimdt',	
        'V0vbmsradelyrecal'
    ],
    'visual_memory':[
        'V0vimprimdt',	
        'V0vimprdely',	
        'V0vimprdelyrecog',
        'V0vimimdt',	
        'V0vimdelyrecal',	
        'V0vimdelyrecog'
    ],
    'visual_construction':[
        'V0visconwmsrvr',	
        'V0visconclkdrawtst',	
        'V0visconwaisblkdsgn'
    ],
    'visual_motor_speed':[
        'V0vismotordct',
        'V0vismotorsdmt',	
        'minus_V0vismotormzetsk'
    ]
}

# From Marco's script, should use minus maze task scores
data_in_analysis['minus_V0vismotormzetsk'] = -1*data_in_analysis['V0vismotormzetsk']

# Select the control group
MCI_group = data_in_analysis.loc[~data_in_analysis['MCI_diagnosis'].isin(['NCI','No subj memory complaint, NSMC'])]
control_group = data_in_analysis.loc[data_in_analysis['MCI_diagnosis'].isin(['NCI','No subj memory complaint, NSMC'])]

assert (MCI_group.shape[0]+control_group.shape[0]) == data_in_analysis.shape[0]

control_stats = {}
all_test_names = []
for domain in list(cognitive_tests.keys()):
    control_stats[domain] = {}
    all_test_names += cognitive_tests[domain]

    for test in cognitive_tests[domain]:
        test_mean = control_group[test].mean()
        test_std = control_group[test].std()
        control_stats[domain][test] = {
            'mean': test_mean,
            'std': test_std
        }
        data_in_analysis['Z_'+test] = (data_in_analysis[test]-test_mean)/test_std


# Compute executive function 
required_Z_score_names = ['Z_'+i for i in cognitive_tests['executive_function']]
data_in_analysis['EF_Rui'] = data_in_analysis[required_Z_score_names].mean(axis=1, skipna=False)

# Compute processing speed
required_Z_score_names = ['Z_'+i for i in cognitive_tests['visual_motor_speed']]
data_in_analysis['PS_Rui'] = data_in_analysis[required_Z_score_names].mean(axis=1, skipna=False)

# Compute global cognition 
required_Z_score_names = ['Z_'+i for i in all_test_names]
data_in_analysis['Global_Rui'] = data_in_analysis[required_Z_score_names].mean(axis=1, skipna=False)


#     # Compute domain average scores
#     domain_test_names = ['Z_'+i for i in cognitive_tests[domain]]
#     data_in_analysis['Avg_'+domain] = data_in_analysis[domain_test_names].mean(axis=1)

#     # Standardise domain scores
#     control_group_updated = data_in_analysis.loc[data_in_analysis['MCI_diagnosis'].isin(['NCI','No subj memory complaint, NSMC'])]
#     domain_mean = control_group_updated['Avg_'+domain].mean()
#     domain_std = control_group_updated['Avg_'+domain].std()
#     data_in_analysis['Z_'+domain] = (data_in_analysis['Avg_'+domain]-domain_mean)/domain_std
# #%%
# # Compute executive function 
# data_in_analysis['EF_Rui'] = data_in_analysis['Z_executive_function']

# # Compute processing speed
# data_in_analysis['PS_Rui'] = data_in_analysis['Z_visual_motor_speed']

# # Compute global cognition 
# all_domain_avg_score_names = ['Avg_'+i for i in list(cognitive_tests.keys())]
# data_in_analysis['Avg_Global'] = data_in_analysis[all_domain_avg_score_names].mean(axis=1)
# control_group_updated = data_in_analysis.loc[data_in_analysis['MCI_diagnosis'].isin(['NCI','No subj memory complaint, NSMC'])]
# global_mean = control_group_updated['Avg_Global'].mean()
# global_std = control_group_updated['Avg_Global'].std()
# data_in_analysis['Global_Rui'] = (data_in_analysis['Avg_Global']-global_mean)/global_std


control_group_updated = data_in_analysis.loc[data_in_analysis['MCI_diagnosis'].isin(['NCI','No subj memory complaint, NSMC'])]
MCI_group_updated = data_in_analysis.loc[~data_in_analysis['MCI_diagnosis'].isin(['NCI','No subj memory complaint, NSMC'])]
for i in ['EF_Rui', 'PS_Rui', 'Global_Rui']:
    print('-----------')
    print(i)
    print('MCI mean (std): {:.2f} ({:.2f})'.format(MCI_group_updated[i].mean(), MCI_group_updated[i].std()))
    print('control mean (std): {:.2f} ({:.2f})'.format(control_group_updated[i].mean(), control_group_updated[i].std()))
    print('All mean (std): {:.2f} ({:.2f})'.format(data_in_analysis[i].mean(), data_in_analysis[i].std()))

merged_df = pd.merge(singapore_complete_data, data_in_analysis[['PID', 'EF_Rui', 'PS_Rui', 'Global_Rui']], how='outer', on='PID')

#%%
merged_df.to_csv('/Users/lirui/Downloads/Cohort_Dementia_Prediction/Cohort_Data/HARMONISATION/Full_data_with_age_std_cog_scores_by_Rui.csv')
