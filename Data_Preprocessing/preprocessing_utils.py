import numpy as np
import pandas as pd


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
        return np.log(array+0.005) # natural logarithm
    # elif feature == 'WMH_vol_ml':
    #     return np.log(array+0.1)
    # elif feature in ['num_lacunes', 'num_mb']:
    #     return np.log(array+1)
    # elif feature in ['MMSE','MOCA']:
    #     return np.power(array,2)
    elif feature == 'PSMD':
        return np.log(array)
    else:
        #print('Did not need adjust skewness.')
        return array

def reverse_adjust_skewness(array, feature):
    if feature == 'Trans_SVDp':
        return np.exp(array)-0.005
    # elif feature == 'Trans_WMH_vol_ml':
    #     return np.exp(array)-0.1
    # elif feature in ['Trans_num_lacunes', 'Trans_num_mb']:
    #     return np.exp(array)-1
    # elif feature in ['Trans_MMSE','Trans_MOCA']:
    #     return np.sqrt(array)
    elif feature == 'Trans_PSMD':
        return np.exp(array)
    else:
        return array

def adjust_skewness_for_dataframe(data):
    features = list(data.columns)
    for feature in features:
        data['Trans_'+feature] = adjust_skewness(data[feature], feature)
    return data

def reverse_adjust_skewness_for_dataframe(data):
    features = list(data.columns)
    for feature in features:
        data[feature.removeprefix('Trans_')] = reverse_adjust_skewness(data[feature], feature)
    return data
