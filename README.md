# Dementia-Prediction-in-SVD

This repository contains the code for the paper **Predicting Incident Dementia in Cerebral Small Vessel Disease: Comparison of Machine Learning and Traditional Models**.

### Data_Preprocessing
This folder contains code used to extract and preprocess data from each cohort.

### Survival
This folder contains code and results from the survival analysis. 
`train_validate_survival_models.py` is the main script for running the survival analysis for different survival models and with different feature sets.
Results for each model and feature set are stored under the folders `Nested_CV_Results/[model]/[feature_set]`, in which the `[model]_[feature_set].pkl` files store the full cross validation results along with the final trained models and any imputers needed in the following structure:
```
{
    'full_validation_results': full_validation_results, 
    'final_model': final_model,
    'final_input_scaler': final_input_scaler,
    'final_imputer': final_imputer, # ignore this as no imputation was done in the end
    'final_imputer_scaler': final_imputer_scaler, # ignore this as no imputation was done in the end
    'config': config
}
```
To use these final models trained to predict on any future participants, one needs to read the corresponding `[model]_[feature_set].pkl` file, and can use the `do_external_validation_survival()` function. 

### Classification
This folder contains code and results from the classification analysis. It has identical folder structure as the `Survival` folder.

### Feature_Importance
Here stores the rankings of different input features by each model.

### Scripts
`analyse_cohort_data.ipynb` analyses the missing data and characteristics of the baseline data from each cohort.
`analyse_results_survival.ipynb` and `analyse_results_classification.ipynb` analyse the cross validation results from each type of analysis and report them in figure or tabular format.
`analyse_final_models.ipynb` examine the results from feature importance analysis.
