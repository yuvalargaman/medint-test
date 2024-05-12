# -*- coding: utf-8 -*-
"""
    Model selection and comparison
"""
import numpy as np
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, PredefinedSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from itertools import chain

# =============================================================================
#     Auxilliary functions
# =============================================================================


def create_classifiers():
    """
    Creates 4 classifier instances within a dictionary
    """
    classifiers = {'svc': OneVsRestClassifier(SVC(random_state=42)),
                   'logistic_regression': OneVsRestClassifier(LogisticRegression(random_state=42)),
                   'random_forest': OneVsRestClassifier(RandomForestClassifier(random_state=42)),
                   'ridge': OneVsRestClassifier(RidgeClassifier(random_state=42))}
    return classifiers


def prepare_and_scale(X_train, X_val, X_test):
    """
    Function to scale the predictors and separate them from the classes 
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.iloc[:, -2:].values)
    X_val_scaled = scaler.transform(X_val.iloc[:, -2:].values)
    X_test_scaled = scaler.transform(X_test.iloc[:, -2:].values)
    y_train = X_train.loc[:, "Star Rating"]
    y_val = X_val.loc[:, "Star Rating"]
    y_test = X_test.loc[:, "Star Rating"]
    return X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test


def leave_one_subject_out(data, subject_column, val_prop=0.33):
    """
    Generator function to create leave-one-subject-out cross-validation sets
    using the LeaveOneGroupOut class
    """

    logo_train = LeaveOneGroupOut()

    # Map the unique inputs
    groups = data[subject_column]

    sets = []

    # Iterate over each fold, defined by leaving out each subject
    for train_index, test_index in logo_train.split(data, groups=groups):
        train_data = data.iloc[train_index, :].reset_index(drop=True)
        test_data = data.iloc[test_index, :].reset_index(drop=True)

        # next, find the indices for each group in the trainning data
        # for further the validation split
        train_groups = train_data.groupby('Input').groups
        train_group, val_group = train_test_split(
            list(train_groups.keys()), test_size=val_prop)

        # A lambda function to ease getting the data indices right
        def chain_lists(x): return list(chain(*x))

        new_train_index = chain_lists([train_groups[t] for t in train_group])
        val_index = chain_lists([train_groups[v] for v in val_group])

        new_train_data = train_data.iloc[new_train_index, :]
        val_data = train_data.iloc[val_index, :]

        sets.append([new_train_data, val_data, test_data])
    return sets


# =============================================================================
# Load the datasets and plot the relationship between the TF-IDF and
# embedding similarities, colored by the star rating.
# =============================================================================
df_mockup_scores = pd.read_csv("Mockup_Scores.csv")
df_similarities = pd.read_csv("Mockup_Question_to_Input_Similarities.csv")

data = pd.concat([df_mockup_scores, df_similarities], axis=1)

# =============================================================================
# Plot the TF-IDF similarities vs the embedding similarities
# =============================================================================
sb.scatterplot(data=data, x='TF-IDF Similarities',
               y='Embedding Similarities', hue='Star Rating')

# =============================================================================
# Shuffle the data and perform train-test split
# =============================================================================
np.random.seed(42)
indices = np.array(data.index)
data_shuffled = data.iloc[indices, :]

# =============================================================================
# Create datasets for leave-one-subject-out split. The sets will be used for
# the following model evaluation steps.
# Each sets will contain:
# 1 test input
# 3 validation inputs
# 6 training inputs
# =============================================================================
sets = leave_one_subject_out(data_shuffled, 'Input')

# =============================================================================
# Define classifier names and create a dictionary that will hold the results
# for each fold
# =============================================================================
classifier_names = ['svc', 'random_forest', 'logistic_regression', 'ridge']
results = []

# =============================================================================
# Iterate over the folds and collect the results for each classfier
# This will help us select the most appropriate classifier over all folds
# Then, the most appropriate classifier will be used for predicting the
# test output
# =============================================================================
for s in sets:
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_and_scale(*s)
    test_input_ = s[-1]['Input'].unique()
    classifiers = create_classifiers()
    for classifier_name, classifier in classifiers.items():
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_val)
        # get the classification report for the validation data and extract
        # the weighted average of the precision, recall, f1-score, and support
        report = classification_report(y_val, y_pred).split()
        metrics = report[:4]
        weighted_avg = [float(s) for s in report[-4:]]
        output = dict(zip(metrics, weighted_avg))
        output['Classifier'] = classifier_name
        output['Fold'] = test_input_          
        output = pd.DataFrame(output).loc[:, ['Fold', 'Classifier', 'precision', 'recall', 'f1-score', 'support']]
        results.append(output)
df_results = pd.concat(results).reset_index(drop=True).drop('support', axis=1)
# =============================================================================
# Visualize the results
# =============================================================================
df_results_melted = df_results.melt(id_vars=['Fold', 'Classifier'], var_name='Metric', value_name='Value')
sb.barplot(data=df_results_melted, x='Metric', y='Value', hue='Classifier')

metrics_descriptives = df_results.groupby('Classifier').mean().drop(['Fold', 'support'], axis=1)

        
    

# Uncomment the following if you plan to use GridSearchCV
# param_grids = {'svc': {'C': [0.1, 1, 10, 100, 1000],
#                        'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
#                        'kernel': ['rbf', 'linear'],
#                        'decision_function_shape': ['ovo', 'ovr']},
#                'random_forest': {'bootstrap': [True, False],
#                                  'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
#                                  'max_features': ['auto', 'sqrt'],
#                                  'min_samples_leaf': [1, 2, 4],
#                                  'min_samples_split': [2, 5, 10],
#                                  'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]},
#                'logistic_regression': {'penalty': ['l1', 'l2'],
#                                        'C': [1, 10, 100, 1000]},
#                'ridge': {'alpha': np.logspace(-4, 4, num=10),
#                          'solver': ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
#                          'tol': [1e-3, 1e-4]}}

gskwargs = {'refit': True, 'verbose': 3}  # grid search kwargs
grids = {k: GridSearchCV(v, param_grids[k], **gskwargs)
         for k, v in classifiers.items()}

X_train, y_train, X_val, y_val, X_test, y_test = prepare_and_scale(*sets[0])
X_combined = np.vstack((X_train, X_val))
y_combined = np.hstack((y_train, y_val))

# Combine the training and validation sets but use the PredefinedSplit to
# tell GridSearchCV which set is used for training and which is used
# for validation

# Create the split index (-1 for train, 0 for validation)
test_fold = np.full(X_combined.shape[0], -1)  # Default to -1 (training)
test_fold[len(X_train):] = 0  # Validation fold

# Create the PredefinedSplit
ps = PredefinedSplit(test_fold)

# Create a SVM classifier

for classifier, grid in grids.items():
    grid.fit(X_combined, y_combined)
    y_pred = grid.predict(X_test)
    results[classifier].append(
        [grid.best_params_, grid.best_score_, classification_report(y_test, y_pred)])
