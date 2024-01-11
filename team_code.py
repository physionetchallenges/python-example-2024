#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

import joblib
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import sys

from helper_code import *

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments of the functions.
#
################################################################################

# Train your digitization model.
def train_digitization_model(data_folder, model_folder, verbose):
    # Find data files.
    if verbose:
        print('Training the digitization model...')
        print('... finding the Challenge data...')

    records = find_records(data_folder)
    num_records = len(records)

    if num_records == 0:
        raise FileNotFoundError('No data was provided.')

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Extract the features and labels.
    if verbose:
        print('... extracting features and labels from the data...')

    features = list()

    for i in range(num_records):
        if verbose:
            width = len(str(num_records))
            print(f'...    {i+1:>{width}}/{num_records}...')

        record = os.path.join(data_folder, records[i])
        current_features = get_features(record)
        features.append(current_features)

    # Train the model.
    if verbose:
        print('... training the model on the data...')

    # Define parameters for random forest classifier and regressor.

    ###
    ### TO-DO: ADD CODE.
    ###

    # Fit the models.

    ###
    ### TO-DO: ADD CODE.
    ###

    model = np.mean(features)

    # Save the models.
    save_digitization_model(model_folder, model)

    if verbose:
        print('Done.')
        print()

# Train your diagnosis model.
def train_diagnosis_model(data_folder, model_folder, verbose):
    # Find data files.
    if verbose:
        print('Training the diagnosis model...')
        print('... finding the Challenge data...')

    records = find_records(data_folder)
    num_records = len(records)

    if num_records == 0:
        raise FileNotFoundError('No data was provided.')

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Extract the features and labels.
    if verbose:
        print('... extracting features and labels from the data...')

    features = list()
    diagnoses = list()

    for i in range(num_records):
        if verbose:
            width = len(str(num_records))
            print(f'...    {records[i]}; {i+1:>{width}}/{num_records}...')

        record = os.path.join(data_folder, records[i])
        current_features = get_features(record)
        features.append(current_features)

        # Extract labels.
        header_file = get_header_file(record)
        header = load_text(header_file)
        diagnosis = get_diagnosis(header)
        diagnoses.append(diagnosis)

    features = np.vstack(features)
    classes = sorted(set.union(*map(set, diagnoses)))
    diagnoses = compute_one_hot_encoding(diagnoses, classes)

    # Train the models.
    if verbose:
        print('... training the model on the data...')

    # Define parameters for random forest classifier and regressor.
    n_estimators   = 123  # Number of trees in the forest.
    max_leaf_nodes = 456  # Maximum number of leaf nodes in each tree.
    random_state   = 789  # Random state; set for reproducibility.

    # Fit the models.
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, random_state=random_state).fit(features, diagnoses)

    # Save the models.
    save_diagnosis_model(model_folder, model, classes)

    if verbose:
        print('Done.')
        print()

# Load your trained digitization model. This function is *required*. You should edit this function to add your code, but do *not*
# change the arguments of this function. If you do not train a digitization model, then you can return None.
def load_digitization_model(model_folder, verbose):
    filename = os.path.join(model_folder, 'digitization_model.sav')
    return joblib.load(filename)

# Load your trained diagnosis model. This function is *required*. You should edit this function to add your code, but do *not*
# change the arguments of this function. If you do not train a diagnosis model, then you can return None.
def load_diagnosis_model(model_folder, verbose):
    filename = os.path.join(model_folder, 'diagnosis_model.sav')
    return joblib.load(filename)

# Run your trained digitization model. This function is *required*. You should edit this function to add your code, but do *not*
# change the arguments of this function.
def run_digitization_model(digitization_model, record, verbose):
    ###
    ### TO-DO: ADD CODE.
    ###

    ###
    ### TO-DO: REMOVE BELOW LINES.
    ###

    seed = digitization_model['model']

    header_file = get_header_file(record)
    header = load_text(header_file)

    num_samples = get_num_samples(header)
    num_signals = get_num_signals(header)

    signal = np.random.default_rng(seed=int(round(seed))).uniform(low=-1000, high=1000, size=(num_samples, num_signals))
    signal = np.asarray(signal, dtype=np.int16)

    return signal

# Run your trained diagnosis model. This function is *required*. You should edit this function to add your code, but do *not* change
# the arguments of this function.
def run_diagnosis_model(diagnosis_model, record, verbose):
    model = diagnosis_model['model']
    classes = diagnosis_model['classes']

    # Extract features.
    features = get_features(record)
    features = features.reshape(1, -1)

    # Get model probabilities.
    probabilities = model.predict_proba(features)
    probabilities = np.asarray(probabilities, dtype=np.float32)[:, 0, 1]

    # Choose the class(es) with the highest probability as the label(s).
    max_probability = np.nanmax(probabilities)
    labels = [classes[i] for i, probability in enumerate(probabilities) if probability == max_probability]

    return labels

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Extract features.
def get_features(record):
    images = load_image(record)
    mean = 0.0
    std = 0.0
    for image in images:
        image = np.asarray(image)
        mean += np.mean(image)
        std += np.std(image)
    return np.array([mean, std])  

# Save your trained digitization model.
def save_digitization_model(model_folder, model):
    d = {'model': model}
    filename = os.path.join(model_folder, 'digitization_model.sav')
    joblib.dump(d, filename, protocol=0)

# Save your trained diagnosis model.
def save_diagnosis_model(model_folder, model, classes):
    d = {'model': model, 'classes': classes}
    filename = os.path.join(model_folder, 'diagnosis_model.sav')
    joblib.dump(d, filename, protocol=0)