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
        print('Finding the Challenge data...')

    records = find_records(data_folder)
    num_records = len(records)

    if num_records == 0:
        raise FileNotFoundError('No data was provided.')

    # Extract the features and labels.
    if verbose:
        print('Extracting features and labels from the data...')

    features = list()

    for i in range(num_records):
        if verbose:
            width = len(str(num_records))
            print(f'- {i+1:>{width}}/{num_records}: {records[i]}...')

        record = os.path.join(data_folder, records[i])

        # Extract the features from the image...
        current_features = extract_features(record)
        features.append(current_features)

    # Train the model.
    if verbose:
        print('Training the model on the data...')

    # This overly simple model uses the mean of these overly simple features as a seed for a random number generator.
    model = np.mean(features)

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Save the model.
    save_digitization_model(model_folder, model)

    if verbose:
        print('Done.')
        print()

# Train your dx classification model.
def train_dx_model(data_folder, model_folder, verbose):
    # Find data files.
    if verbose:
        print('Training the dx classification model...')
        print('Finding the Challenge data...')

    records = find_records(data_folder)
    num_records = len(records)

    if num_records == 0:
        raise FileNotFoundError('No data was provided.')

    # Extract the features and labels.
    if verbose:
        print('Extracting features and labels from the data...')

    features = list()
    dxs = list()

    for i in range(num_records):
        if verbose:
            width = len(str(num_records))
            print(f'- {i+1:>{width}}/{num_records}: {records[i]}...')

        record = os.path.join(data_folder, records[i])

        # Extract the features from the image, but only if the image has one or more dx classes.
        dx = load_dx(record)
        if dx:
            current_features = extract_features(record)
            features.append(current_features)
            dxs.append(dx)

    if not dxs:
        raise Exception('There are no labels for the data.')

    features = np.vstack(features)
    classes = sorted(set.union(*map(set, dxs)))
    dxs = compute_one_hot_encoding(dxs, classes)

    # Train the model.
    if verbose:
        print('Training the model on the data...')

    # Define parameters for random forest classifier and regressor.
    n_estimators   = 12  # Number of trees in the forest.
    max_leaf_nodes = 34  # Maximum number of leaf nodes in each tree.
    random_state   = 56  # Random state; set for reproducibility.

    # Fit the model.
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, random_state=random_state).fit(features, dxs)

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Save the model.
    save_dx_model(model_folder, model, classes)

    if verbose:
        print('Done.')
        print()

# Load your trained digitization model. This function is *required*. You should edit this function to add your code, but do *not*
# change the arguments of this function. If you do not train a digitization model, then you can return None.
def load_digitization_model(model_folder, verbose):
    filename = os.path.join(model_folder, 'digitization_model.sav')
    return joblib.load(filename)

# Load your trained dx classification model. This function is *required*. You should edit this function to add your code, but do
# *not* change the arguments of this function. If you do not train a dx classification model, then you can return None.
def load_dx_model(model_folder, verbose):
    filename = os.path.join(model_folder, 'dx_model.sav')
    return joblib.load(filename)

# Run your trained digitization model. This function is *required*. You should edit this function to add your code, but do *not*
# change the arguments of this function.
def run_digitization_model(digitization_model, record, verbose):
    model = digitization_model['model']

    # Extract features.
    features = extract_features(record)

    # Load the dimensions of the signal.
    header_file = get_header_file(record)
    header = load_text(header_file)

    num_samples = get_num_samples(header)
    num_signals = get_num_signals(header)

    # For a overly simply minimal working example, generate "random" waveforms.
    seed = int(round(model + np.mean(features)))
    signal = np.random.default_rng(seed=seed).uniform(low=-1000, high=1000, size=(num_samples, num_signals))
    signal = np.asarray(signal, dtype=np.int16)

    return signal

# Run your trained dx classification model. This function is *required*. You should edit this function to add your code, but do
# *not* change the arguments of this function.
def run_dx_model(dx_model, record, signal, verbose):
    model = dx_model['model']
    classes = dx_model['classes']

    # Extract features.
    features = extract_features(record)
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
def extract_features(record):
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

# Save your trained dx classification model.
def save_dx_model(model_folder, model, classes):
    d = {'model': model, 'classes': classes}
    filename = os.path.join(model_folder, 'dx_model.sav')
    joblib.dump(d, filename, protocol=0)