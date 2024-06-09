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

# Train your models. This function is *required*. You should edit this function to add your code, but do *not* change the arguments
# of this function. If you do not train one of the models, then you can return None for the model.

# Train your digitization model.
def train_models(data_folder, model_folder, verbose):
    # Find the data files.
    if verbose:
        print('Finding the Challenge data...')

    records = find_records(data_folder)
    num_records = len(records)

    if num_records == 0:
        raise FileNotFoundError('No data were provided.')

    if verbose:
        print('Training the digitization model...')

    # Extract the features and labels from the data.
    if verbose:
        print('Extracting features and labels from the data...')

    digitization_features = list()
    classification_features = list()
    classification_labels = list()

    # Iterate over the records.
    for i in range(num_records):
        if verbose:
            width = len(str(num_records))
            print(f'- {i+1:>{width}}/{num_records}: {records[i]}...')

        record = os.path.join(data_folder, records[i])

        # Extract the features from the image; this simple example uses the same features for the digitization and classification
        # tasks.
        features = extract_features(record)
        
        digitization_features.append(features)

        # Some images may not be labeled...
        labels = load_labels(record)
        if labels:
            classification_features.append(features)
            classification_labels.append(labels)

    # ... but we expect some images to be labeled for classification.
    if not classification_labels:
        raise Exception('There are no labels for the data.')

    # Train the models.
    if verbose:
        print('Training the models on the data...')

    # Train the digitization model. This very simple model uses the mean of these very simple features as a seed for a random number
    # generator.
    digitization_model = np.mean(features)

    # Train the classification model. This very simple model trains a random forest model with these very simple features.

    classification_features = np.vstack(classification_features)
    classes = sorted(set.union(*map(set, classification_labels)))
    classification_labels = compute_one_hot_encoding(classification_labels, classes)

    # Define parameters for random forest classifier and regressor.
    n_estimators   = 12  # Number of trees in the forest.
    max_leaf_nodes = 34  # Maximum number of leaf nodes in each tree.
    random_state   = 56  # Random state; set for reproducibility.

    # Fit the model.
    classification_model = RandomForestClassifier(
        n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, random_state=random_state).fit(classification_features, classification_labels)

    # Create a folder for the models if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Save the models.
    save_models(model_folder, digitization_model, classification_model, classes)

    if verbose:
        print('Done.')
        print()

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function. If you do not train one of the models, then you can return None for the model.
def load_models(model_folder, verbose):
    digitization_filename = os.path.join(model_folder, 'digitization_model.sav')
    digitization_model = joblib.load(digitization_filename)

    classification_filename = os.path.join(model_folder, 'classification_model.sav')
    classification_model = joblib.load(classification_filename)
    return digitization_model, classification_model

# Run your trained digitization model. This function is *required*. You should edit this function to add your code, but do *not*
# change the arguments of this function. If you did not train one of the models, then you can return None for the model.
def run_models(record, digitization_model, classification_model, verbose):
    # Run the digitization model; if you did not train this model, then you can set signal = None.
    model = digitization_model['model']

    # Extract features.
    features = extract_features(record)

    # Load the dimensions of the signal.
    header_file = get_header_file(record)
    header = load_text(header_file)

    num_samples = get_num_samples(header)
    num_signals = get_num_signals(header)

    # Extract features.
    features = extract_features(record)
    features = features.reshape(1, -1)
    
    # Generate "random" waveforms using the a random seed from the feature.
    seed = int(round(model + np.mean(features)))
    signal = np.random.default_rng(seed=seed).uniform(low=-1, high=1, size=(num_samples, num_signals))
    
    # Run the classification model.
    model = classification_model['model']
    classes = classification_model['classes']



    # Get model probabilities.
    probabilities = model.predict_proba(features)
    probabilities = np.asarray(probabilities, dtype=np.float32)[:, 0, 1]

    # Choose the class or classes with the highest probability as the label or labels.
    max_probability = np.nanmax(probabilities)
    labels = [classes[i] for i, probability in enumerate(probabilities) if probability == max_probability]

    return signal, labels

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Extract features.
def extract_features(record):
    images = load_images(record)
    mean = 0.0
    std = 0.0
    for image in images:
        image = np.asarray(image)
        mean += np.mean(image)
        std += np.std(image)
    return np.array([mean, std])

# Save your trained models.
def save_models(model_folder, digitization_model=None, classification_model=None, classes=None):
    if digitization_model is not None:
        d = {'model': digitization_model}
        filename = os.path.join(model_folder, 'digitization_model.sav')
        joblib.dump(d, filename, protocol=0)

    if classification_model is not None:
        d = {'model': classification_model, 'classes': classes}
        filename = os.path.join(model_folder, 'classification_model.sav')
        joblib.dump(d, filename, protocol=0)