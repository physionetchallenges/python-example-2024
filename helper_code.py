#!/usr/bin/env python

# Do *not* edit this script.
# These are helper functions that you can use with your code.
# Check the example code to see how to use these functions in your code.

import numpy as np
import os
import scipy as sp
import sys
import wfdb

from scipy.signal import fftconvolve
from scipy.ndimage import gaussian_filter

### Challenge variables
substring_labels = '# Labels:'
substring_images = '# Image:'

### Challenge data I/O functions

# Find the records in a folder and its subfolders.
def find_records(folder):
    records = set()
    for root, directories, files in os.walk(folder):
        for file in files:
            extension = os.path.splitext(file)[1]
            if extension == '.hea':
                record = os.path.relpath(os.path.join(root, file), folder)[:-4]
                records.add(record)
    records = sorted(records)
    return records

# Load the header for a record.
def load_header(record):
    header_file = get_header_file(record)
    header = load_text(header_file)
    return header

# Load the signals for a record.
def load_signals(record):
    signal_files = get_signal_files(record)
    path = os.path.split(record)[0]
    signal_files_exist = all(os.path.isfile(os.path.join(path, signal_file)) for signal_file in signal_files)
    if signal_files and signal_files_exist:
        signal, fields = wfdb.rdsamp(record)
    else:
        signal, fields = None, None
    return signal, fields

# Load the images for a record.
def load_images(record):
    from PIL import Image

    path = os.path.split(record)[0]
    image_files = get_image_files(record)

    images = list()
    for image_file in image_files:
        image_file_path = os.path.join(path, image_file)
        if os.path.isfile(image_file_path):
            image = Image.open(image_file_path)
            images.append(image)

    return images

# Load the labels for a record.
def load_labels(record):
    header = load_header(record)
    labels = get_labels_from_header(header)
    return labels

# Save the header for a record.
def save_header(record, header):
    header_file = get_header_file(record)
    save_text(header_file, header)

# Save the signals for a record.
def save_signals(record, signal, comments=list()):
    header = load_header(record)
    path, record = os.path.split(record)
    sampling_frequency = get_sampling_frequency(header)
    signal_formats = get_signal_formats(header)
    adc_gains = get_adc_gains(header)
    baselines = get_baselines(header)
    signal_units = get_signal_units(header)
    signal_names = get_signal_names(header)
    comments = [comment.replace('#', '').strip() for comment in comments]

    wfdb.wrsamp(record, fs=sampling_frequency, units=signal_units, sig_name=signal_names, \
                p_signal=signal, fmt=signal_formats, adc_gain=adc_gains, baseline=baselines, comments=comments,
                write_dir=path)

# Save the labels for a record.
def save_labels(record, labels):
    header_file = get_header_file(record)
    header = load_text(header_file)
    header += substring_labels + ' ' + ', '.join(labels) + '\n'
    save_text(header_file, header)
    return header

### Helper Challenge functions

# Load a text file as a string.
def load_text(filename):
    with open(filename, 'r') as f:
        string = f.read()
    return string

# Save a string as a text file.
def save_text(filename, string):
    with open(filename, 'w') as f:
        f.write(string)

# Get a variable from a string.
def get_variable(string, variable_name):
    variable = ''
    has_variable = False
    for l in string.split('\n'):
        if l.startswith(variable_name):
            variable = l[len(variable_name):].strip()
            has_variable = True
    return variable, has_variable

# Get variables from a string.
def get_variables(string, variable_name, sep=','):
    variables = list()
    has_variable = False
    for l in string.split('\n'):
        if l.startswith(variable_name):
            variables += [variable.strip() for variable in l[len(variable_name):].strip().split(sep)]
            has_variable = True
    return variables, has_variable

# Get the signal files from a header or a similar string.
def get_signal_files_from_header(string):
    signal_files = list()
    for i, l in enumerate(string.split('\n')):
        arrs = [arr.strip() for arr in l.split(' ')]
        if i==0 and not l.startswith('#'):
            num_channels = int(arrs[1])
        elif i<=num_channels and not l.startswith('#'):
            signal_file = arrs[0]
            if signal_file not in signal_files:
                signal_files.append(signal_file)
        else:
            break
    return signal_files

# Get the image files from a header or a similar string.
def get_image_files_from_header(string):
    images, has_image = get_variables(string, substring_images)
    if not has_image:
        raise Exception('No images available: did you forget to generate or include the images?')
    return images

# Get the labels from a header or a similar string.
def get_labels_from_header(string):
    labels, has_labels = get_variables(string, substring_labels)
    if not has_labels:
        raise Exception('No labels available: are you trying to load the labels from the held-out data, or did you forget to prepare the data to include the labels?')
    return labels

# Get the header file for a record.
def get_header_file(record):
    if not record.endswith('.hea'):
        header_file = record + '.hea'
    else:
        header_file = record
    return header_file

# Get the signal files for a record.
def get_signal_files(record):
    header_file = get_header_file(record)
    header = load_text(header_file)
    signal_files = get_signal_files_from_header(header)
    return signal_files

# Get the image files for a record.
def get_image_files(record):
    header_file = get_header_file(record)
    header = load_text(header_file)
    image_files = get_image_files_from_header(header)
    return image_files

### WFDB functions

# Get the record name from a header file.
def get_record_name(string):
    value = string.split('\n')[0].split(' ')[0].split('/')[0].strip()
    return value

# Get the number of signals from a header file.
def get_num_signals(string):
    value = string.split('\n')[0].split(' ')[1].strip()
    if is_integer(value):
        value = int(value)
    else:
        value = None
    return value

# Get the sampling frequency from a header file.
def get_sampling_frequency(string):
    value = string.split('\n')[0].split(' ')[2].split('/')[0].strip()
    if is_number(value):
        value = float(value)
    else:
        value = None
    return value

# Get the number of samples from a header file.
def get_num_samples(string):
    value = string.split('\n')[0].split(' ')[3].strip()
    if is_integer(value):
        value = int(value)
    else:
        value = None
    return value

# Get the signal formats from a header file.
def get_signal_formats(string):
    num_signals = get_num_signals(string)
    values = list()
    for i, l in enumerate(string.split('\n')):
        if 1 <= i <= num_signals:
            field = l.split(' ')[1]
            if 'x' in field:
                field = field.split('x')[0]
            if ':' in field:
                field = field.split(':')[0]
            if '+' in field:
                field = field.split('+')[0]
            value = field
            values.append(value)
    return values

# Get the ADC gains from a header file.
def get_adc_gains(string):
    num_signals = get_num_signals(string)
    values = list()
    for i, l in enumerate(string.split('\n')):
        if 1 <= i <= num_signals:
            field = l.split(' ')[2]
            if '/' in field:
                field = field.split('/')[0]
            if '(' in field and ')' in field:
                field = field.split('(')[0]
            value = float(field)
            values.append(value)
    return values

# Get the baselines from a header file.
def get_baselines(string):
    num_signals = get_num_signals(string)
    values = list()
    for i, l in enumerate(string.split('\n')):
        if 1 <= i <= num_signals:
            field = l.split(' ')[2]
            if '/' in field:
                field = field.split('/')[0]
            if '(' in field and ')' in field:
                field = field.split('(')[1].split(')')[0]
            else:
                field = get_adc_zeros(string)[i-1]
            value = int(field)
            values.append(value)
    return values

# Get the signal units from a header file.
def get_signal_units(string):
    num_signals = get_num_signals(string)
    values = list()
    for i, l in enumerate(string.split('\n')):
        if 1 <= i <= num_signals:
            field = l.split(' ')[2]
            if '/' in field:
                value = field.split('/')[1]
            else:
                value = 'mV'
            values.append(value)
    return values

# Get the ADC resolutions from a header file.
def get_adc_resolutions(string):
    num_signals = get_num_signals(string)
    values = list()
    for i, l in enumerate(string.split('\n')):
        if 1 <= i <= num_signals:
            field = l.split(' ')[3]
            value = int(field)
            values.append(value)
    return values

# Get the ADC zeros from a header file.
def get_adc_zeros(string):
    num_signals = get_num_signals(string)
    values = list()
    for i, l in enumerate(string.split('\n')):
        if 1 <= i <= num_signals:
            field = l.split(' ')[4]
            value = int(field)
            values.append(value)
    return values

# Get the initial values of a signal from a header file.
def get_initial_values(string):
    num_signals = get_num_signals(string)
    values = list()
    for i, l in enumerate(string.split('\n')):
        if 1 <= i <= num_signals:
            field = l.split(' ')[5]
            value = int(field)
            values.append(value)
    return values

# Get the checksums of a signal from a header file.
def get_checksums(string):
    num_signals = get_num_signals(string)
    values = list()
    for i, l in enumerate(string.split('\n')):
        if 1 <= i <= num_signals:
            field = l.split(' ')[6]
            value = int(field)
            values.append(value)
    return values

# Get the block sizes of a signal from a header file.
def get_block_sizes(string):
    num_signals = get_num_signals(string)
    values = list()
    for i, l in enumerate(string.split('\n')):
        if 1 <= i <= num_signals:
            field = l.split(' ')[7]
            value = int(field)
            values.append(value)
    return values

# Get the signal names from a header file.
def get_signal_names(string):
    num_signals = get_num_signals(string)
    values = list()
    for i, l in enumerate(string.split('\n')):
        if 1 <= i <= num_signals:
            value = l.split(' ')[8]
            values.append(value)
    return values

### Evaluation functions

# Construct the binary one-vs-rest confusion matrices, where the columns are the expert labels and the rows are the classifier
# for the given classes.
def compute_one_vs_rest_confusion_matrix(labels, outputs, classes):
    assert np.shape(labels) == np.shape(outputs)

    num_instances = len(labels)
    num_classes = len(classes)

    A = np.zeros((num_classes, 2, 2))
    for i in range(num_instances):
        for j in range(num_classes):
            if labels[i, j] == 1 and outputs[i, j] == 1: # TP
                A[j, 0, 0] += 1
            elif labels[i, j] == 0 and outputs[i, j] == 1: # FP
                A[j, 0, 1] += 1
            elif labels[i, j] == 1 and outputs[i, j] == 0: # FN
                A[j, 1, 0] += 1
            elif labels[i, j] == 0 and outputs[i, j] == 0: # TN
                A[j, 1, 1] += 1

    return A

# Compute macro F-measure.
def compute_f_measure(labels, outputs):
    # Compute confusion matrix.
    classes = sorted(set.union(*map(set, labels)))
    labels = compute_one_hot_encoding(labels, classes)
    outputs = compute_one_hot_encoding(outputs, classes)
    A = compute_one_vs_rest_confusion_matrix(labels, outputs, classes)

    num_classes = len(classes)
    per_class_f_measure = np.zeros(num_classes)
    for k in range(num_classes):
        tp, fp, fn, tn = A[k, 0, 0], A[k, 0, 1], A[k, 1, 0], A[k, 1, 1]
        if 2 * tp + fp + fn > 0:
            per_class_f_measure[k] = float(2 * tp) / float(2 * tp + fp + fn)
        else:
            per_class_f_measure[k] = float('nan')

    if np.any(np.isfinite(per_class_f_measure)):
        macro_f_measure = np.nanmean(per_class_f_measure)
    else:
        macro_f_measure = float('nan')

    return macro_f_measure, per_class_f_measure, classes

# Normalize the channel names.
def normalize_names(names_ref, names_est):
    tmp = list()
    for a in names_est:
        for b in names_ref:
            if a.casefold() == b.casefold():
                tmp.append(b)
                break
    return tmp

# Reorder channels in signal.
def reorder_signal(input_signal, input_channels, output_channels):
    # Do not allow repeated channels with potentially different values in a signal.
    assert(len(set(input_channels)) == len(input_channels))
    assert(len(set(output_channels)) == len(output_channels))

    if input_channels == output_channels:
        output_signal = input_signal
    else:
        output_channels = normalize_names(input_channels, output_channels)

        input_signal = np.asarray(input_signal)
        num_samples = np.shape(input_signal)[0]
        num_channels = len(output_channels)
        data_type = input_signal.dtype

        output_signal = np.zeros((num_samples, num_channels), dtype=data_type)
        for i, output_channel in enumerate(output_channels):
            for j, input_channel in enumerate(input_channels):
                if input_channel == output_channel:
                    output_signal[:, i] = input_signal[:, j]

    return output_signal

# Quantize the 1D signal amplitudes to convert the 1D real-valued signal to a 2D binarized signal.
def convert_signal(x, num_quant_levels, min_amp, max_amp, max_t):
    idx = np.isfinite(x)

    t = np.arange(1, np.size(x) + 1)
    t = t[idx]

    y = x[idx]
    y = np.round((num_quant_levels - 1) * (y - min_amp) / (max_amp - min_amp) + 1).astype(int)
    y = np.clip(y, 1, num_quant_levels)

    A = np.zeros((num_quant_levels, max_t))
    A[y - 1, t - 1] = 1
    return A

# Correlate the 2D signals in the spectral domain.
def fft_correlate(A_ref, A_est):
    # Flip the digitized signal for correlation. 
    A_est_flipped = np.flip(np.flip(A_est, axis=0), axis=1)
    return fftconvolve(A_ref, A_est_flipped, mode='full')

def align_signals(x_ref, x_est, num_quant_levels, smooth=True, sigma=0.5):
    # Estimate the vertical and horizontal offsets of the estimated signal vs. a reference signal
    # in noisy conditions.
    # Reza Sameni, Zuzana Koscova, Matthew Reyna, July 2024

    # Summarize the durations and amplitudes of the signals.
    min_amp = min(np.nanmin(x_ref), np.nanmin(x_est))
    max_amp = max(np.nanmax(x_ref), np.nanmax(x_est))
    max_t = max(np.size(x_ref), np.size(x_est))

    # Quantize the 1D signal amplitudes to convert the 1D real-valued signals to 2D binarized signals.
    A_ref = convert_signal(x_ref, num_quant_levels, min_amp, max_amp, max_t)
    A_est = convert_signal(x_est, num_quant_levels, min_amp, max_amp, max_t)

    # Apply Gaussian smoothing to the 2D binarized signals (optional).
    if smooth:
        A_ref = gaussian_filter(A_ref, sigma)
        A_est = gaussian_filter(A_est, sigma)
    
    # Compute the cross-correlation of 2D reference and estimated signals in the spectral domain.
    A_cross = fft_correlate(A_ref, A_est)
    idx_cross = np.unravel_index(np.argmax(A_cross), A_cross.shape)
                                 
    # Compute the auto-correlation of the reference signal in the spectral domain.
    A_auto = fft_correlate(A_ref, A_ref)
    idx_auto = np.unravel_index(np.argmax(A_auto), A_auto.shape)
   
    # Estimate vertical and horizontal offsets from the cross-correlation peak lags.
    offset_hz = idx_auto[1] - idx_cross[1]
    offset_vt = idx_auto[0] - idx_cross[0]
    offset_vt = offset_vt / (num_quant_levels - 1) * (max_amp - min_amp)

    # Shift the estimated signal by the estimated offsets.
    if offset_hz < 0:
        x_est_shifted = np.concatenate((np.nan*np.ones(-offset_hz), x_est))
    else:
        x_est_shifted = np.concatenate((x_est[offset_hz:], np.nan*np.ones(offset_hz)))
    x_est_shifted -= offset_vt

    return x_est_shifted, offset_hz, offset_vt

def compute_snr(x_ref, x_est, keep_nans=True, signal_median=False, noise_median=False):
    # Check the reference and estimated signals.
    x_ref = np.asarray(x_ref).copy()
    x_est = np.asarray(x_est).copy()
    assert(x_ref.ndim == x_est.ndim == 1)

    # Pad the shorter signal with NaNs so that both signals have the same length.
    n_ref = np.size(x_ref)
    n_est = np.size(x_est)
    if n_est < n_ref:
        x_est = np.concatenate((x_est, np.nan*np.ones(n_ref - n_est)))
    elif n_est > n_ref:
        x_ref = np.concatenate((x_ref, np.nan*np.ones(n_est - n_ref)))

    # Identify the samples with finite values, i.e., not NaN, +\infty, or -\infty.
    idx_ref = np.isfinite(x_ref)
    idx_est = np.isfinite(x_est) 

    # Either only consider samples with finite values in both signals (default) or replace the non-finite values in the estimated signal with zeros.
    if keep_nans:
        idx = np.logical_and(idx_ref, idx_est)
    else:
        x_est[~idx_est] = 0
        idx = idx_ref

    x_ref = x_ref[idx]
    x_est = x_est[idx]

    # Compute the noise.
    x_noise = x_ref - x_est

    # Compute the power for the signal and the noise using either the mean (default) or the median.
    if not signal_median:
        p_signal = np.mean(x_ref**2)
    else:
        p_signal = np.median(x_ref**2)

    if not noise_median:
        p_noise = np.mean(x_noise**2)
    else:
        p_noise = np.median(x_noise**2)

    # Compute the SNR.
    if p_signal > 0 and p_noise > 0:
        snr = 10 * np.log10(p_signal / p_noise)
    elif p_noise == 0:
        snr = float('inf')
    else:
        snr = float('nan')

    # If only considering the samples with finite values in both signals, then penalize the samples with non-finite values in the
    # estimated signal but not in the reference signal.
    if keep_nans:
        alpha = np.sum(idx) / np.sum(idx_ref)
        snr *= alpha

    return snr, p_signal, p_noise

# Compute a metric inspired by the Kolmogorov-Smirnov test statistic.
def compute_ks_metric(x_ref, x_est, keep_nans=True):
    # Check the reference and estimated signals.
    x_ref = np.asarray(x_ref).copy()
    x_est = np.asarray(x_est).copy()
    assert(x_ref.ndim == x_est.ndim == 1)

    # Pad the shorter signal with NaNs so that both signals have the same length.
    n_ref = np.size(x_ref)
    n_est = np.size(x_est)
    if n_est < n_ref:
        x_est = np.concatenate((x_est, np.nan*np.ones(n_ref - n_est)))
    elif n_est > n_ref:
        x_ref = np.concatenate((x_ref, np.nan*np.ones(n_est - n_ref)))

     # Identify the samples with finite values, i.e., not NaN, +\infty, or -\infty.
    idx_ref = np.isfinite(x_ref)
    idx_est = np.isfinite(x_est) 

    # Either only consider samples with finite values in both signals (default) or replace the non-finite values in the estimated signal with zeros.
    if keep_nans:
        idx = np.logical_and(idx_ref, idx_est)
    else:
        x_est[~idx_est] = 0
        idx = idx_ref

    x_ref = x_ref[idx]
    x_est = x_est[idx]

    x_ref_cdf = np.nancumsum(np.abs(x_ref))
    x_est_cdf = np.nancumsum(np.abs(x_est))

    if x_ref_cdf[-1] > 0:
        x_ref_cdf = x_ref_cdf / x_ref_cdf[-1]
    if x_est_cdf[-1] > 0:
        x_est_cdf = x_est_cdf / x_est_cdf[-1]

    goodness_of_fit = 1.0 - np.max(np.abs(x_ref_cdf - x_est_cdf))

    return goodness_of_fit

# Compute the adaptive signed correlation index (ASCI) metric.
def compute_asci_metric(x_ref, x_est, beta=0.05, keep_nans=True):
    # Check the reference and estimated signals.
    x_ref = np.asarray(x_ref).copy()
    x_est = np.asarray(x_est).copy()
    assert(x_ref.ndim == x_est.ndim == 1)

    # Pad the shorter signal with NaNs so that both signals have the same length.
    n_ref = np.size(x_ref)
    n_est = np.size(x_est)
    if n_est < n_ref:
        x_est = np.concatenate((x_est, np.nan*np.ones(n_ref - n_est)))
    elif n_est > n_ref:
        x_ref = np.concatenate((x_ref, np.nan*np.ones(n_est - n_ref)))

    # Identify the samples with finite values, i.e., not NaN, +\infty, or -\infty.
    idx_ref = np.isfinite(x_ref)
    idx_est = np.isfinite(x_est) 

    # Either only consider samples with finite values in both signals (default) or replace the non-finite values in the estimated signal with zeros.
    if keep_nans:
        idx = np.logical_and(idx_ref, idx_est)
    else:
        x_est[~idx_est] = 0
        idx = idx_ref

    x_ref = x_ref[idx]
    x_est = x_est[idx]

    # Check the threshold parameter beta and discretize the nose.
    if beta <= 0 or beta > 1:
        raise ValueError('The beta value should be greater than 0 and less than or equal to 1.')

    threshold = beta * np.std(x_ref)

    x_noise = np.abs(x_ref - x_est)

    x_noise_discretized = np.zeros_like(x_noise)
    x_noise_discretized[x_noise <= threshold] = 1
    x_noise_discretized[x_noise > threshold] = -1

    asci = np.mean(x_noise_discretized)

    return asci

# Compute a weighted absolute difference metric.
def compute_weighted_absolute_difference(x_ref, x_est, sampling_frequency, keep_nans=True):
    # Check the reference and estimated signals.
    x_ref = np.asarray(x_ref).copy()
    x_est = np.asarray(x_est).copy()
    assert(x_ref.ndim == x_est.ndim == 1)

    # Pad the shorter signal with NaNs so that both signals have the same length.
    n_ref = np.size(x_ref)
    n_est = np.size(x_est)
    if n_est < n_ref:
        x_est = np.concatenate((x_est, np.nan*np.ones(n_ref - n_est)))
    elif n_est > n_ref:
        x_ref = np.concatenate((x_ref, np.nan*np.ones(n_est - n_ref)))

    # Identify the samples with finite values, i.e., not NaN, +\infty, or -\infty.
    idx_ref = np.isfinite(x_ref)
    idx_est = np.isfinite(x_est) 

    # Either only consider samples with finite values in both signals (default) or replace the non-finite values in the estimated signal with zeros.
    if keep_nans:
        idx = np.logical_and(idx_ref, idx_est)
    else:
        x_est[~idx_est] = 0
        idx = idx_ref

    x_ref = x_ref[idx]
    x_est = x_est[idx]

    # Filter the reference signal and compute a weighted absolute difference metric between the signals.
    from scipy.signal import filtfilt

    m = round(0.1 * sampling_frequency)
    w = filtfilt(np.ones(m), m, x_ref, method='gust')
    w = 1 - 0.5/np.max(w) * w
    n = np.sum(w)

    weighted_absolute_difference_metric = np.sum(np.abs(x_ref - x_est) * w)/n

    return weighted_absolute_difference_metric

### Other helper functions

# Check if a variable is a number or represents a number.
def is_number(x):
    try:
        float(x)
        return True
    except (ValueError, TypeError):
        return False

# Check if a variable is an integer or represents an integer.
def is_integer(x):
    if is_number(x):
        return float(x).is_integer()
    else:
        return False

# Check if a variable is a finite number or represents a finite number.
def is_finite_number(x):
    if is_number(x):
        return np.isfinite(float(x))
    else:
        return False

# Check if a variable is a NaN (not a number) or represents a NaN.
def is_nan(x):
    if is_number(x):
        return np.isnan(float(x))
    else:
        return False

# Cast a value to an integer if an integer, a float if a non-integer float, and an unknown value otherwise.
def cast_int_float_unknown(x):
    if is_integer(x):
        x = int(x)
    elif is_finite_number(x):
        x = float(x)
    elif is_number(x):
        x = 'Unknown'
    else:
        raise NotImplementedError(f'Unable to cast {x}.')
    return x

# Construct the one-hot encoding of data for the given classes.
def compute_one_hot_encoding(data, classes):
    num_instances = len(data)
    num_classes = len(classes)

    one_hot_encoding = np.zeros((num_instances, num_classes), dtype=np.bool_)
    unencoded_data = list()
    for i, x in enumerate(data):
        for y in x:
            for j, z in enumerate(classes):
                if (y == z) or (is_nan(y) and is_nan(z)):
                    one_hot_encoding[i, j] = 1

    return one_hot_encoding