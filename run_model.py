#!/usr/bin/env python

# Do *not* edit this script. Changes will be discarded so that we can run the trained models consistently.

# This file contains functions for running models for the Challenge. You can run it as follows:
#
#   python run_model.py -d data -m model -o outputs -v
#
# where 'data' is a folder containing the Challenge data, 'models' is a folder containing the your trained model(s), 'outputs' is a
# folder for saving your model's or models outputs, and -v is an optional verbosity flag.

import argparse
import os
import sys

from helper_code import *
from team_code import load_digitization_model, load_diagnosis_model, run_digitization_model, run_diagnosis_model

# Parse arguments.
def get_parser():
    description = 'Run the trained Challenge model(s).'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-d', '--data_folder', type=str, required=True)
    parser.add_argument('-m', '--model_folder', type=str, required=True)
    parser.add_argument('-o', '--output_folder', type=str, required=True)
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-f', '--allow_failures', action='store_true')
    return parser

# Run the code.
def run(args):
    # Load model(s).
    if args.verbose:
        print('Loading the Challenge model...')

    # You can use these functions to perform tasks, such as loading your model(s), that you only need to perform once.
    digitization_model = load_digitization_model(args.model_folder, args.verbose) ### Teams: Implement this function!!!
    diagnosis_model = load_diagnosis_model(args.model_folder, args.verbose) ### Teams: Implement this function!!!

    # Find the Challenge data.
    if args.verbose:
        print('Finding the Challenge data...')

    records = find_records(args.data_folder)
    num_records = len(records)

    if num_records==0:
        raise Exception('No data were provided.')

    # Create a folder for the Challenge outputs if it does not already exist.
    os.makedirs(args.output_folder, exist_ok=True)

    # Run the team's model(s) on the Challenge data.
    if args.verbose:
        print('Running the Challenge model(s) on the Challenge data...')

    # Iterate over the records.
    for i in range(num_records):
        if args.verbose:
            width = len(str(num_records))
            print(f'...    {records[i]}; {i+1:>{width}}/{num_records}...')

        record = os.path.join(args.data_folder, records[i])
        header_file = get_header_file(record)
        header = load_text(header_file)

        # Run the digitization model. Allow or disallow the model to fail on some of the data, which can be helpful for debugging.
        try:
            signal = run_digitization_model(digitization_model, record, args.verbose) ### Teams: Implement this function!!!
        except:
            if args.allow_failures:
                if args.verbose:
                    print('... digitization failed.')
                signal = None
            else:
                raise

        # Run the diagnosis model. Allow or disallow the model to fail on some of the data, which can be helpful for debugging.
        try:
            diagnosis = run_diagnosis_model(diagnosis_model, record, args.verbose) ### Teams: Implement this function!!!
        except:
            if args.allow_failures:
                if args.verbose >= 2:
                    print('... diagnosis failed.')
                diagnosis = None
            else:
                raise

        # Save Challenge outputs.
        record = os.path.join(args.output_folder, records[i])
        path = os.path.split(record)[0]
        os.makedirs(path, exist_ok=True)

        header_file = get_header_file(record)
        save_text(header_file, header)

        if signal is not None:
            header = save_signal(record, signal)
        if diagnosis is not None:
            header = save_diagnosis(record, diagnosis)

    if args.verbose:
        print('Done.')

if __name__ == '__main__':
    run(get_parser().parse_args(sys.argv[1:]))