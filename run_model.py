#!/usr/bin/env python

# Please do *not* edit this script. Changes will be discarded so that we can run the trained models consistently.

# This file contains functions for running models for the Challenge. You can run it as follows:
#
#   python run_model.py -d data -m model -o outputs -v
#
# where 'data' is a folder containing the Challenge data, 'models' is a folder containing the your trained models, 'outputs' is a
# folder for saving your models' outputs, and -v is an optional verbosity flag.

import argparse
import os
import sys

from helper_code import *
from team_code import load_models, run_models

# Parse arguments.
def get_parser():
    description = 'Run the trained Challenge models.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-d', '--data_folder', type=str, required=True)
    parser.add_argument('-m', '--model_folder', type=str, required=True)
    parser.add_argument('-o', '--output_folder', type=str, required=True)
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-f', '--allow_failures', action='store_true')
    return parser

# Run the code.
def run(args):
    # Load the models.
    if args.verbose:
        print('Loading the Challenge model...')

    # You can use these functions to perform tasks, such as loading your models, that you only need to perform once.
    digitization_model, classification_model = load_models(args.model_folder, args.verbose) ### Teams: Implement this function!!!

    # Find the Challenge data.
    if args.verbose:
        print('Finding the Challenge data...')

    records = find_records(args.data_folder)
    num_records = len(records)

    if num_records==0:
        raise Exception('No data were provided.')

    # Create a folder for the Challenge outputs if it does not already exist.
    os.makedirs(args.output_folder, exist_ok=True)

    # Run the team's models on the Challenge data.
    if args.verbose:
        print('Running the Challenge model(s) on the Challenge data...')

    # Iterate over the records.
    for i in range(num_records):
        if args.verbose:
            width = len(str(num_records))
            print(f'- {i+1:>{width}}/{num_records}: {records[i]}...')

        data_record = os.path.join(args.data_folder, records[i])
        output_record = os.path.join(args.output_folder, records[i])

        # Run the models. Allow or disallow the models to fail on some of the data, which can be helpful for debugging.
        try:
            signals, labels = run_models(data_record, digitization_model, classification_model, args.verbose) ### Teams: Implement this function!!!
        except:
            if args.allow_failures:
                if args.verbose:
                    print('... failed.')
                signals = None
                labels = None
            else:
                raise

        # Save Challenge outputs.
        output_path = os.path.split(output_record)[0]
        os.makedirs(output_path, exist_ok=True)

        data_header = load_header(data_record)
        save_header(output_record, data_header)

        if signals is not None:
            comments = [l for l in data_header.split('\n') if l.startswith('#')]
            save_signals(output_record, signals, comments)
        if labels is not None:
            save_labels(output_record, labels)

    if args.verbose:
        print('Done.')

if __name__ == '__main__':
    run(get_parser().parse_args(sys.argv[1:]))