#!/usr/bin/env python

# Please do *not* edit this script. Changes will be discarded so that we can train the models consistently.

# This file contains functions for training models for the Challenge. You can run it as follows:
#
#   python train_model.py -d data -m model -v
#
# where 'data' is a folder containing the Challenge data, 'model' is a folder for saving your models, and , and -v is an optional
# verbosity flag.

import argparse
import sys

from helper_code import *
from team_code import train_models

# Parse arguments.
def get_parser():
    description = 'Train the Challenge models.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-d', '--data_folder', type=str, required=True)
    parser.add_argument('-m', '--model_folder', type=str, required=True)
    parser.add_argument('-v', '--verbose', action='store_true')
    return parser

# Run the code.
def run(args):
    train_models(args.data_folder, args.model_folder, args.verbose) ### Teams: Implement this function!!!

if __name__ == '__main__':
    run(get_parser().parse_args(sys.argv[1:]))