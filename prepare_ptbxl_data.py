#!/usr/bin/env python

# Load libraries.
import argparse
import ast
import numpy as np
import os
import os.path
import pandas as pd
import shutil
import sys

from helper_code import *

# Parse arguments.
def get_parser():
    description = 'Prepare the PTB-XL database for use in the Challenge.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-i', '--input_folder', type=str, required=True)
    parser.add_argument('-d', '--database_file', type=str, required=True) # ptbxl_database.csv
    parser.add_argument('-s', '--statements_file', type=str, required=True) # scp_statements.csv
    parser.add_argument('-o', '--output_folder', type=str, required=True)
    return parser

# Run script.
def run(args):
    # Load the PTB-XL database.
    df = pd.read_csv(args.database_file, index_col='ecg_id')
    df.scp_codes = df.scp_codes.apply(lambda x: ast.literal_eval(x))

    # Load the SCP statements.
    dg = pd.read_csv(args.statements_file, index_col=0)

    # Identify the header files.
    records = find_records(args.input_folder)

    # Update the header files and copy the signal files.
    for record in records:

        # Extract the demographics data.
        record_path, record_basename = os.path.split(record)
        ecg_id = int(record_basename.split('_')[0])
        row = df.loc[ecg_id]

        recording_date_string = row['recording_date']
        date_string, time_string = recording_date_string.split(' ')
        yyyy, mm, dd = date_string.split('-')
        date_string = f'{dd}/{mm}/{yyyy}'

        age = row['age']
        age = cast_int_float_unknown(age)

        sex = row['sex']
        if sex == 0:
            sex = 'Male'
        elif sex == 1:
            sex = 'Female'
        else:
            sex = 'Unknown'

        height = row['height']
        height = cast_int_float_unknown(height)

        weight = row['weight']
        weight = cast_int_float_unknown(weight)

        # Extract the diagnostic superclasses.
        scp_codes = row['scp_codes']
        if 'NORM' in scp_codes:
            dx = 'Normal'
        else:
            dx = 'Abnormal'

        # Update the header file.
        input_header_file = os.path.join(args.input_folder, record + '.hea')
        output_header_file = os.path.join(args.output_folder, record + '.hea')

        input_path = os.path.join(args.input_folder, record_path)
        output_path = os.path.join(args.output_folder, record_path)

        os.makedirs(output_path, exist_ok=True)

        with open(input_header_file, 'r') as f:
            input_header = f.read()

        lines = input_header.split('\n')
        record_line = ' '.join(lines[0].strip().split(' ')[:4]) + '\n'
        signal_lines = '\n'.join(l.strip() for l in lines[1:] \
            if l.strip() and not l.startswith('#')) + '\n'
        comment_lines = '\n'.join(l.strip() for l in lines[1:] \
            if l.startswith('#') and not any((l.startswith(x) for x in ('#Age:', '#Sex:', '#Height:', '#Weight:', '#Dx:')))) + '\n'

        record_line = record_line.strip() + f' {time_string} {date_string} ' + '\n'
        signal_lines = signal_lines.strip() + '\n'
        comment_lines = comment_lines.strip() + f'#Age: {age}\n#Sex: {sex}\n#Height: {height}\n#Weight: {weight}\n#Dx: {dx}\n'

        output_header = record_line + signal_lines + comment_lines

        with open(output_header_file, 'w') as f:
            f.write(output_header)

        # Copy the signal files if the input and output folders are different.
        if os.path.normpath(args.input_folder) != os.path.normpath(args.output_folder):
            relative_path = os.path.split(record)[0]

            signal_files = get_signal_files(input_header_file)
            for signal_file in signal_files:
                input_signal_file = os.path.join(args.input_folder, relative_path, signal_file)
                output_signal_file = os.path.join(args.output_folder, relative_path, signal_file)
                if os.path.isfile(input_signal_file):
                    shutil.copy2(input_signal_file, output_signal_file)

if __name__=='__main__':
    run(get_parser().parse_args(sys.argv[1:]))