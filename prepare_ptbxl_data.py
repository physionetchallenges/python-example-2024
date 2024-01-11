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
    description = 'Prepare PTB-XL data.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-i', '--input_folder', type=str, required=True)
    parser.add_argument('-d', '--database_file', type=str, required=True) # ptbxl_database.csv
    parser.add_argument('-s', '--statements_file', type=str, required=True) # scp_statements.csv
    parser.add_argument('-o', '--output_folder', type=str, required=True)
    return parser

# Run script.
def run(args):
    # Identify PTB-XL superclasses.
    df = pd.read_csv(args.statements_file, index_col=0)
    subclass_to_superclass = dict()
    for i, row in df.iterrows():
        if row['diagnostic'] == 1:
            subclass = i.strip()
            superclass = row['diagnostic_class'].strip()
            subclass_to_superclass[subclass] = superclass

    # For the PTB-XL database, assign superclasses to subclasses; commands from PhysioNet project documentation.
    def assign_superclass(subclasses):
        superclasses = list()
        for subclass in subclasses:
            if subclass in subclass_to_superclass:
                superclass = subclass_to_superclass[subclass]
                if superclass not in superclasses:
                    superclasses.append(superclass)
        for superclass in superclasses:
            if superclass.startswith(' ') or superclass.endswith(' '):
                print(superclass)
        return superclasses

    # Apply PTB-XL superclasses.
    dg = pd.read_csv(args.database_file, index_col='ecg_id')
    dg.scp_codes = dg.scp_codes.apply(lambda x: ast.literal_eval(x))
    dg['diagnostic_superclass'] = dg.scp_codes.apply(assign_superclass)

    # Identify header files.
    records = find_records(args.input_folder)

    # Update header files and copy signal files.
    for record in records:
        record_path, record_basename = os.path.split(record)
        ecg_id = int(record_basename.split('_')[0])
        row = dg.loc[ecg_id]

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

        dx = ', '.join(row['diagnostic_superclass'])

        if dx:
            input_header_file = os.path.join(args.input_folder, record + '.hea')
            output_header_file = os.path.join(args.output_folder, record + '.hea')

            input_path = os.path.join(args.input_folder, record_path)
            output_path = os.path.join(args.output_folder, record_path)

            os.makedirs(output_path, exist_ok=True)

            with open(input_header_file, 'r') as f:
                input_header = f.read()

            record_line = input_header.split('\n')[0].strip() + '\n'
            signal_lines = '\n'.join(l.strip() for l in input_header.split('\n')[1:] if l.strip() and not l.startswith('#')) + '\n'
            comment_lines = '\n'.join(l.strip() for l in input_header.split('\n')[1:] if l.startswith('#'))

            record_line = record_line.strip() + f' {time_string} {date_string} ' + '\n'
            signal_lines = signal_lines
            comment_lines = comment_lines.strip() + f'#Age: {age}\n#Sex: {sex}\n#Height: {height}\n#Weight: {weight}\n#Dx: {dx}\n'

            output_header = record_line + signal_lines + comment_lines

            with open(output_header_file, 'w') as f:
                f.write(output_header)

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