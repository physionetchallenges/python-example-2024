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
    parser.add_argument('-pd', '--ptbxl_database_file', type=str, required=True) # ptbxl_database.csv
    parser.add_argument('-pm', '--ptbxl_mapping_file', type=str, required=True) # scp_statements.csv
    parser.add_argument('-sd', '--sl_database_file', type=str, required=True) # 12sl_statements.csv
    parser.add_argument('-sm', '--sl_mapping_file', type=str, required=True) # 12slv23ToSNOMED.csv
    parser.add_argument('-o', '--output_folder', type=str, required=True)
    return parser

# Run script.
def run(args):
    # Assign each class to a superclass; these commands were adapted from the PhysioNet project documentation.
    df_ptbxl_mapping = pd.read_csv(args.ptbxl_mapping_file, index_col=0)
    subclass_to_superclass = dict()
    for i, row in df_ptbxl_mapping.iterrows():
        if row['diagnostic'] == 1:
            subclass_to_superclass[i] = row['diagnostic_class']

    def assign_superclass(subclasses):
        superclasses = list()
        for subclass in subclasses:
            if subclass in subclass_to_superclass:
                superclass = subclass_to_superclass[subclass]
                if superclass not in superclasses:
                    superclasses.append(superclass)
        return superclasses

    # Load the PTB-XL labels.
    df_ptbxl_database = pd.read_csv(args.ptbxl_database_file, index_col='ecg_id')
    df_ptbxl_database.scp_codes = df_ptbxl_database.scp_codes.apply(lambda x: ast.literal_eval(x))

    # Map the PTB-XL classes to superclasses.
    df_ptbxl_database['diagnostic_superclass'] = df_ptbxl_database.scp_codes.apply(assign_superclass)

    # Load the 12SL labels.
    df_sl_database = pd.read_csv(args.sl_database_file, index_col='ecg_id')

    # Map the 12SL classes to the PTB-XL classes for the following acute myocardial infarction (MI) classes; PTB-XL does not include
    # a separate acute MI class.
    df_sl_mapping = pd.read_csv(args.sl_mapping_file, index_col='StatementNumber')

    acute_mi_statements = set([821, 822, 823, 827, 829, 902, 903, 904, 963, 964, 965, 966, 967, 968])
    acute_mi_classes = set()
    for statement in acute_mi_statements:
        if statement in df_sl_mapping.index:
            acute_mi_classes.add(df_sl_mapping.loc[statement]['Acronym'])

    # Identify the header files.
    records = find_records(args.input_folder)

    # Update the header files to include demographics data and labels and copy the signal files unchanged.
    for record in records:

        # Extract the demographics data.
        record_path, record_basename = os.path.split(record)
        ecg_id = int(record_basename.split('_')[0])
        row = df_ptbxl_database.loc[ecg_id]

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

        scp_code_dict = row['scp_codes']
        scp_codes = [scp_code for scp_code in scp_code_dict if scp_code_dict[scp_code] >= 0]
        superclasses = row['diagnostic_superclass']

        if ecg_id in df_sl_database.index:
            sl_codes = df_sl_database.loc[ecg_id]['statements']
        else:
            sl_codes = list()

        labels = list()
        if 'NORM' in superclasses:
            labels.append('NORM')
        if any(c in sl_codes for c in acute_mi_classes):
            labels.append('Acute MI')
        if 'MI' in superclasses and not any(c in sl_codes for c in acute_mi_classes):
            labels.append('Old MI')      
        if 'STTC' in superclasses:
            labels.append('STTC')
        if 'CD' in superclasses:
            labels.append('CD')
        if 'HYP' in superclasses:
            labels.append('HYP')
        if 'PAC' in scp_codes:
            labels.append('PAC')
        if 'PVC' in scp_codes:
            labels.append('PVC')
        if 'AFIB' in scp_codes or 'AFLT' in scp_codes:
            labels.append('AFIB/AFL')
        if 'STACH' in scp_codes or 'SVTAC' in scp_codes or 'PSVT' in scp_codes:
            labels.append('TACHY')
        if 'SBRAD' in scp_codes:
            labels.append('BRADY') 
        labels = ', '.join(labels)

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
            if l.startswith('#') and not any((l.startswith(x) for x in ('# Age:', '# Sex:', '# Height:', '# Weight:', f'{substring_labels}')))) + '\n'

        record_line = record_line.strip() + f' {time_string} {date_string} ' + '\n'
        signal_lines = signal_lines.strip() + '\n'
        comment_lines = comment_lines.strip() + f'# Age: {age}\n# Sex: {sex}\n# Height: {height}\n# Weight: {weight}\n{substring_labels} {labels}\n'

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