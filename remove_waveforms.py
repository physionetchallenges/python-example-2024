#!/usr/bin/env python

# Load libraries.
import argparse
import os
import os.path
import shutil
import sys

from helper_code import *

# Parse arguments.
def get_parser():
    description = 'Remove waveforms (and, optionally, diagnoses).'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-i', '--input_folder', type=str, required=True)
    parser.add_argument('-d', '--exclude_diagnoses', action='store_true')
    parser.add_argument('-o', '--output_folder', type=str, required=True)
    return parser

# Run script.
def run(args):
    # Identify header files.
    records = find_records(args.input_folder)

    # Update header files and copy signal files.
    for record in records:
        record_path, record_basename = os.path.split(record)

        input_header_file = os.path.join(args.input_folder, record + '.hea')
        output_header_file = os.path.join(args.output_folder, record + '.hea')

        input_header = load_text(input_header_file)
        output_header = ''

        num_signals = get_num_signals(input_header)
        for i, l in enumerate(input_header.split('\n')):
            arrs = l.split(' ')
            if i == 0:
                output_header += l + '\n'
            elif 1 <= i <= num_signals:
                output_header += ' '.join([arrs[0], arrs[1], arrs[2], arrs[3], arrs[4], '', '', '', arrs[8]]) + '\n'
            elif l.startswith('#Dx:') and not args.exclude_diagnoses:
                output_header += l + '\n'
            elif l.startswith('#Image:'):
                output_header += l + '\n'

        input_path = os.path.join(args.input_folder, record_path)
        output_path = os.path.join(args.output_folder, record_path)

        os.makedirs(output_path, exist_ok=True)

        with open(output_header_file, 'w') as f:
            f.write(output_header)

        if os.path.normpath(args.input_folder) != os.path.normpath(args.output_folder):
            relative_path = os.path.split(record)[0]

            image_files = get_image_files(input_header_file)
            for image_file in image_files:
                input_image_file = os.path.join(args.input_folder, relative_path, image_file)
                output_image_file = os.path.join(args.output_folder, relative_path, image_file)
                if os.path.isfile(input_image_file):
                    shutil.copy2(input_image_file, output_image_file)

if __name__=='__main__':
    run(get_parser().parse_args(sys.argv[1:]))