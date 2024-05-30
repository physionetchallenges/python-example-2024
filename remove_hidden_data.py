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
    description = 'Remove hidden data.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-i', '--input_folder', type=str, required=True)
    parser.add_argument('-w', '--include_waveforms', action='store_true')
    parser.add_argument('-l', '--include_labels', action='store_true')
    parser.add_argument('-m', '--include_images', action='store_true')
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
                output_header += ' '.join(arrs[:4]) + '\n'
            elif 1 <= i <= num_signals:
                output_header += ' '.join(arrs[:5] + ['', '', ''] + [arrs[8]]) + '\n'
            elif l.startswith(substring_labels):
                if args.include_labels:
                    output_header += l + '\n'
            elif l.startswith(substring_images):
                if args.include_images:                
                    output_header += l + '\n'

        input_path = os.path.join(args.input_folder, record_path)
        output_path = os.path.join(args.output_folder, record_path)

        os.makedirs(output_path, exist_ok=True)

        with open(output_header_file, 'w') as f:
            f.write(output_header)

        relative_path = os.path.split(record)[0]

        if args.include_waveforms and os.path.normpath(args.input_folder) != os.path.normpath(args.output_folder):
            signal_files = get_signal_files(input_header_file)
            for signal_file in signal_files:
                input_signal_file = os.path.join(args.input_folder, relative_path, signal_file)
                output_signal_file = os.path.join(args.output_folder, relative_path, signal_file)
                if os.path.isfile(input_signal_file):
                    shutil.copy2(input_signal_file, output_signal_file)
        elif not args.include_waveforms and os.path.normpath(args.input_folder) == os.path.normpath(args.output_folder):
            signal_files = get_signal_files(input_header_file)
            for signal_file in signal_files:
                input_signal_file = os.path.join(args.input_folder, relative_path, signal_file)
                output_signal_file = os.path.join(args.output_folder, relative_path, signal_file)
                if os.path.isfile(output_signal_file):
                    os.remove(output_signal_file)

        if args.include_images:
            image_files = get_image_files(input_header_file)
            for image_file in image_files:
                input_image_file = os.path.join(args.input_folder, relative_path, image_file)
                output_image_file = os.path.join(args.output_folder, relative_path, image_file)
                if os.path.isfile(input_image_file):
                    shutil.copy2(input_image_file, output_image_file)

if __name__=='__main__':
    run(get_parser().parse_args(sys.argv[1:]))