#!/usr/bin/env python

# Load libraries.
import argparse
import json
import os
import os.path
import shutil
import sys
from collections import defaultdict

from helper_code import *

# Parse arguments.
def get_parser():
    description = 'Prepare the ECG image data from ECG-Image-Kit for the Challenge.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-i', '--input_folder', type=str, required=True)
    parser.add_argument('-o', '--output_folder', type=str, required=True)
    return parser

# Find files.
def find_files(folder, extensions, remove_extension=False, sort=False):
    selected_files = set()
    for root, directories, files in os.walk(folder):
        for file in files:
            extension = os.path.splitext(file)[1]
            if extension in extensions:
                file = os.path.relpath(os.path.join(root, file), folder)
                if remove_extension:
                    file = os.path.splitext(file)[0]
                selected_files.add(file)
    if sort:
        selected_files = sorted(selected_files)
    return selected_files

# Run script.
def run(args):
    # Define variables.
    image_file_types = ['.png', '.jpg', '.jpeg']

    # Find the header files.
    records = find_records(args.input_folder)

    # Find the image files.
    image_files = find_files(args.input_folder, image_file_types)
    record_to_image_files = defaultdict(set)
    for image_file in image_files:
        root, ext = os.path.splitext(image_file)
        record = '-'.join(root.split('-')[:-1])
        basename = os.path.basename(image_file)
        record_to_image_files[record].add(basename)

    # Update the header files and copy signal files.
    for record in records:
        record_path, record_basename = os.path.split(record)
        record_image_files = record_to_image_files[record]

        # Sort the images numerically if numerical and alphanumerically otherwise.
        record_suffixes = [os.path.splitext(image_file)[0].split('-')[-1] for image_file in record_image_files]
        if all(is_number(suffix) for suffix in record_suffixes):
            record_image_files = sorted(record_image_files, key=lambda image_file: float(os.path.splitext(image_file)[0].split('-')[-1]))
        else:
            record_image_files = sorted(record_image_files)
        
        # Update the header files.
        input_header_file = os.path.join(args.input_folder, record + '.hea')
        output_header_file = os.path.join(args.output_folder, record + '.hea')

        input_header = load_text(input_header_file)
        output_header = ''
        for l in input_header.split('\n'):
            if not l.startswith(substring_images) and l:
                output_header += l + '\n'

        record_image_string = ', '.join(record_image_files)
        output_header += f'{substring_images} {record_image_string}\n'

        input_path = os.path.join(args.input_folder, record_path)
        output_path = os.path.join(args.output_folder, record_path)

        os.makedirs(output_path, exist_ok=True)

        with open(output_header_file, 'w') as f:
            f.write(output_header)

        # Copy the signal and image files if available.
        if os.path.normpath(args.input_folder) != os.path.normpath(args.output_folder):
            relative_path = os.path.split(record)[0]

            signal_files = get_signal_files(output_header_file)
            relative_path = os.path.split(record)[0]
            for signal_file in signal_files:
                input_signal_file = os.path.join(args.input_folder, relative_path, signal_file)
                output_signal_file = os.path.join(args.output_folder, relative_path, signal_file)
                if os.path.isfile(input_signal_file):
                    shutil.copy2(input_signal_file, output_signal_file)

            image_files = get_image_files(output_header_file)
            for image_file in image_files:
                input_image_file = os.path.join(args.input_folder, relative_path, image_file)
                output_image_file = os.path.join(args.output_folder, relative_path, image_file)
                if os.path.isfile(input_image_file):
                    shutil.copy2(input_image_file, output_image_file)

if __name__=='__main__':
    run(get_parser().parse_args(sys.argv[1:]))