#!/usr/bin/env python

# Load libraries.
import argparse
import os
import os.path
import shutil
import sys
from collections import defaultdict

from helper_code import *

# Parse arguments.
def get_parser():
    description = 'Add image filenames to header files.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-i', '--input_folder', type=str, required=True)
    parser.add_argument('-o', '--output_folder', type=str, required=True)
    return parser

# Find images.
def find_images(folder, extensions):
    images = set()
    for root, directories, files in os.walk(folder):
        for file in files:
            extension = os.path.splitext(file)[1]
            if extension in extensions:
                image = os.path.relpath(os.path.join(root, file), folder)
                images.add(image)
    images = sorted(images)
    return images

# Run script.
def run(args):
    # Find the header files.
    records = find_records(args.input_folder)

    # Find the image files.
    image_types = ['.png', '.jpg', '.jpeg']
    images = find_images(args.input_folder, image_types)
    record_to_images = defaultdict(set)
    for image in images:
        root, ext = os.path.splitext(image)
        record = '-'.join(root.split('-')[:-1])
        basename = os.path.basename(image)
        record_to_images[record].add(basename)

    # Update the header files and copy signal files.
    for record in records:
        record_path, record_basename = os.path.split(record)
        record_images = record_to_images[record]

        # Sort the images numerically if numerical and alphanumerically otherwise.
        record_suffixes = [os.path.splitext(image)[0].split('-')[-1] for image in record_images]
        if all(is_number(suffix) for suffix in record_suffixes):
            record_images = sorted(record_images, key=lambda image: float(os.path.splitext(image)[0].split('-')[-1]))
        else:
            record_images = sorted(record_images)

        # Update the header files.
        input_header_file = os.path.join(args.input_folder, record + '.hea')
        output_header_file = os.path.join(args.output_folder, record + '.hea')

        input_header = load_text(input_header_file)
        output_header = ''
        for l in input_header.split('\n'):
            if not l.startswith('#Image:') and l:
                output_header += l + '\n'

        record_image_string = ', '.join(record_images)
        output_header += f'#Image: {record_image_string}\n'

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