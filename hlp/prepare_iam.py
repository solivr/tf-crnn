#!/usr/bin/env python
__author__ = "solivr"
__license__ = "GPL"

from taputapu.databases import iam
import os
from glob import glob
from .string_data_manager import tf_crnn_label_formatting
from .alphabet_helpers import generate_alphabet_file
import click


@click.command()
@click.option('--download_dir')
@click.option('--generated_data_dir')
def prepare_iam_data(download_dir: str,
                     generated_data_dir: str):

    # Download data
    print('Starting downloads...')
    iam.download(download_dir)

    # Extract archives
    print('Starting extractions...')
    iam.extract(download_dir)

    print('Generating files for the experiment...')
    # Generate splits (same format as ascii files)
    export_splits_dir = os.path.join(generated_data_dir, 'generated_splits')
    os.makedirs(export_splits_dir, exist_ok=True)

    iam.generate_splits_txt(os.path.join(download_dir, 'ascii', 'lines.txt'),
                            os.path.join(download_dir, 'largeWriterIndependentTextLineRecognitionTask'),
                            export_splits_dir)

    # Generate csv from .txt splits files
    export_csv_dir = os.path.join(generated_data_dir, 'generated_csv')
    os.makedirs(export_csv_dir, exist_ok=True)

    for file in glob(os.path.join(export_splits_dir, '*')):
        export_basename = os.path.basename(file).split('.')[0]
        iam.create_experiment_csv(file,
                                  os.path.join(download_dir, 'lines'),
                                  os.path.join(export_csv_dir, export_basename + '.csv'),
                                  False,
                                  True)

    # Format string label to tf_crnn formatting
        for csv_filenames in glob(os.path.join(export_csv_dir, '*')):
            tf_crnn_label_formatting(csv_filenames)

    # Generate alphabet
    alphabet_dir = os.path.join(generated_data_dir, 'generated_alphabet')
    os.makedirs(alphabet_dir, exist_ok=True)

    generate_alphabet_file(glob(os.path.join(export_csv_dir, '*')),
                           os.path.join(alphabet_dir, 'iam_alphabet_lookup.json'))


if __name__ == '__main__':
    prepare_iam_data()
