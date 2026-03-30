"""Utility functions for resource management."""

import logging
import os
import zipfile
from io import BytesIO

import boto3

s3_client = boto3.client('s3')

log = logging.getLogger(__name__)


def get_and_unzip_resource(bucket_name: str, zip_file_key: str, extract_dir_name: str) -> None:
    """Downloads a zip file from S3 and extacts it to a local directory.

    Checks if the directory is already populated to avoid re-work.

    Args:
            bucket_name: The name of the S3 bucket.
            zip_file_key: The S3 key for the zip file.
            extract_dir_name: The local directory to extract files into.

    Raises:
            Exception: If download or extraction fails.
    """
    marker_file = os.path.join(extract_dir_name, '.unzipped_complete')
    if os.path.exists(marker_file):
        log.info(f'Resources already unzipped in {extract_dir_name}.')
        return

    log.info(f'Downloading {zip_file_key}')
    os.makedirs(extract_dir_name, exist_ok=True)

    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=zip_file_key)
        zip_data = response['Body'].read()

        log.info(f'Unzipping {zip_file_key} to {extract_dir_name}.')
        with zipfile.ZipFile(BytesIO(zip_data)) as z:
            z.extractall(extract_dir_name)

        with open(marker_file, 'w') as f:
            f.write('success')

        log.info(f'Successfully unzipped {zip_file_key}.')
    except Exception as e:
        log.error(f'Failed to download or unzip {zip_file_key}', exc_info=True)
        raise e
