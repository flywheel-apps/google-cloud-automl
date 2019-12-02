#!/usr/bin/env python3
import base64
import json
import logging
import os
import pprint
import shutil
import subprocess
import sys
import tempfile
import zipfile

import flywheel
from google.cloud.automl_v1beta1 import PredictionServiceClient

log = logging.getLogger('automl.predict')
logging.basicConfig(
    format='%(asctime)s %(name)15.15s %(levelname)4.4s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)]
)


# TODO limit scopes ASAP (scopes are now available)
SCOPES = ['https://www.googleapis.com/auth/cloud-platform']


def main(context):
    # get inputs
    service_account_path = context.get_input_path('service_account')
    training_result_path = context.get_input_path('training_result')
    input_path = context.get_input_path('input')

    training_result = json.load(open(training_result_path))

    # get config
    gcp_project = training_result['gcp_project']
    aml_location = training_result['aml_location']
    aml_dataset = training_result['aml_dataset']
    aml_model = training_result['aml_model']
    img_frame_selection = context.config.get('img_frame_selection')
    img_slice_selection = context.config.get('img_slice_selection')
    score_threshold = context.config.get('score_threshold')

    # create predict client
    predict_client = PredictionServiceClient.from_service_account_json(service_account_path)

    # extract the slice and upload for prediction
    log.info('Extracting prediction slice')
    stage_dir = tempfile.mkdtemp()
    image_path = extract_prediction_image(input_path, stage_dir, img_frame_selection, img_slice_selection)
    image = open(stage_dir + '/' + image_path, 'rb').read()

    log.info('Running AutoML Vision prediction')
    payload = {'image': {'image_bytes': image}}
    classification = predict_client.predict(aml_model, payload).payload

    # TODO handle multi-label
    score = classification[0].classification.score
    label_key = classification[0].display_name
    label = training_result['label_map'][label_key]
    log.info('Prediction: (confidence={}):\n{}'.format(score, pprint.pformat(label)))

    if score < score_threshold:
        log.error('Confidence score is below threshold')
        sys.exit(1)

    file_meta = {'name': os.path.basename(input_path)}
    for key, value in label.items():
        node = file_meta
        key_parts = key.replace('file.', '').split('.')
        for part in key_parts[:-1]:
            node[part] = {}
            node = node[part]
        node[key_parts[-1]] = value
    with open('output/.metadata.json', 'wt') as f:
        # TODO enable for non-acqisition files?
        json.dump({'acquisition': {'files': [file_meta]}}, f)


def extract_prediction_image(file_path, dest_dir, img_frame_selection, img_slice_selection):
    """Return a single jpg filename extracted from a given [dicom.]zip/nii.gz as downloaded from fw."""
    # handle types - unzip dicoms, use nifti as is (.gz handled by med2image)
    log.info('Exctracting image from {} (frame: {}, slice: {})'.format(
        os.path.basename(file_path), img_frame_selection, img_slice_selection))
    if file_path.endswith('.zip'):  # handle UI (.zip) and reaper uploads (.dicom.zip)
        dicom_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(file_path) as zf:
            zf.extractall(dicom_dir)
        # med2image discovers all related dicoms in the same dir - use 1st dcm
        input_file = dicom_dir + '/' + list(sorted(os.listdir(dicom_dir)))[0]
        if os.path.isdir(input_file):
            input_file += '/' + list(sorted(os.listdir(input_file)))[0]
        output_stem = '%SOPInstanceUID'
    elif file_path.endswith('nii.gz'):
        input_file = file_path
        output_stem = os.path.basename(input_file).replace('.nii.gz', '')
    else:
        # TODO handle standalone images
        raise ValueError('Unhandled file type: ' + file_path)

    # extract pngs from dicoms/niftis
    # TODO replace subprocess with direct call for flexibility
    temp_dir = tempfile.mkdtemp()
    command = [
        'med2image',
        '--inputFile', input_file,
        '--frameToConvert', {'mid': 'm'}.get(img_frame_selection, img_frame_selection),
        '--sliceToConvert', {'mid': 'm'}.get(img_slice_selection, img_slice_selection),
        '--outputDir', temp_dir,
        '--outputFileStem', output_stem,
        '--outputFileType', 'jpg',
    ]
    log.info('Running `{}`'.format(' '.join(command)))
    try:
        output = subprocess.check_output(command, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as exc:
        log.error('med2image exited with ' + str(exc.returncode))
        log.error(exc.output.decode('utf8'))
        sys.exit(1)
    image = list(sorted(os.listdir(temp_dir)))[0]
    os.rename(temp_dir + '/' + image, dest_dir + '/' + image)
    log.info('Extracted image')

    # cleanup extracted dicoms
    if file_path.endswith('.zip'):
        shutil.rmtree(dicom_dir)

    # cleanup unused images left in outdir
    shutil.rmtree(temp_dir)

    return image


if __name__ == '__main__':
    with flywheel.GearContext() as context:
        context.init_logging()
        context.log_config()
        main(context)
