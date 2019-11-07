#!/usr/bin/env python3
import collections
import json
import logging
import multiprocessing
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import zipfile

import flywheel
import google.cloud.exceptions
import pystache
from google.cloud import storage
from google.cloud.automl_v1beta1 import AutoMlClient
import shortuuid

log = logging.getLogger('flywheel:automl-train')
logging.basicConfig(
    format='%(asctime)s %(name)15.15s %(levelname)4.4s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)]
)

# global vars keeping track of labels encountered in the training set via extract_label()
LABEL_KEYS = []
LABEL_VALS = []


SUMMARY_TEMPLATE = '''
### Google AutoML training results
___

- Total images: {{total_images}}
- Precision:    {{avg_precision}}
- Train budget: {{train_budget}}
- Dataset:      [{{aml_dataset_displayname}}](https://cloud.google.com/automl/ui/vision/datasets/details?project={{gcp_project}}&dataset={{aml_dataset}})
- Model:        [{{aml_model_displayname}}](https://cloud.google.com/automl/ui/vision/datasets/evaluate?project={{gcp_project}}&dataset={{aml_dataset}}&model={{aml_model}})

**Label mapping**

{{#label_map}}
{{key}}: {{{value}}}
{{/label_map}}
'''


def main(context):
    # dev workaround for accessing docker.local - set own host ip
    # ip -o route get to 8.8.8.8 | sed -n 's/.*src \([0-9.]\+\).*/\1/p'
    host_ip = '192.168.50.189'
    with open('/etc/hosts', 'a') as f:
        f.write(host_ip + '\tdocker.local.flywheel.io\n')

    # get inputs
    service_account_path = context.get_input_path('service_account')
    training_set_path = context.get_input_path('training_set')

    # load service account and training set
    service_account = json.load(open(service_account_path))
    training_set = json.load(open(training_set_path))

    # get config
    gcp_project = service_account['project_id']
    aml_location = context.config.get('aml_location')
    gcs_prefix = context.config.get('gcs_prefix')
    img_frame_selection = context.config.get('img_frame_selection')
    img_slice_selection = context.config.get('img_slice_selection')
    train_budget = context.config.get('train_budget')

    # generate aml_dataset_displayname and aml_model_displayname
    train_id = shortuuid.uuid()[:6]
    aml_dataset_displayname = 'fw_aml_dataset_' + train_id
    aml_model_displayname = 'fw_aml_model_' + train_id

    # download niftis/dicoms and stage extracted slices
    log.info('Downloading files from the training set and extracting slices')
    input_map = collections.OrderedDict()
    stage_dir = tempfile.mkdtemp()
    for train_file in training_set['files']:
        cont_id, name = train_file['parent_id'], train_file['name']
        file_path = '/tmp/' + cont_id + '_' + name
        context.client.download_file_from_container(cont_id, name, file_path)
        label = extract_label(train_file['labels'])
        for image in extract_training_images(file_path, stage_dir, img_frame_selection, img_slice_selection):
            input_map[image] = label
        os.remove(file_path)

    storage_client = storage.Client.from_service_account_json(service_account_path)
    if not gcs_prefix:
        bucket_name = '{}-vcm'.format(gcp_project)
        try:
            storage_client.create_bucket(bucket_name)
        except google.cloud.exceptions.Conflict:
            pass
        gcs_prefix = 'gs://{}'.format(bucket_name)
    gcs_prefix = '{}/{}'.format(gcs_prefix.rstrip('/'), train_id)

    # generate aml input csv with (gs:// image path, label) pairs
    input_csv = 'fw_aml_input_' + train_id + '.csv'
    with open(stage_dir + '/' + input_csv, 'wt') as f:
        for image_name, label in input_map.items():
            f.write('{}/{},{}\n'.format(gcs_prefix, image_name, label))

    # upload staged images and input.csv in parallel
    log.info('Uploading extracted slices to ' + gcs_prefix)
    staged_files = sorted(os.listdir(stage_dir))
    pool = multiprocessing.Pool()
    pool.map(upload_file, [(service_account_path, gcs_prefix, stage_dir + '/' + name) for name in staged_files])

    # create automl client
    automl_client = AutoMlClient.from_service_account_json(service_account_path)

    # create aml dataset, import data using input.csv and train new model
    log.info('Creating AutoML dataset with name ' + aml_dataset_displayname)
    ds_parent = automl_client.location_path(gcp_project, aml_location)
    dataset = {
        'display_name': aml_dataset_displayname,
        'image_classification_dataset_metadata': {
            'classification_type': 'MULTICLASS'
        },
    }
    aml_dataset = automl_client.create_dataset(ds_parent, dataset)
    log.info('Created AutoML dataset ' + aml_dataset.display_name)

    log.info('Importing training set into dataset')
    input_csv_gs_path = '{}/{}'.format(gcs_prefix, input_csv)
    import_op = automl_client.import_data(aml_dataset.name, {'gcs_source': {'input_uris': [input_csv_gs_path]}})
    import_op.result()
    log.info('Imported training set')

    log.info('Cleanup storage bucket')
    bucket, _, prefix = gcs_prefix.replace('gs://', '').partition('/')
    bucket = storage_client.get_bucket(bucket)
    blobs = bucket.list_blobs(prefix=prefix)
    for blob in blobs:
        blob.delete()

    log.info('Creating AutoML model with name ' + aml_model_displayname)
    # TODO handle multi-label
    model_payload = {
        'display_name': aml_model_displayname,
        'dataset_id': aml_dataset.name.split('/')[-1],
        'image_classification_model_metadata': {'train_budget': train_budget}
    }
    train_op = automl_client.create_model(ds_parent, model_payload)
    resp = train_op.result()
    model_full_id = resp.name
    # List all the model evaluations in the model by applying filter.
    aml_model_evaluation = list(automl_client.list_model_evaluations(model_full_id))

    # store model, inputs used and label map alongside training-set (used by predictor)
    log.info('Saving training results')
    training_result_file = 'training_result_' + train_id + '.json'
    training_result = {
        'training_set': training_set,
        'train_budget': train_budget,
        'gcs_prefix': gcs_prefix,
        'gcp_project': gcp_project,
        'aml_location': aml_location,
        'aml_dataset': aml_dataset.name,
        'aml_dataset_displayname': aml_dataset.display_name,
        'aml_model': model_full_id,
        'aml_model_displayname': aml_model_displayname,
        'input_map': input_map,
        'label_map': {str(i): label for i, label in enumerate(LABEL_VALS)},
    }
    with open('output/' + training_result_file, 'wt') as f:
        json.dump(training_result, f, indent=4, sort_keys=True)

    # create a viewer-friendly markdown summary using pystache
    summary_file = 'training_result_' + train_id + '.md'
    summary_vars = training_result.copy()
    last_model_eval = aml_model_evaluation[-1]
    avg_precision = last_model_eval.classification_evaluation_metrics.au_prc
    summary_vars.update({
        'total_images': len(input_map),
        'train_budget': '{} hour{}'.format(train_budget, '' if train_budget == 1 else 's'),
        'avg_precision': '{:.3f} ({:.0f}%)'.format(avg_precision, avg_precision * 100),
        'label_map': [{'key': str(i), 'value': label} for i, label in enumerate(LABEL_VALS)],
    })
    with open('output/' + summary_file, 'wt') as f:
        f.write(pystache.render(SUMMARY_TEMPLATE, summary_vars))

    # create metadata json
    with open('output/.metadata.json', 'wt') as f:
        json.dump({'files': [
            {'name': training_result, 'type': 'source code'},
            {'name': summary_file, 'type': 'markdown'},
        ]}, f)


def extract_label(obj):
    """
    Return numeric string key (usable as an aml label) for any* given object.

    Calling with an equal object twice, returns the same key. Uses module
    level variables LABEL_KEYS and LABEL_VALS to keep track of labels seen.
    """
    if isinstance(obj, (dict, list)):
        # TODO consider "intended unordered" lists
        label_key = json.dumps(obj, separators=(',', ':'))
    else:
        label_key = obj
    if label_key not in LABEL_KEYS:
        LABEL_KEYS.append(label_key)
        LABEL_VALS.append(obj)
    return str(LABEL_KEYS.index(label_key))


def extract_training_images(file_path, dest_dir, img_frame_selection, img_slice_selection):
    """Return a list of jpg filenames extracted from a given [dicom.]zip/nii.gz as downloaded from fw."""
    # handle types - unzip dicoms, use nifti as is (.gz handled by med2image)
    log.info('Exctracting images from {} (frame: {}, slice: {})'.format(
        os.path.basename(file_path), img_frame_selection, img_slice_selection))
    if file_path.endswith('.zip'):  # handle UI (.zip) and reaper uploads (.dicom.zip)
        dicom_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(file_path) as zf:
            zf.extractall(dicom_dir)
        # med2image discovers all related dicoms in the same dir - use 1st dcm
        input_file = dicom_dir + '/' + list(sorted(os.listdir(dicom_dir)))[0]
        if os.path.isdir(input_file):
            input_file += '/' + list(sorted(os.listdir(input_file)))[0]
        output_stem = os.path.basename(file_path).replace('.dicom.zip', '').replace('.zip', '')
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
        '--frameToConvert', {'all': '-1', 'mid': 'm'}.get(img_frame_selection, img_frame_selection),
        '--sliceToConvert', {'all': '-1', 'mid': 'm'}.get(img_slice_selection, img_slice_selection),
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
    images = list(sorted(os.listdir(temp_dir)))
    for image in images:
        os.rename(temp_dir + '/' + image, dest_dir + '/' + image)
    log.info('Extracted {} image(s)'.format(len(images)))

    # cleanup extracted dicoms
    if file_path.endswith('.zip'):
        shutil.rmtree(dicom_dir)

    return images


def upload_file(args_tuple):
    """Upload a single file to object storage."""
    service_account_path, gcs_prefix, file_path = args_tuple
    bucket, _, prefix = gcs_prefix.replace('gs://', '').partition('/')
    storage_client = storage.Client.from_service_account_json(service_account_path)
    bucket = storage_client.get_bucket(bucket)
    object_name = prefix + '/' + os.path.basename(file_path).lstrip('/')
    blob = bucket.blob(object_name)
    return blob.upload_from_filename(filename=file_path)


if __name__ == '__main__':
    with flywheel.GearContext() as context:
        context.init_logging()
        context.log_config()
        main(context)
