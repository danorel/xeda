import os
import json
import s3fs
import tempfile
import zipfile

from tensorflow import keras

from constants import (
    AWS_ACCESS_KEY,
    AWS_SECRET_KEY
)


def get_s3fs():
  return s3fs.S3FileSystem(key=AWS_ACCESS_KEY, secret=AWS_SECRET_KEY)


def _zip_dir(path, ziph):
    length = len(path)
    for root, dirs, files in os.walk(path):
        folder = root[length:]
        for file in files:
            ziph.write(os.path.join(root, file), os.path.join(folder, file))


def push_keras_model(bucket_name, policy_name, model, model_name):
    with tempfile.TemporaryDirectory() as tempdir:
        model.save(f"{tempdir}/{model_name}")
        zipf = zipfile.ZipFile(f"{tempdir}/{model_name}.zip", "w", zipfile.ZIP_STORED)
        _zip_dir(f"{tempdir}/{model_name}", zipf)
        zipf.close()
        s3fs = get_s3fs()
        s3fs.put(f"{tempdir}/{model_name}.zip", f"{bucket_name}/policies/{policy_name}/final/{model_name}.zip")


def push_info_json(bucket_name, policy_name, info):
    with tempfile.TemporaryDirectory() as tempdir:
        with open(f"{tempdir}/info.json", "w") as f:
            json.dump(info, f)
        s3fs = get_s3fs()
        s3fs.put(f"{tempdir}/info.json", f"{bucket_name}/policies/{policy_name}/info.json")


def push_pipeline_json(bucket_name, pipeline_name, pipeline):
    with tempfile.TemporaryDirectory() as tempdir:
        with open(f"{tempdir}/pipeline.json", "w") as f:
            json.dump(pipeline, f)
        s3fs = get_s3fs()
        s3fs.put(f"{tempdir}/pipeline.json", f"{bucket_name}/pipelines/{pipeline_name}/pipeline.json")


def pull_keras_model(bucket_name, policy_name, model_name):
    with tempfile.TemporaryDirectory() as tempdir:
        s3fs = get_s3fs()
        s3fs.get(f"{bucket_name}/policies/{policy_name}/final/{model_name}.zip", f"{tempdir}/{model_name}.zip")
        with zipfile.ZipFile(f"{tempdir}/{model_name}.zip") as zip_ref:
            zip_ref.extractall(f"{tempdir}/{model_name}")
        return keras.models.load_model(f"{tempdir}/{model_name}")


def pull_info_json(bucket_name, policy_name):
    with tempfile.TemporaryDirectory() as tempdir:
        s3fs = get_s3fs()
        s3fs.get(f"{bucket_name}/policies/{policy_name}/info.json", f"{tempdir}/info.json")
        with open(f"{tempdir}/info.json") as f:
            info = json.load(f)
            return info
