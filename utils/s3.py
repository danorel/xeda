import os
import json
import tempfile
import zipfile

from tensorflow import keras


def _zip_dir(path, ziph):
    length = len(path)
    for root, dirs, files in os.walk(path):
        folder = root[length:]
        for file in files:
            ziph.write(os.path.join(root, file), os.path.join(folder, file))


def push_keras_model(s3fs, bucket_name, policy_name, model, model_name):
    with tempfile.TemporaryDirectory() as tempdir:
        model.save(f"{tempdir}/{model_name}")
        zipf = zipfile.ZipFile(f"{tempdir}/{model_name}.zip", "w", zipfile.ZIP_STORED)
        _zip_dir(f"{tempdir}/{model_name}", zipf)
        zipf.close()
        s3fs.put(
            f"{tempdir}/{model_name}.zip",
            f"{bucket_name}/policies/{policy_name}/final/{model_name}.zip",
        )


def push_info_json(s3fs, bucket_name, policy_name, policy_config):
    with tempfile.TemporaryDirectory() as tempdir:
        with open(f"{tempdir}/info.json", "w") as f:
            json.dump(policy_config, f)
        s3fs.put(
            f"{tempdir}/info.json", f"{bucket_name}/policies/{policy_name}/info.json"
        )


def push_pipeline_json(s3fs, bucket_name, pipeline_folder, pipeline_name, pipeline):
    with tempfile.TemporaryDirectory() as tempdir:
        with open(f"{tempdir}/pipeline.json", "w") as f:
            json.dump(pipeline, f)
        s3fs.put(
            f"{tempdir}/pipeline.json",
            f"{bucket_name}/{pipeline_folder}/{pipeline_name}.json",
        )


def pull_keras_model(s3fs, bucket_name, policy_name, model_name):
    with tempfile.TemporaryDirectory() as tempdir:
        s3fs.get(
            f"{bucket_name}/policies/{policy_name}/final/{model_name}.zip",
            f"{tempdir}/{model_name}.zip",
        )
        with zipfile.ZipFile(f"{tempdir}/{model_name}.zip") as zip_ref:
            zip_ref.extractall(f"{tempdir}/{model_name}")
        return keras.models.load_model(f"{tempdir}/{model_name}")


def pull_info_json(s3fs, bucket_name, policy_name):
    with tempfile.TemporaryDirectory() as tempdir:
        s3fs.get(
            f"{bucket_name}/policies/{policy_name}/info.json", f"{tempdir}/info.json"
        )
        with open(f"{tempdir}/info.json") as f:
            info = json.load(f)
            return info


def pull_pipeline_json(s3fs, bucket_name, pipeline_folder, pipeline_name):
    with tempfile.TemporaryDirectory() as tempdir:
        s3fs.get(
            f"{bucket_name}/{pipeline_folder}/{pipeline_name}.json",
            f"{tempdir}/pipeline.json",
        )
        with open(f"{tempdir}/pipeline.json") as f:
            pipeline = json.load(f)
            return pipeline
