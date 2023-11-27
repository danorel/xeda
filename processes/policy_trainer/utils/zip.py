import os
import tempfile
import zipfile

from tensorflow import keras

def zip_dir(path, ziph):
  length = len(path)
  for root, dirs, files in os.walk(path):
    folder = root[length:]
    for file in files:
      ziph.write(os.path.join(root, file), os.path.join(folder, file))


def push_keras_model(s3, bucket_name, policy_name, model, model_name):
  with tempfile.TemporaryDirectory() as tempdir:
    model.save(f"{tempdir}/{model_name}")
    zipf = zipfile.ZipFile(f"{tempdir}/{model_name}.zip", "w", zipfile.ZIP_STORED)
    zip_dir(f"{tempdir}/{model_name}", zipf)
    zipf.close()
    s3.upload_file(
        f"{tempdir}/{model_name}.zip",
        bucket_name, 
        f"policies/{policy_name}/final/{model_name}.zip"
    )


def pull_keras_model(s3, bucket_name, model_name):
  with tempfile.TemporaryDirectory() as tempdir:
    s3.get(f"{bucket_name}/{model_name}.zip", f"{tempdir}/{model_name}.zip")
    with zipfile.ZipFile(f"{tempdir}/{model_name}.zip") as zip_ref:
        zip_ref.extractall(f"{tempdir}/{model_name}")
    return keras.models.load_model(f"{tempdir}/{model_name}")
