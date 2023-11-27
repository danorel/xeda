import os

from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv(".env"))

BUCKET_NAME = os.environ.get("S3_BUCKET")
