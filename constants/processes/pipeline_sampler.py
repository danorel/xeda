import os

PIPELINE_MIN_SIZE = int(os.getenv("PIPELINE_MIN_SIZE", "4"))
PIPELINE_MAX_SIZE = int(os.getenv("PIPELINE_MAX_SIZE", "10"))
