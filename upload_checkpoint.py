import boto3
import os

s3 = boto3.client('s3')

CN_MODEL_NAME = 'shc-lm-v3'

for root, dirs, files in os.walk(CN_MODEL_NAME):
  for filename in files:
    local_path = os.path.join(root, filename)
    relative_path = os.path.relpath(local_path, CN_MODEL_NAME)
    s3_path = os.path.join('shc-lm-v3', relative_path)
    s3.upload_file(local_path, 'shc-ai-models', s3_path)
