import boto3
import os

s3 = boto3.client('s3')

CN_MODEL_NAME = 'shc-lm-v3'

for root, dirs, files in os.walk(CN_MODEL_NAME):
  for filename in files:
    s3.upload_file(filename, 'shc-ai-models', f'shc-lm-v3/checkpoint/{filename}')
