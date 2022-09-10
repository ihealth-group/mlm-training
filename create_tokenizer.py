from tokenizers import ByteLevelBPETokenizer
from glob import glob
import boto3
import tqdm
import os

CORPUS_BUCKET = 'shc-mlm-corpus'
CORPUS_TRAIN = 'corpus.shc'
TOKENIZER_DIR = 'shc_cn_tokenizer_bpe_52k'

s3 = boto3.client('s3')
if not os.path.exists(CORPUS_TRAIN):
  kwargs = {"Bucket": CORPUS_BUCKET, "Key": CORPUS_TRAIN}
  object_size = s3.head_object(**kwargs)["ContentLength"]
  with tqdm.tqdm(total=object_size, unit="B", unit_scale=True, desc=CORPUS_TRAIN) as pbar:
    s3.download_file(
      CORPUS_BUCKET,
      CORPUS_TRAIN,
      CORPUS_TRAIN,
      Callback=lambda bytes_transferred: pbar.update(bytes_transferred)
    )

paths = list(
  glob(CORPUS_TRAIN)
)
# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer(lowercase=False)

# Customize training
tokenizer.train(files=paths, vocab_size=52000, min_frequency=3, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

# Save files to disk
os.makedirs(TOKENIZER_DIR, exist_ok=True)
tokenizer.save_model(TOKENIZER_DIR)
