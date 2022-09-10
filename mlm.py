from transformers import set_seed
from datasets import load_dataset
from pathlib import Path
import tarfile
import boto3
import wandb
import tqdm
import os

from transformers import (
  AutoConfig,
  AutoModelForMaskedLM,
  RobertaTokenizerFast,
  DataCollatorForLanguageModeling,
  Trainer,
  TrainingArguments
)

CN_MODEL_NAME = Path('shc-lm-v3.1')
CORPUS_BUCKET = 'shc-mlm-corpus'
ROOT_BUCKET = 'shc-ai-models'
CORPUS_TRAIN = 'corpus.shc'
CORPUS_DEV = 'corpus_dev.shc'
BERT_MODEL_NAME = Path('shc-lm-v3')
PROJECT_NAME = 'shc-lm'
TOKENIZER_DIR = 'shc_cn_tokenizer_bpe_52k'

wandb.login()

w_run = wandb.init(project=PROJECT_NAME, notes="LM Model from XLM Roberta Base")

s3 = boto3.client('s3')

if not BERT_MODEL_NAME.exists():
  kwargs = {"Bucket": 'shc-ai-models', "Key": f'language_model/{BERT_MODEL_NAME}.tar.gz'}
  object_size = s3.head_object(**kwargs)["ContentLength"]
  with tqdm.tqdm(total=object_size, unit="B", unit_scale=True, desc=f'language_model/{BERT_MODEL_NAME}.tar.gz') as pbar:
    s3.download_file(
      ROOT_BUCKET,
      f'language_model/{BERT_MODEL_NAME}.tar.gz',
      f'{BERT_MODEL_NAME}.tar.gz',
      Callback=lambda bytes_transferred: pbar.update(bytes_transferred)
    )

    tar = tarfile.open(f'{str(BERT_MODEL_NAME)}.tar.gz')
    tar.close()

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

training_args = TrainingArguments(
  output_dir=CN_MODEL_NAME,
  overwrite_output_dir=True,
  num_train_epochs=1,
  per_device_train_batch_size=16,
  gradient_accumulation_steps=2,
  gradient_checkpointing=True,
  optim="adafactor",
  save_steps=1000,
  save_total_limit=1,
  warmup_steps=1000,
  weight_decay=0.01,
  learning_rate=1e-5,
  report_to=["wandb"],
  logging_steps=500,
  do_eval=False,
  fp16=True
)

set_seed(42)

trains_ds = load_dataset('text', data_files={'train': [CORPUS_TRAIN]})

config = AutoConfig.from_pretrained(BERT_MODEL_NAME)
tokenizer = RobertaTokenizerFast.from_pretrained(BERT_MODEL_NAME, max_len=512, add_prefix_space=True)

model = AutoModelForMaskedLM.from_pretrained(
  BERT_MODEL_NAME,
  config=config
)


def tokenize_function(examples):
  return tokenizer(
    examples["text"],
    padding=True,
    truncation=True,
    max_length=512,
    return_special_tokens_mask=True
  )


tokenized_ds_train = trains_ds['train'].map(
  tokenize_function,
  batched=True,
  num_proc=5,
  remove_columns=['text'],
  load_from_cache_file=True,
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

trainer = Trainer(
  model=model,
  args=training_args,
  train_dataset=tokenized_ds_train,
  tokenizer=tokenizer,
  data_collator=data_collator
)

trainer.train()
trainer.save_model()
w_run.finish()
