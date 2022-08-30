from transformers import set_seed
from datasets import load_dataset
import boto3
import wandb
import os

from transformers import (
  AutoConfig,
  AutoModelForMaskedLM,
  AutoTokenizer,
  DataCollatorForLanguageModeling,
  Trainer,
  TrainingArguments
)

CN_MODEL_NAME   = 'shc-lm-v3'
CORPUS_BUCKET   = 'shc-mlm-corpus'
CORPUS_TRAIN    = 'corpus_train.shc'
CORPUS_DEV      = 'corpus_dev.shc'
BERT_MODEL_NAME = 'xlm-roberta-base'
PROJECT_NAME    = 'shc'

wandb.login()

w_run = wandb.init(project=PROJECT_NAME, notes="Transfer learning from V1")

s3 = boto3.client('s3')
if not os.path.exists(CORPUS_TRAIN):
  s3.download_file(CORPUS_BUCKET, CORPUS_TRAIN, CORPUS_TRAIN)
if not os.path.exists(CORPUS_DEV):
  s3.download_file(CORPUS_BUCKET, CORPUS_DEV, CORPUS_DEV)

training_args = TrainingArguments(
  output_dir=f'./{CN_MODEL_NAME}',
  overwrite_output_dir=True,
  num_train_epochs=2,
  per_device_train_batch_size=2,
  save_steps=1000,
  save_total_limit=1,
  gradient_accumulation_steps=8,
  warmup_steps=1000,
  weight_decay=0.01,
  learning_rate=1e-5,
  report_to=["wandb"],
  logging_steps=500,
  fp16=True
)

# Works on v100 and T4:
# batch: 2
# grad acc: 8

set_seed(42)

trains_ds = load_dataset('text', data_files={'train': [CORPUS_TRAIN]})

vals_ds = load_dataset('text', data_files={'test': [CORPUS_DEV]})

config = AutoConfig.from_pretrained(BERT_MODEL_NAME)

tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
model = AutoModelForMaskedLM.from_pretrained(
  BERT_MODEL_NAME,
  config=config
)


def tokenize_function(examples):
  return tokenizer(
    examples["text"],
    padding=True,
    truncation=True,
    max_length=510,
    return_special_tokens_mask=True
  )


tokenized_ds_eval = vals_ds['test'].map(
  tokenize_function,
  batched=True,
  num_proc=1,
  remove_columns=['text'],
  load_from_cache_file=True,
)

tokenized_ds_train = trains_ds['train'].map(
  tokenize_function,
  batched=True,
  num_proc=1,
  remove_columns=['text'],
  load_from_cache_file=True,
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

trainer = Trainer(
  model=model,
  args=training_args,
  train_dataset=tokenized_ds_train,
  eval_dataset=tokenized_ds_eval,
  tokenizer=tokenizer,
  data_collator=data_collator
)

trainer.train()
trainer.save_model()
w_run.finish()
