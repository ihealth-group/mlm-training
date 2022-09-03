from datasets import load_metric, load_dataset
import numpy as np
import boto3
import os

metric = load_metric("accuracy")

CORPUS_DEV = 'corpus_small_train.shc'
CORPUS_BUCKET = 'shc-mlm-corpus'
MODEL = 'shc-lm-v3'

from transformers import (
  AutoConfig,
  AutoModelForMaskedLM,
  RobertaTokenizerFast,
  DataCollatorForLanguageModeling,
  Trainer,
  TrainingArguments
)

config = AutoConfig.from_pretrained(MODEL)
tokenizer = RobertaTokenizerFast.from_pretrained(MODEL, max_len=512)

model = AutoModelForMaskedLM.from_pretrained(
  MODEL,
  config=config
)

s3 = boto3.client('s3')

if not os.path.exists(CORPUS_DEV):
  s3.download_file(CORPUS_BUCKET, CORPUS_DEV, CORPUS_DEV)

vals_ds = load_dataset('text', data_files={'test': [CORPUS_DEV]})


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

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)


def compute_metrics(eval_pred):
  logits, labels = eval_pred
  predictions = np.argmax(logits, axis=-1)

  indices = [[i for i, x in enumerate(labels[row]) if x != -100] for row in range(len(labels))]

  labels = [labels[row][indices[row]] for row in range(len(labels))]
  labels = [item for sublist in labels for item in sublist]

  predictions = [predictions[row][indices[row]] for row in range(len(predictions))]
  predictions = [item for sublist in predictions for item in sublist]

  return metric.compute(predictions=predictions, references=labels)


training_args = TrainingArguments(
  output_dir='eval',
  per_device_train_batch_size=1,
  gradient_accumulation_steps=1,
  per_device_eval_batch_size=4,
  eval_accumulation_steps=1,
  do_eval=True,
  fp16=True
)

trainer = Trainer(
  model=model,
  args=training_args,
  compute_metrics=compute_metrics,
  tokenizer=tokenizer,
  data_collator=data_collator,
  eval_dataset=tokenized_ds_eval
)

results = trainer.evaluate()
print(results)
