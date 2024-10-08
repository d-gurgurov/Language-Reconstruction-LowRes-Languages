import pandas as pd
from transformers import MarianMTModel, MarianTokenizer, Trainer, TrainingArguments, Seq2SeqTrainingArguments, MarianConfig
from datasets import Dataset
import torch # type: ignore
from sklearn.model_selection import train_test_split
import numpy as np
import argparse

# parsing command line arguments
parser = argparse.ArgumentParser(description='Training a reconstruction model on half data')
parser.add_argument('--language', type=str, required=True, help='Target language code (e.g., "sw" for Swahili)')
args = parser.parse_args()

# setting language code from command line argument
language = args.language
lang_map = {"sw": "swahili", "mt": "maltese", "ga": "irish", "is": "icelandic",
            "tl": "tagalog", "hr": "croatian", "nn": "norwegian"}

# training data
train_data = pd.read_csv(f'/netscratch/dgurgurov/projects2024/mt_lrls/data/train_{lang_map[language]}/first_half_badlrl.csv', keep_default_na=False)
train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)

# converting the data into HF datasets
train_dataset = Dataset.from_pandas(train_data)
val_dataset = Dataset.from_pandas(val_data)

# tokenizer and model
model_name = f'Helsinki-NLP/opus-mt-en-{language}'
config = MarianConfig.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel(config) # type: ignore

# tokenizing the data
def tokenize_function(examples):
    # TODO: fix the column choice
    # UPDATE: fixed now. let's rerun the experiments
    inputs = tokenizer(examples['bad_lrl'], truncation=True, padding='max_length', max_length=512)
    with tokenizer.as_target_tokenizer():
        targets = tokenizer(examples[language], truncation=True, padding='max_length', max_length=512)
    inputs['labels'] = targets['input_ids']
    return inputs

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, num_proc=4, batch_size=32)
tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True, num_proc=4, batch_size=32)

# training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=f'/netscratch/dgurgurov/projects2024/mt_lrls/models/{lang_map[language]}/reconstruction/results',
    evaluation_strategy="steps",
    eval_steps=2000, # change to 10000
    save_steps=2000,
    learning_rate=5e-5,
    per_device_train_batch_size=64, # (64 for ga, is) and (32 for mt, sw, tl)
    per_device_eval_batch_size=64, 
    num_train_epochs=20,
    load_best_model_at_end=True,
    weight_decay=0.01,
    save_total_limit=2,
    predict_with_generate=True,
    gradient_accumulation_steps=4,
    ddp_find_unused_parameters=False,
    fp16=True, 
    torch_compile=True,
    seed=42
)

from datasets import load_metric
bleu_metric = load_metric('bleu')

def compute_bleu(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # BLEU expects a list of predictions and a list of lists of references
    # so we need to wrap each reference in a list
    decoded_preds = [pred.split() for pred in decoded_preds]
    decoded_labels = [[label.split()] for label in decoded_labels]
    bleu_score = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels) # type: ignore
    return {"bleu": bleu_score}

def preprocess_logits(logits, labels):
    """
    Original Trainer may have a memory leak. 
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    pred_ids = torch.argmax(logits[0], dim=-1)
    return pred_ids, labels

# trainer
trainer = Trainer( 
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset, # type: ignore
    eval_dataset=tokenized_val_dataset, # type: ignore
    compute_metrics=compute_bleu,
    preprocess_logits_for_metrics=preprocess_logits # type: ignore
)

# train
trainer.train()

# save
trainer.save_model(f'/netscratch/dgurgurov/projects2024/mt_lrls/models/{lang_map[language]}/reconstruction')
tokenizer.save_pretrained(f'/netscratch/dgurgurov/projects2024/mt_lrls/models/{lang_map[language]}/reconstruction')
