import pandas as pd
from transformers import MarianMTModel, MarianTokenizer, MarianConfig, Trainer, Seq2SeqTrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
import torch # type: ignore
import numpy as np

# data
first_half = pd.read_csv('/netscratch/dgurgurov/thesis/mt_lrls/results/tinystories_badlrl_1.csv')
second_half = pd.read_csv('/netscratch/dgurgurov/thesis/mt_lrls/data/second_half.csv')

# combining data
data = pd.concat([first_half, second_half], ignore_index=True)

# splitting into train and validation sets
train_data, val_data = train_test_split(data, test_size=0.1, random_state=42)

# converting to HF datasets
train_dataset = Dataset.from_pandas(train_data)
val_dataset = Dataset.from_pandas(val_data)

# initializing tokenizer and model from scratch
tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-mt')
config = MarianConfig.from_pretrained('Helsinki-NLP/opus-mt-en-mt')
model = MarianMTModel(config) # type: ignore

# tokenization function
def tokenize_function(examples):
    inputs = tokenizer(examples['en'], truncation=True, padding='max_length', max_length=256)
    with tokenizer.as_target_tokenizer():
        targets = tokenizer(examples['mt'], truncation=True, padding='max_length', max_length=256)
    inputs['labels'] = targets['input_ids']
    return inputs

# tokenizing datasets
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, num_proc=4, batch_size=32)
tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True, num_proc=4, batch_size=32)

# training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir='./results',
    evaluation_strategy="steps",
    eval_steps=2000,
    save_steps=2000,
    learning_rate=2e-5,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    num_train_epochs=10,
    load_best_model_at_end=True,
    weight_decay=0.01,
    save_total_limit=2,
    predict_with_generate=True,
    gradient_accumulation_steps=2,
    ddp_find_unused_parameters=False,
    fp16=True,  # using FP16 only if a GPU is available
    torch_compile=True,
)

# metrics 
from datasets import load_metric
bleu_metric = load_metric('bleu')

# BLEU score
def compute_bleu(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # BLEU expects a list of predictions and a list of lists of references
    # So we need to wrap each reference in a list
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

# training
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    compute_metrics=compute_bleu,
    preprocess_logits_for_metrics=preprocess_logits 
)

trainer.train()

trainer.save_model('/netscratch/dgurgurov/thesis/mt_lrls/models/safecheck/')
tokenizer.save_pretrained('/netscratch/dgurgurov/thesis/mt_lrls/models/safecheck/')
