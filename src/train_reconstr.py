import pandas as pd
from transformers import MarianMTModel, MarianTokenizer, Trainer, TrainingArguments, Seq2SeqTrainingArguments, MarianConfig
from datasets import Dataset
import torch # type: ignore
from sklearn.model_selection import train_test_split
import numpy as np

# training data
train_data = pd.read_csv('/netscratch/dgurgurov/thesis/mt_lrls/data/badlrl_first_half.csv')
train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)

# converting the data into HF datasets
train_dataset = Dataset.from_pandas(train_data)
val_dataset = Dataset.from_pandas(val_data)

# tokenizer and model
model_name = 'Helsinki-NLP/opus-mt-mt-en'
config = MarianConfig.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# tokenizing the data
def tokenize_function(examples):
    # TODO: fix the column choice
    # UPDATE: fixed now. let's rerun the experiments
    inputs = tokenizer(examples['bad_lrl'], truncation=True, padding='max_length', max_length=256)
    with tokenizer.as_target_tokenizer():
        targets = tokenizer(examples['mt'], truncation=True, padding='max_length', max_length=256)
    inputs['labels'] = targets['input_ids']
    return inputs

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, num_proc=4, batch_size=32)
tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True, num_proc=4, batch_size=32)

# training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir='/netscratch/dgurgurov/thesis/mt_lrls/models/reconstruction_30',
    evaluation_strategy="steps",
    eval_steps=5000, # change to 10000
    save_steps=5000,
    learning_rate=5e-5,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128, 
    num_train_epochs=30,
    load_best_model_at_end=True,
    weight_decay=0.01,
    save_total_limit=2,
    predict_with_generate=True,
    gradient_accumulation_steps=2,
    ddp_find_unused_parameters=False,
    fp16=True, 
    torch_compile=True,
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
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    compute_metrics=compute_bleu,
    preprocess_logits_for_metrics=preprocess_logits 
)

# train
trainer.train()

# save
trainer.save_model('/netscratch/dgurgurov/thesis/mt_lrls/models/reconstruction_30')
tokenizer.save_pretrained('/netscratch/dgurgurov/thesis/mt_lrls/models/reconstruction_30')
