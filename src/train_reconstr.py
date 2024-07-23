import pandas as pd
from transformers import MarianMTModel, MarianTokenizer, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split

# training data
train_data = pd.read_csv('/netscratch/dgurgurov/thesis/mt_lrls/results/badlrl_first_half.csv')
train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)

# converting the data into HF datasets
train_dataset = Dataset.from_pandas(train_data)
val_dataset = Dataset.from_pandas(val_data)

# tokenizer and model
model_name = 'Helsinki-NLP/opus-mt-mt-en'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# tokenizing the data
def tokenize_function(examples):
    inputs = tokenizer(examples['bad_lrl'], truncation=True, padding='max_length', max_length=256)
    targets = tokenizer(examples['mt'], truncation=True, padding='max_length', max_length=256)
    inputs['labels'] = targets['input_ids']
    return inputs

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, num_proc=4, batch_size=32)
tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True, num_proc=4, batch_size=32)

# training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="steps",
    eval_steps=5000,
    save_steps=5000,
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64, 
    num_train_epochs=3,
    load_best_model_at_end=True,
    weight_decay=0.01,
    save_total_limit=2,
    eval_accumulation_steps=5, 
    fp16=True, 
    ddp_find_unused_parameters=False,
    torch_compile=True,
)

from datasets import load_metric
bleu_metric = load_metric('bleu')

def compute_bleu(eval_preds):
    preds, labels = eval_preds
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    decoded_labels = [[label for label in label_seq if label != -100] for label_seq in decoded_labels]
    
    bleu_score = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels) # type: ignore
    return {"bleu": bleu_score}

# trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    compute_metrics=compute_bleu
)

# train
trainer.train()

# save
trainer.save_model('/netscratch/dgurgurov/thesis/mt_lrls/results/')
tokenizer.save_pretrained('/netscratch/dgurgurov/thesis/mt_lrls/results/')
