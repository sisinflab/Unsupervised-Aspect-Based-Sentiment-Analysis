import os
from transformers import TrainingArguments, Trainer, AutoConfig
import numpy as np
import evaluate

from model.model import *
from train_process import DatasetProcessor

os.environ["WANDB_DISABLED"] = "true"
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

#load train dataset
processor = DatasetProcessor("...data/train/train_dataset.json")
train_dataset, val_dataset = processor.prepare_datasets()

config = AutoConfig.from_pretrained(".../model/config.json")

#load model
model = CustomBertForSequenceClassification(config)
model = model.to(device)

#freeze bert weights
for param in model.bert.parameters():
    param.requires_grad = False
for param in model.distilbert.parameters():
    param.requires_grad = False
    
training_args = TrainingArguments(
    save_steps=500,
    load_best_model_at_end=True,
    save_total_limit=3,
    eval_strategy = "steps")

metric = evaluate.load("accuracy")

training_args.learning_rate = 5e-05
training_args.per_device_eval_batch_size = 16
training_args.per_device_train_batch_size = 16
training_args.gradient_accumulation_steps = 4

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset['train'],
    eval_dataset=val_dataset['train'],
    compute_metrics=compute_metrics,
)

trainer.train()



    