import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments
from transformers import Trainer

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

raw_datasets = load_dataset("glue", "mrpc")

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
training_args = TrainingArguments("test-trainer")

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()