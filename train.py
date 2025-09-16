import re
import datasets
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# --- Helper to parse CoNLL files ---
def parse_conll(path):
    tokens, labels = [], []
    sentences = {"tokens": [], "ner_tags": []}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    sentences["tokens"].append(tokens)
                    sentences["ner_tags"].append(labels)
                    tokens, labels = [], []
                continue
            parts = re.split(r"\s+", line)
            if len(parts) < 2:
                continue
            token, label = parts[0], parts[-1]
            tokens.append(token)
            labels.append(label)
    if tokens:  # last sentence
        sentences["tokens"].append(tokens)
        sentences["ner_tags"].append(labels)
    return sentences

train_data = parse_conll("data/econ_ie/train.conll")
dev_data = parse_conll("data/econ_ie/dev.conll")
test_data = parse_conll("data/econ_ie/test.conll")

# Build label list dynamically
label_list = sorted(set(l for seq in train_data["ner_tags"] for l in seq))
label2id = {l: i for i, l in enumerate(label_list)}
id2label = {i: l for l, i in label2id.items()}

# Convert labels to IDs
def encode_labels(data):
    return [[label2id[l] for l in seq] for seq in data["ner_tags"]]

train_data["ner_tags"] = encode_labels(train_data)
dev_data["ner_tags"] = encode_labels(dev_data)
test_data["ner_tags"] = encode_labels(test_data)

# Convert to Dataset objects
dataset = datasets.DatasetDict({
    "train": datasets.Dataset.from_dict(train_data),
    "validation": datasets.Dataset.from_dict(dev_data),
    "test": datasets.Dataset.from_dict(test_data),
})

from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("worldbank/econberta", use_fast=False)


model = AutoModelForTokenClassification.from_pretrained(
    model_name, num_labels=len(label_list), id2label=id2label, label2id=label2id
)


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], is_split_into_words=True, truncation=True, padding="max_length", max_length=128
    )
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_predictions = [
        [label_list[p] for (p, l) in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]
    results = precision_recall_fscore_support(
        [l for seq in true_labels for l in seq],
        [p for seq in true_predictions for p in seq],
        average="macro"
    )
    return {"precision": results[0], "recall": results[1], "f1": results[2], "accuracy": accuracy_score(
        [l for seq in true_labels for l in seq],
        [p for seq in true_predictions for p in seq],
    )}




'''
Hyper-parameter Value
✅ Dropout of Task Layer 0.2
✅ Learning Rate [5e-5, 6e-5, 7e-5]
✅ Batch size 12
✅ Gradient Acc. Steps 4
✅ Weight Decay 0
✅ Maximum Training Epochs 10

These required too much fiddling w/ HF and I skipped them. 
AdamW is already the default optimizer
❌ Learning Rate Decay Slanted Triangular
❌ Fraction of steps 6%
❌ Adam ϵ 1e-8
❌ Adam β1 0.9
❌ Adam β2 0.999
'''

# they use dropout of Task Layer 0.2
model.classifier.dropout.p = 0.2

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=6e-5,
    per_device_train_batch_size=12,
    per_device_eval_batch_size=12,
    gradient_accumulation_steps=4,
    num_train_epochs=10,
    logging_steps=50,
    weight_decay=0.0,
    save_strategy="epoch",
    eval_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

