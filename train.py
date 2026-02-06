from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import evaluate

dataset = load_dataset("armanc/pubmed-rct20k")

print(dataset["train"].features)

dataset = dataset.class_encode_column("label")
label_names = dataset["train"].features["label"].names
num_labels = len(label_names)

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_fn(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )

train_dataset = dataset["train"].select(range(2000))
eval_dataset = dataset["validation"].select(range(500))

train_dataset = train_dataset.map(tokenize_fn, batched=True)
eval_dataset = eval_dataset.map(tokenize_fn, batched=True)

print(f"Train size: {len(train_dataset)} | Eval size: {len(eval_dataset)}")

train_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"]
)
eval_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"]
)

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=num_labels,
    id2label={i: name for i, name in enumerate(label_names)},
    label2id={name: i for i, name in enumerate(label_names)}
)

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return accuracy.compute(predictions=preds, references=labels)

args = TrainingArguments(
    output_dir="output",
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    max_steps=200,
    logging_steps=10,
    disable_tqdm=True,
    dataloader_pin_memory=False,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

print("Starting training...")
try:
    trainer.train()
except Exception as exc:
    print(f"Training failed: {exc!r}")
    raise
print("Training finished.")

metrics = trainer.evaluate()
print(metrics)

trainer.save_model("clinical_nlp_model")
tokenizer.save_pretrained("clinical_nlp_model")
