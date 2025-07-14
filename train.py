# This script handles both tasks. It loads the appropriate dataset, optionally subsamples it, 
# then loads the model. For 8-bit quantization, we use BitsAndBytesConfig to load the model in 8-bit 
# (this requires the bitsandbytes library). We also ensure the tokenizer has a pad token 
# (especially needed for GPT-Neo or BitNet, which are causal LMs) to avoid errors during batching. 
# We then fine-tune using Trainer, with evaluation each epoch. 
# We compute accuracy and F1 via a compute_metrics function using scikit-learn. 
# Finally, the model is saved to output_dir.

import os, time, argparse, yaml
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import BitsAndBytesConfig
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# Parse arguments and config
parser = argparse.ArgumentParser(description="Fine-tune models on SST-2 or XNLI(MultiNLI)")
parser.add_argument("--task", choices=["sst2", "xnli"], help="Task: 'sst2' or 'xnli'", required=True)
parser.add_argument("--model", choices=["bitnet", "distilbert", "gptneo"], help="Which model to fine-tune", required=True)
parser.add_argument("--train_samples", type=int, default=None, help="Number of training samples to use (subset). -1 for full.")
parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
parser.add_argument("--batch_size", type=int, default=None, help="Batch size per iteration")
parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate")
parser.add_argument("--max_seq_length", type=int, default=None, help="Maximum sequence length for tokenization")
parser.add_argument("--bf16", action="store_true", help="Use bfloat16 mixed precision")
parser.add_argument("--fp16", action="store_true", help="Use float16 mixed precision")
parser.add_argument("--quant8", action="store_true", help="Load model in 8-bit precision (requires bitsandbytes)")
args = parser.parse_args()

# Load default config.yaml
config = {}
if os.path.exists("config.yaml"):
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

# Override config with CLI args if provided
task = args.task
model_choice = args.model
train_samples = args.train_samples if args.train_samples is not None else config.get("train_samples", -1)
num_epochs = args.epochs if args.epochs is not None else config.get("num_epochs", 3)
batch_size = args.batch_size if args.batch_size is not None else config.get("batch_size", 16)
learning_rate = args.learning_rate if args.learning_rate is not None else config.get("learning_rate", 2e-5)
max_seq_length = args.max_seq_length if args.max_seq_length is not None else config.get("max_seq_length", 128)
use_bf16 = args.bf16 or config.get("bf16", False)
use_fp16 = args.fp16 or config.get("fp16", False)
use_quant8 = args.quant8 or config.get("quant8", False)
output_dir = os.path.join(config.get("output_dir", "outputs"), f"{model_choice}-{task}")

# Choose model name or path for HuggingFace
if model_choice == "bitnet":
    model_name = "microsoft/bitnet-b1.58-2B-4T-bf16"  # BF16 fine-tunable BitNet weights
elif model_choice == "distilbert":
    # Use multilingual DistilBERT for XNLI to handle non-English, otherwise English uncased for SST-2
    model_name = "distilbert-base-multilingual-cased" if task == "xnli" else "distilbert-base-uncased"
elif model_choice == "gptneo":
    model_name = "EleutherAI/gpt-neo-2.7B"
else:
    raise ValueError("Unknown model choice!")

# Load dataset
if task == "sst2":
    ds = load_dataset("glue", "sst2")
    train_data = ds["train"]
    val_data = ds["validation"]
elif task == "xnli":
    # Use MultiNLI for training, XNLI for validation (English dev as val set for simplicity)
    mnli = load_dataset("multi_nli")
    train_data = mnli["train"]
    # Use MultiNLI matched validation as a proxy for English dev
    val_data = mnli["validation_matched"]
else:
    raise ValueError("Unknown task!")

# Subsample training data if specified
if train_samples is not None and train_samples > 0:
    if train_samples < len(train_data):
        train_data = train_data.select(range(train_samples))
        print(f"Subsampling train set to {train_samples} samples")
# (No need to subsample val data since it's already small enough for quick evaluation)

# Load tokenizer and model (with optional 8-bit quantization)
print(f"Loading model {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model_kwargs = {}
if use_quant8:
    # Configure 8-bit loading (quantization aware training)
    model_kwargs["load_in_8bit"] = True
    model_kwargs["bnb_config"] = BitsAndBytesConfig(load_in_8bit=True)
if use_bf16:
    model_kwargs["torch_dtype"] = "bfloat16"
elif use_fp16:
    model_kwargs["torch_dtype"] = "auto"  # fp16 autocast if available

# Some models (GPT-Neo, BitNet) might not have a pad token, so set it if missing
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=(2 if task=="sst2" else 3),
    **model_kwargs
)
# If using pad token, also resize embeddings if new pad_token added:
if model.config.vocab_size != len(tokenizer):
    model.resize_token_embeddings(len(tokenizer))

# Tokenization function
def preprocess_function(examples):
    # For NLI, we have premise and hypothesis
    if task == "xnli":
        return tokenizer(examples['premise'], examples['hypothesis'], 
                         truncation=True, padding='max_length', max_length=max_seq_length)
    else:  # SST-2
        return tokenizer(examples['sentence'], 
                         truncation=True, padding='max_length', max_length=max_seq_length)

# Prepare datasets for training
train_dataset = train_data.map(preprocess_function, batched=True)
val_dataset = val_data.map(preprocess_function, batched=True)
# Transform to PyTorch format
train_dataset.set_format(type='torch', columns=['input_ids','attention_mask','label'])
val_dataset.set_format(type='torch', columns=['input_ids','attention_mask','label'])

# Define metrics computation
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    if task == "sst2":
        # For binary, F1 of positive class
        f1 = f1_score(labels, preds, average="binary")
    else:
        # For NLI 3-class, use macro-averaged F1
        f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "f1": f1}

# Setup training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    save_strategy="no",  # no intermediate checkpoints to save time/space
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    learning_rate=learning_rate,
    logging_steps=50,
    bf16=use_bf16,
    fp16= (use_fp16 and not use_bf16),  # use fp16 if requested (mutually exclusive with bf16)
    dataloader_pin_memory=False,  # avoid issues on some systems
    report_to="none"  # no Huggingface logging
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# Train and evaluate
start_time = time.time()
trainer.train()
training_time = time.time() - start_time
print(f"Training completed in {training_time/60:.2f} minutes.")

# Final evaluation on validation set
eval_metrics = trainer.evaluate()
print(f"Validation results: {eval_metrics}")

# Save final model and tokenizer
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
