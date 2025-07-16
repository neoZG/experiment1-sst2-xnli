# This script handles both tasks. It loads the appropriate dataset, optionally subsamples it, 
# then loads the model. For 8-bit quantization, we use BitsAndBytesConfig to load the model in 8-bit 
# (this requires the bitsandbytes library). We also ensure the tokenizer has a pad token 
# (especially needed for GPT-Neo or BitNet, which are causal LMs) to avoid errors during batching. 
# We then fine-tune using Trainer, with evaluation each epoch. 
# We compute accuracy and F1 via a compute_metrics function using scikit-learn. 
# Finally, the model is saved to output_dir.

import os, time, argparse, yaml
import torch
import random
import numpy as np
from datasets import load_dataset, set_seed
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import BitsAndBytesConfig
from sklearn.metrics import accuracy_score, f1_score

# Set all random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
set_seed(SEED)

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
    num_labels = 2
    label_map = None  # Binary classification, no mapping needed
elif task == "xnli":
    # Use MultiNLI for training, XNLI for validation (English dev as val set for simplicity)
    mnli = load_dataset("multi_nli")
    train_data = mnli["train"]
    val_data = mnli["validation_matched"]
    num_labels = 3
    # Explicit label mapping for consistency
    label_map = {"contradiction": 0, "entailment": 1, "neutral": 2}
    
    # Verify and map labels if needed
    if "label" in train_data.features and isinstance(train_data.features["label"], datasets.ClassLabel):
        dataset_labels = train_data.features["label"].names
        if dataset_labels != list(label_map.keys()):
            print(f"Warning: Label mapping mismatch. Dataset labels: {dataset_labels}")
            print(f"Expected labels: {list(label_map.keys())}")
            # Create reverse mapping from dataset labels to our expected indices
            reverse_map = {label: i for i, label in enumerate(dataset_labels)}
            # Map dataset indices to our expected indices
            def remap_labels(example):
                example["label"] = label_map[dataset_labels[example["label"]]]
                return example
            print("Remapping labels to maintain consistency...")
            train_data = train_data.map(remap_labels)
            val_data = val_data.map(remap_labels)
else:
    raise ValueError("Unknown task!")

# Save training configuration with detailed label info
config_to_save = {
    "task": task,
    "model": model_choice,
    "seed": SEED,
    "train_samples": train_samples,
    "num_epochs": num_epochs,
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "max_seq_length": max_seq_length,
    "label_map": label_map,
    "num_labels": num_labels,
    "dataset_info": {
        "train_size": len(train_data),
        "val_size": len(val_data),
        "label_distribution": {
            "train": dict(train_data["label"].value_counts().items()),
            "val": dict(val_data["label"].value_counts().items())
        }
    }
}

try:
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "training_config.yaml"), 'w') as f:
        yaml.dump(config_to_save, f)
except Exception as e:
    print(f"Warning: Failed to save training configuration: {str(e)}")

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

# Tokenization function with resource monitoring
def preprocess_function(examples):
    # For NLI, we have premise and hypothesis
    if task == "xnli":
        return tokenizer(examples['premise'], examples['hypothesis'], 
                         truncation=True, padding='max_length', max_length=max_seq_length)
    else:  # SST-2
        return tokenizer(examples['sentence'], 
                         truncation=True, padding='max_length', max_length=max_seq_length)

# Prepare datasets for training with parallel processing
train_dataset = train_data.map(
    preprocess_function, 
    batched=True, 
    num_proc=os.cpu_count(),
    desc="Tokenizing training data"
)
val_dataset = val_data.map(
    preprocess_function, 
    batched=True, 
    num_proc=os.cpu_count(),
    desc="Tokenizing validation data"
)

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

# Resource monitoring callback
class ResourceMonitor(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.peak_memory = 0
        self.training_time = 0
        self.start_time = None
        
    def on_train_begin(self, args, state, control, **kwargs):
        """Called before training starts"""
        super().on_train_begin(args, state, control)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.start_time = time.time()
        
    def on_train_end(self, args, state, control, **kwargs):
        """Called after training ends"""
        super().on_train_end(args, state, control)
        self.training_time = time.time() - self.start_time
        if torch.cuda.is_available():
            self.peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB
        else:
            self.peak_memory = 0
        
        # Save resource usage
        try:
            resources = {
                "peak_gpu_memory_gb": self.peak_memory,
                "training_time_minutes": self.training_time / 60,
                "average_batch_time_ms": (self.training_time * 1000) / (len(self.train_dataset) * self.args.num_train_epochs / self.args.per_device_train_batch_size),
                "batch_size": self.args.per_device_train_batch_size,
                "device": str(self.args.device),
                "fp16": self.args.fp16,
                "bf16": self.args.bf16
            }
            os.makedirs(self.args.output_dir, exist_ok=True)
            with open(os.path.join(self.args.output_dir, "resource_usage.yaml"), 'w') as f:
                yaml.dump(resources, f)
        except Exception as e:
            print(f"Warning: Failed to save resource usage stats: {str(e)}")

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
    fp16=(use_fp16 and not use_bf16),  # use fp16 if requested (mutually exclusive with bf16)
    dataloader_pin_memory=False,  # avoid issues on some systems
    report_to="none",  # no Huggingface logging
    seed=SEED  # Set seed in training args too
)

trainer = ResourceMonitor(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# Train and evaluate
trainer.train()

# Final evaluation on validation set
eval_metrics = trainer.evaluate()
print(f"Validation results: {eval_metrics}")

# Save final model, tokenizer and results
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# Save evaluation metrics
with open(os.path.join(output_dir, "eval_results.yaml"), 'w') as f:
    yaml.dump(eval_metrics, f)
