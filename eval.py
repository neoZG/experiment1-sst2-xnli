# This evaluation script loads the fine-tuned model from the outputs/ directory and evaluates it on the specified dataset. 
# For SST-2, it simply uses the validation set (English). For XNLI, it filters the combined XNLI 
# validation set by language to evaluate each language separately. We tokenize the data similarly to training. 
# Then, for each language's set, we loop through in batches (to measure time manually) and accumulate predictions. 
# We compute accuracy and F1, and also measure the average time per sample (in milliseconds). 
# The results are printed for each language.

import time, argparse, os, yaml
import torch
import random
import numpy as np
from datasets import load_dataset, set_seed
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

# Set all random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
set_seed(SEED)

parser = argparse.ArgumentParser(description="Evaluate fine-tuned models on SST-2 or XNLI")
parser.add_argument("--task", choices=["sst2", "xnli"], required=True, help="Task name")
parser.add_argument("--model", choices=["bitnet", "distilbert", "gptneo"], required=True, help="Which model to evaluate")
parser.add_argument("--languages", type=str, default="en", help="Comma-separated XNLI languages to evaluate (e.g. 'en,es,de'). Use 'all' for all languages.")
args = parser.parse_args()

task = args.task
model_choice = args.model
langs = args.languages.split(",") if args.languages != "all" else ["ar","bg","de","el","en","es","fr","hi","ru","sw","th","tr","ur","vi","zh"]

# Determine model path and load config
model_dir = f"outputs/{model_choice}-{task}"
config_path = os.path.join(model_dir, "training_config.yaml")
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        training_config = yaml.safe_load(f)
    max_seq_length = training_config.get("max_seq_length", 128)
    label_map = training_config.get("label_map", None)
    batch_size = training_config.get("batch_size", 16)  # Use training batch size or default to 16
else:
    print("Warning: No training config found. Using default settings.")
    max_seq_length = 128
    label_map = None
    batch_size = 16

# Load tokenizer & model
print(f"Loading model from {model_dir}...")
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)

# Setup device and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# Reset CUDA memory stats
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

# Load evaluation data
if task == "sst2":
    dataset = load_dataset("glue", "sst2")["validation"]
    eval_sets = {"en": dataset}
elif task == "xnli":
    eval_sets = {}
    xnli = load_dataset("xnli")
    for lang in langs:
        subset = xnli["validation"].filter(lambda ex: ex["language"] == lang)
        eval_sets[lang] = subset
        print(f"Language '{lang}': {len(subset)} examples")
else:
    raise ValueError("Unsupported task for evaluation")

# Preprocess function (same as in training)
def preprocess_function(examples, task):
    if task == "xnli":
        return tokenizer(examples['premise'], examples['hypothesis'], 
                         truncation=True, padding='max_length', max_length=max_seq_length)
    else:
        return tokenizer(examples['sentence'], 
                         truncation=True, padding='max_length', max_length=max_seq_length)

# Results dictionary to save
results = {
    "task": task,
    "model": model_choice,
    "device": str(device),
    "batch_size": batch_size,
    "max_seq_length": max_seq_length,
    "results_by_language": {}
}

# Evaluate for each requested language
for lang, data in eval_sets.items():
    print(f"\nEvaluating language: {lang}")
    print(f"Number of examples: {len(data)}")
    
    try:
        # Tokenize the evaluation set
        data = data.map(
            lambda x: preprocess_function(x, task), 
            batched=True,
            num_proc=os.cpu_count(),
            desc=f"Preprocessing {lang}"
        )
        data.set_format(type='torch', columns=['input_ids','attention_mask','label'])
        
        # Create batches
        num_samples = len(data)
        
        # Gather predictions
        all_preds = []
        all_labels = []
        total_time = 0
        
        for i in range(0, num_samples, batch_size):
            batch = data[i: i+batch_size]
            batch_inputs = {k: v.to(device) for k, v in batch.items() if k != 'label'}
            
            start = time.time()
            with torch.no_grad():
                outputs = model(**batch_inputs)
            end = time.time()
            total_time += end - start
            
            logits = outputs.logits.cpu()
            preds = logits.argmax(dim=-1).numpy()
            labels = batch['label'].numpy()
            all_preds.append(preds)
            all_labels.append(labels)
        
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        
        # Compute metrics
        acc = accuracy_score(all_labels, all_preds)
        if task == "sst2":
            f1 = f1_score(all_labels, all_preds, average="binary")
        else:
            f1 = f1_score(all_labels, all_preds, average="macro")
        avg_time = total_time / num_samples
        
        # Store results
        lang_results = {
            "accuracy": float(acc * 100),
            "f1": float(f1),
            "avg_time_ms": float(avg_time * 1000),
            "num_examples": num_samples
        }
        results["results_by_language"][lang] = lang_results
        
        print(f"Results for language '{lang}':")
        print(f"- Accuracy = {acc*100:.2f}%")
        print(f"- F1 = {f1:.3f}")
        print(f"- Runtime per sample = {avg_time*1000:.2f} ms")
    except Exception as e:
        print(f"Error evaluating language {lang}: {str(e)}")
        results["results_by_language"][lang] = {"error": str(e)}

# Add resource usage information
if torch.cuda.is_available():
    try:
        results["peak_gpu_memory_gb"] = torch.cuda.max_memory_allocated() / (1024 ** 3)
    except Exception as e:
        print(f"Warning: Failed to get GPU memory stats: {str(e)}")

# Save results
try:
    output_file = os.path.join(model_dir, f"eval_results_{'-'.join(langs)}.yaml")
    with open(output_file, 'w') as f:
        yaml.dump(results, f)
    print(f"\nResults saved to {output_file}")
except Exception as e:
    print(f"Warning: Failed to save evaluation results: {str(e)}")
