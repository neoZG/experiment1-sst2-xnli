# This evaluation script loads the fine-tuned model from the outputs/ directory and evaluates it on the specified dataset. 
# For SST-2, it simply uses the validation set (English). For XNLI, it filters the combined XNLI 
# validation set by language to evaluate each language separately. We tokenize the data similarly to training. 
# Then, for each language's set, we loop through in batches (to measure time manually) and accumulate predictions. 
# We compute accuracy and F1, and also measure the average time per sample (in milliseconds). 
# The results are printed for each language.

import time, argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

parser = argparse.ArgumentParser(description="Evaluate fine-tuned models on SST-2 or XNLI")
parser.add_argument("--task", choices=["sst2", "xnli"], required=True, help="Task name")
parser.add_argument("--model", choices=["bitnet", "distilbert", "gptneo"], required=True, help="Which model to evaluate")
parser.add_argument("--languages", type=str, default="en", help="Comma-separated XNLI languages to evaluate (e.g. 'en,es,de'). Use 'all' for all languages.")
args = parser.parse_args()

task = args.task
model_choice = args.model
langs = args.languages.split(",") if args.languages != "all" else ["ar","bg","de","el","en","es","fr","hi","ru","sw","th","tr","ur","vi","zh"]  # all XNLI languages

# Determine model path (should match output_dir from training)
model_dir = f"outputs/{model_choice}-{task}"
# Load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)

# Load evaluation data
if task == "sst2":
    dataset = load_dataset("glue", "sst2")["validation"]
    eval_sets = {"en": dataset}  # just one set (English)
elif task == "xnli":
    eval_sets = {}
    xnli = load_dataset("xnli")
    for lang in langs:
        # XNLI dataset provides 'validation' split for all languages combined
        # Filter by language
        subset = xnli["validation"].filter(lambda ex: ex["language"] == lang)
        eval_sets[lang] = subset
else:
    raise ValueError("Unsupported task for evaluation")

# Preprocess function (same as in training)
def preprocess_function(examples, task):
    if task == "xnli":
        return tokenizer(examples['premise'], examples['hypothesis'], truncation=True, padding='max_length', max_length=256)
    else:
        return tokenizer(examples['sentence'], truncation=True, padding='max_length', max_length=128)

# Evaluate for each requested language
for lang, data in eval_sets.items():
    # Tokenize the evaluation set
    data = data.map(lambda x: preprocess_function(x, task), batched=True)
    data.set_format(type='torch', columns=['input_ids','attention_mask','label'])
    # Create batches
    batch_size = 16
    num_samples = len(data)
    # Gather predictions
    all_preds = []
    all_labels = []
    start = time.time()
    for i in range(0, num_samples, batch_size):
        batch = data[i: i+batch_size]
        with torch.no_grad():
            outputs = model(batch['input_ids'], attention_mask=batch['attention_mask'])
        logits = outputs.logits
        preds = logits.argmax(dim=-1).cpu().numpy()
        labels = batch['label'].numpy()
        all_preds.append(preds)
        all_labels.append(labels)
    end = time.time()
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    # Compute metrics
    acc = accuracy_score(all_labels, all_preds)
    if task == "sst2":
        f1 = f1_score(all_labels, all_preds, average="binary")
    else:
        f1 = f1_score(all_labels, all_preds, average="macro")
    avg_time = (end - start) / num_samples
    print(f"Results for language '{lang}': Accuracy = {acc*100:.2f}%, F1 = {f1:.3f}, Runtime per sample = {avg_time*1000:.2f} ms")
