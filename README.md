# Resource-Efficient Models: SST-2 and XNLI Experiment

This repository contains code to fine-tune and evaluate several **resource-efficient language models** (2–3B parameters or smaller) on two NLP classification tasks:
- **SST-2**: English sentiment analysis (binary classification)
- **XNLI**: Multilingual natural language inference (entailment/contradiction/neutral)

The models included are:
- **BitNet b1.58** (2B parameters, 1-bit weight quantization)
- **DistilBERT** (66M, distilled BERT baseline that preserves 98% of capabilities from base model)
- **GPT-Neo 2.7B** (2.7B GPT-based model, using 8-bit quantization for fine-tuning)
*(See [README](#) for full details on each model.)*

We provide scripts to fine-tune each model on the tasks and evaluate accuracy, F1, and runtime.

## Setup

1. **Install dependencies**: Create a Python 3.10+ environment with PyTorch (with CUDA). Then install required packages:
```bash
pip install -r requirements.txt
```
Note: If using BitNet, make sure to install the special Transformers version as noted below.

2. **(Optional) Download datasets**: The scripts will automatically download SST-2 and XNLI/MultiNLI via the HuggingFace Datasets library. Ensure you have internet access for the first run, or pre-download and cache the datasets.

3. **BitNet Transformers fork**: BitNet requires a specific fork of Hugging Face Transformers for full support. If you encounter issues loading BitNet, install the recommended version:
```bash
pip install git+https://github.com/huggingface/transformers.git@096f25ae1f501a084d8ff2dcaf25fbc2bd60eba4
```
This provides the BitLinear implementation needed for BitNet.

## How to Run Experiments
### Fine-tuning
Use the `train.py` script to fine-tune a model on a task. Training parameters can be specified via command-line arguments or by editing `config.yaml`. For example:

- Fine-tune BitNet on SST-2 (using 1k training samples, 3 epochs):
```bash
python train.py --task sst2 --model bitnet --train_samples 1000 --epochs 3 --batch_size 8 --bf16
```

- Fine-tune DistilBERT on SST-2 (full training set, 3 epochs):
```bash
python train.py --task sst2 --model distilbert --epochs 3 --batch_size 16
```

- Fine-tune GPT-Neo 2.7B on MultiNLI for XNLI (10k samples, 2 epochs, 8-bit precision):
```bash
python train.py --task xnli --model gptneo --train_samples 10000 --epochs 2 --batch_size 4 --fp16 --quant8
```

This will fine-tune on the MultiNLI English training subset and save a model for XNLI.

These commands will save model checkpoints (and tokenizer) to a `outputs/{model}-{task}/` directory by default, and print out validation metrics during training.

### Evaluation

After training, use `eval.py` to evaluate on the dev/test sets:
- Evaluate BitNet model on SST-2 validation:
```bash
python eval.py --task sst2 --model bitnet
```
- Evaluate BitNet model on XNLI (English and Spanish dev sets):
```bash
python eval.py --task xnli --model bitnet --languages en,es
```
(Use ISO language codes separated by commas to specify which XNLI languages to test. By default, English (en) is evaluated.)

The evaluation script will load the fine-tuned model from `outputs/{model}-{task}/` and print Accuracy, F1, and average inference time per sample for the specified evaluation sets.

## Results
After running the above experiments, you can expect output logs summarizing the performance. For example:
- SST-2 (binary sentiment):
    - DistilBERT: ~90% accuracy, F1 ≈ 0.90, ~X ms/sample
    - BitNet b1.58: ~92% accuracy, F1 ≈ 0.92, ~Y ms/sample
    - GPT-Neo 2.7B (8-bit): ~91% accuracy, F1 ≈ 0.91, ~Z ms/sample
- XNLI (multilingual NLI, evaluated zero-shot on other languages):
    - DistilBERT (multilingual): e.g. 78% accuracy on English, 70% on Spanish
    - BitNet b1.58: similar English performance (~79%), moderate drops in other languages (due to limited multilingual training)
    - GPT-Neo 2.7B: similar trend as BitNet (good English, lower on non-English)

(These are illustrative values; actual results will be printed by the scripts.) 
We also measure throughput: e.g., DistilBERT might process N samples/sec, whereas BitNet/GPT-Neo (with 30× more parameters) might handle M samples/sec (hence higher ms per sample). Quantization helps reduce memory usage and may slightly improve speed.

## Licence and Citation
This project uses open-source models:
- BitNet b1.58 by Microsoft Research (MIT License)
- DistilBERT by Hugging Face (Apache 2.0)
- GPT-Neo by EleutherAI (MIT License)
Please refer to the model licenses and the original papers for more information. 