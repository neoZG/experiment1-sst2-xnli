import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import os
from pathlib import Path

def load_yaml(file_path):
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def setup_plotting_style():
    # Use seaborn's default style
    sns.set_theme()
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = [12, 8]
    plt.rcParams['font.size'] = 12
    # Disable interactive mode
    plt.ioff()

def save_plot(name, dpi=300):
    """Save plot in both PNG and SVG formats"""
    plt.savefig(f'plots/{name}.png', dpi=dpi, bbox_inches='tight')
    plt.savefig(f'plots/{name}.svg', format='svg', bbox_inches='tight')
    plt.close()

def plot_sst2_comparison(results_dir="outputs"):
    # Load results for each model
    models = ['bitnet', 'distilbert', 'gptneo']
    metrics = {model: load_yaml(f"{results_dir}/{model}-sst2/eval_results.yaml") for model in models}
    
    # Prepare data
    data = {
        'Model': [],
        'Accuracy (%)': [],
        'F1 Score': [],
        'Latency (ms)': [],
        'Memory (GB)': []
    }
    
    for model in models:
        m = metrics[model]
        data['Model'].append(model.upper())
        data['Accuracy (%)'].append(m['results_by_language']['en']['accuracy'])
        data['F1 Score'].append(m['results_by_language']['en']['f1'])
        data['Latency (ms)'].append(m['results_by_language']['en']['avg_time_ms'])
        data['Memory (GB)'].append(m['peak_gpu_memory_gb'])
    
    df = pd.DataFrame(data)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('SST-2 Performance Comparison', fontsize=16)
    
    # Accuracy plot
    sns.barplot(data=df, x='Model', y='Accuracy (%)', ax=axes[0,0])
    axes[0,0].set_title('Accuracy')
    axes[0,0].set_ylim(85, 95)
    
    # F1 Score plot
    sns.barplot(data=df, x='Model', y='F1 Score', ax=axes[0,1])
    axes[0,1].set_title('F1 Score')
    axes[0,1].set_ylim(0.85, 0.95)
    
    # Latency plot
    sns.barplot(data=df, x='Model', y='Latency (ms)', ax=axes[1,0])
    axes[1,0].set_title('Inference Latency')
    
    # Memory plot
    sns.barplot(data=df, x='Model', y='Memory (GB)', ax=axes[1,1])
    axes[1,1].set_title('Peak Memory Usage')
    
    plt.tight_layout()
    save_plot('sst2_comparison')

def plot_xnli_comparison(results_dir="outputs"):
    # Load results for each model
    models = ['bitnet', 'distilbert', 'gptneo']
    metrics = {model: load_yaml(f"{results_dir}/{model}-xnli/eval_results_en-es-fr-de-zh.yaml") for model in models}
    
    # Prepare cross-lingual data
    languages = ['en', 'es', 'fr', 'de', 'zh']
    data = {
        'Model': [],
        'Language': [],
        'Accuracy (%)': [],
        'F1 Score': [],
        'Latency (ms)': []
    }
    
    for model in models:
        m = metrics[model]
        for lang in languages:
            data['Model'].append(model.upper())
            data['Language'].append(lang.upper())
            data['Accuracy (%)'].append(m['results_by_language'][lang]['accuracy'])
            data['F1 Score'].append(m['results_by_language'][lang]['f1'])
            data['Latency (ms)'].append(m['results_by_language'][lang]['avg_time_ms'])
    
    df = pd.DataFrame(data)
    
    # Create plots
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 1, 1)
    sns.barplot(data=df, x='Language', y='Accuracy (%)', hue='Model')
    plt.title('XNLI Cross-lingual Accuracy')
    plt.ylim(70, 85)
    
    plt.subplot(2, 1, 2)
    sns.barplot(data=df, x='Language', y='Latency (ms)', hue='Model')
    plt.title('XNLI Cross-lingual Latency')
    
    plt.tight_layout()
    save_plot('xnli_comparison')

def plot_training_curves():
    # Load training metrics for each model
    models = ['bitnet', 'distilbert', 'gptneo']
    dfs = {}
    
    for model in models:
        df = pd.read_csv(f"outputs/{model}-sst2/training_metrics.csv")
        dfs[model] = df
    
    # Plot training loss
    plt.figure(figsize=(12, 8))
    for model in models:
        plt.plot(dfs[model]['epoch'], dfs[model]['train_loss'], label=f"{model.upper()} Training Loss")
        plt.scatter(dfs[model]['epoch'][dfs[model]['eval_loss'].notna()], 
                   dfs[model]['eval_loss'][dfs[model]['eval_loss'].notna()],
                   label=f"{model.upper()} Validation Loss")
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    save_plot('training_curves')

def plot_resource_usage():
    # Load resource usage for each model
    models = ['bitnet', 'distilbert', 'gptneo']
    data = {
        'Model': [],
        'GPU Memory (GB)': [],
        'Training Time (min)': [],
        'GPU Utilization (%)': [],
        'Power Draw (W)': []
    }
    
    for model in models:
        metrics = load_yaml(f"outputs/{model}-sst2/resource_usage.yaml")
        data['Model'].append(model.upper())
        data['GPU Memory (GB)'].append(metrics['peak_gpu_memory_gb'])
        data['Training Time (min)'].append(metrics['training_time_minutes'])
        data['GPU Utilization (%)'].append(metrics['gpu_utilization_avg'])
        data['Power Draw (W)'].append(metrics['gpu_power_draw_watts'])
    
    df = pd.DataFrame(data)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Resource Usage Comparison', fontsize=16)
    
    sns.barplot(data=df, x='Model', y='GPU Memory (GB)', ax=axes[0,0])
    axes[0,0].set_title('Peak GPU Memory')
    
    sns.barplot(data=df, x='Model', y='Training Time (min)', ax=axes[0,1])
    axes[0,1].set_title('Training Time')
    
    sns.barplot(data=df, x='Model', y='GPU Utilization (%)', ax=axes[1,0])
    axes[1,0].set_title('GPU Utilization')
    
    sns.barplot(data=df, x='Model', y='Power Draw (W)', ax=axes[1,1])
    axes[1,1].set_title('Power Consumption')
    
    plt.tight_layout()
    save_plot('resource_usage')

if __name__ == "__main__":
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Setup plotting style
    setup_plotting_style()
    
    print("Generating plots...")
    
    # Generate all plots
    plot_sst2_comparison()
    print("- SST-2 comparison plots saved")
    
    plot_xnli_comparison()
    print("- XNLI comparison plots saved")
    
    plot_training_curves()
    print("- Training curves saved")
    
    plot_resource_usage()
    print("- Resource usage plots saved")
    
    print("\nAll plots have been generated in both PNG and SVG formats in the 'plots' directory.") 