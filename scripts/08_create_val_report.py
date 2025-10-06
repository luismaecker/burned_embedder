# scripts/08_create_val_report.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import rootutils
import sys
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score, 
    confusion_matrix, roc_curve, precision_recall_curve, average_precision_score
)

root_path = rootutils.find_root()
sys.path.append(str(root_path))

from deforestation_embedder.classifier.dataset import load_embedding_paths_and_labels, DeforestationDataset
from deforestation_embedder.classifier.model import get_model
from deforestation_embedder.utils import setup_device


def load_continent_dataset(continent_name, input_type='concat', batch_size=32):
    """Load dataset for a specific continent"""
    pos_paths, pos_labels = load_embedding_paths_and_labels('positive', continent=continent_name)
    neg_paths, neg_labels = load_embedding_paths_and_labels('negative', continent=continent_name)
    
    all_paths = pos_paths + neg_paths
    all_labels = pos_labels + neg_labels
    
    dataset = DeforestationDataset(
        all_paths, all_labels, 
        input_type=input_type, 
        normalize=True, 
        augment=False
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4
    )
    
    print(f"{continent_name.replace('_', ' ').title()}: {len(dataset)} samples "
          f"(pos: {sum(all_labels)}, neg: {len(all_labels) - sum(all_labels)})")
    
    return dataloader, all_labels, all_paths


def load_metadata_for_events(continent_name, sample_type='positive'):
    """Load metadata for all events in a continent"""
    embeddings_dir = root_path / "data" / "processed" / "embeddings" / continent_name / sample_type
    
    metadata_list = []
    event_dirs = sorted([d for d in embeddings_dir.iterdir() if d.is_dir() and d.name.startswith('event_')])
    
    for event_dir in event_dirs:
        metadata_path = event_dir / "metadata.npy"
        if metadata_path.exists():
            meta = np.load(metadata_path, allow_pickle=True).item()
            meta['event_dir'] = event_dir
            metadata_list.append(meta)
    
    return metadata_list


def evaluate_model(model, dataloader, device):
    """Run model evaluation and collect predictions"""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for embeddings, labels in dataloader:
            embeddings = embeddings.to(device)
            labels = labels.to(device)
            
            outputs = model(embeddings)
            
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Compute metrics
    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    roc_auc = roc_auc_score(all_labels, all_probs)
    avg_precision = average_precision_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)
    
    # Error rates
    tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    results = {
        'predictions': all_preds,
        'probabilities': all_probs,
        'labels': all_labels,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'roc_auc': roc_auc,
        'avg_precision': avg_precision,
        'confusion_matrix': cm,
        'fpr': fpr,
        'fnr': fnr,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
    }
    
    return results


def evaluate_by_event_size(model, continent_name, config, device):
    """Evaluate model performance by deforestation event size"""
    # Load positive events metadata
    metadata_list = load_metadata_for_events(continent_name, 'positive')
    
    # Create size bins
    sizes = [m['size_pixels'] for m in metadata_list]
    size_bins = [0, 100, 200, 500, np.inf]
    bin_labels = ['0-100', '100-200', '200-500', '500+']
    
    results_by_size = {}
    
    for i, (lower, upper) in enumerate(zip(size_bins[:-1], size_bins[1:])):
        # Filter events in this size range
        events_in_range = [m for m in metadata_list if lower <= m['size_pixels'] < upper]
        
        if len(events_in_range) == 0:
            continue
        
        # Load embeddings for these events
        paths = []
        labels = []
        for event_meta in events_in_range:
            event_dir = event_meta['event_dir']
            before_path = event_dir / "embedd_before.npy"
            after_path = event_dir / "embedd_after.npy"
            if before_path.exists() and after_path.exists():
                paths.append((before_path, after_path))
                labels.append(1)  # positive
        
        if len(paths) == 0:
            continue
        
        # Create dataset
        dataset = DeforestationDataset(
            paths, labels,
            input_type=config['input_type'],
            normalize=True,
            augment=False
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=4
        )
        
        # Evaluate
        model.eval()
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for embeddings, _ in dataloader:
                embeddings = embeddings.to(device)
                outputs = model(embeddings)
                preds = (outputs > 0.5).float()
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(outputs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        
        # Calculate metrics
        accuracy = np.mean(all_preds == 1)  # all should be positive
        mean_prob = np.mean(all_probs)
        
        results_by_size[bin_labels[i]] = {
            'count': len(all_preds),
            'accuracy': accuracy,
            'mean_probability': mean_prob,
            'correct': int(np.sum(all_preds == 1)),
            'incorrect': int(np.sum(all_preds == 0))
        }
    
    return results_by_size


def plot_south_america_summary(results, save_path):
    """Create summary plot for South America with confusion matrix and metrics"""
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2], hspace=0.3)
    
    # Confusion Matrix
    ax1 = fig.add_subplot(gs[0])
    cm = results['confusion_matrix']
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        ax=ax1,
        cbar=False,
        square=True,
        annot_kws={'size': 14, 'weight': 'bold'}
    )
    ax1.set_title('Confusion Matrix', fontsize=13, fontweight='bold', pad=10)
    ax1.set_ylabel('Actual', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Predicted', fontsize=11, fontweight='bold')
    ax1.set_xticklabels(['Negative', 'Positive'], fontsize=10)
    ax1.set_yticklabels(['Negative', 'Positive'], fontsize=10, rotation=0)
    
    # Metrics table
    ax2 = fig.add_subplot(gs[1])
    ax2.axis('tight')
    ax2.axis('off')
    
    metrics_data = [
        ['Metric', 'Value'],
        ['F1 Score', f"{results['f1']:.4f}"],
        ['Precision', f"{results['precision']:.4f}"],
        ['Recall', f"{results['recall']:.4f}"],
        ['ROC-AUC', f"{results['roc_auc']:.4f}"],
        ['Avg Precision', f"{results['avg_precision']:.4f}"],
        ['', ''],
        ['False Positive Rate', f"{results['fpr']:.4f}"],
        ['False Negative Rate', f"{results['fnr']:.4f}"],
        ['', ''],
        ['Total Samples', f"{len(results['labels'])}"],
        ['True Positives', f"{results['tp']}"],
        ['True Negatives', f"{results['tn']}"],
        ['False Positives', f"{results['fp']}"],
        ['False Negatives', f"{results['fn']}"]
    ]
    
    table = ax2.table(
        cellText=metrics_data,
        cellLoc='left',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    
    # Style the table
    for i in range(len(metrics_data)):
        cell = table[(i, 0)]
        cell.set_text_props(weight='bold')
        
        if i == 0:  # Header row
            cell.set_facecolor('#3498db')
            cell.set_text_props(color='white', weight='bold', size=12)
            table[(i, 1)].set_facecolor('#3498db')
            table[(i, 1)].set_text_props(color='white', weight='bold', size=12)
        elif i in [6, 9]:  # Separator rows
            cell.set_facecolor('#ecf0f1')
            table[(i, 1)].set_facecolor('#ecf0f1')
        else:
            if i % 2 == 0:
                cell.set_facecolor('#f8f9fa')
                table[(i, 1)].set_facecolor('#f8f9fa')
    
    ax2.set_title('Performance Metrics', fontsize=13, fontweight='bold', pad=20)
    
    plt.suptitle('South America Test Set - Model Performance Summary',
                 fontsize=15, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_performance_by_size(results_by_size, save_path):
    """Plot model performance by deforestation event size"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    size_bins = list(results_by_size.keys())
    accuracies = [results_by_size[bin]['accuracy'] for bin in size_bins]
    mean_probs = [results_by_size[bin]['mean_probability'] for bin in size_bins]
    counts = [results_by_size[bin]['count'] for bin in size_bins]
    
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
    
    # Accuracy by size
    bars1 = ax1.bar(size_bins, accuracies, color=colors[:len(size_bins)], 
                    alpha=0.8, edgecolor='black', linewidth=1.5)
    
    for bar, count in zip(bars1, counts):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}\n(n={count})',
            ha='center', va='bottom', fontsize=10, fontweight='bold'
        )
    
    ax1.set_xlabel('Event Size (pixels)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Detection Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Detection Accuracy by Event Size', fontsize=13, fontweight='bold', pad=10)
    ax1.set_ylim([0, 1.05])
    ax1.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax1.axhline(y=1.0, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Perfect')
    ax1.legend()
    
    # Mean probability by size
    bars2 = ax2.bar(size_bins, mean_probs, color=colors[:len(size_bins)], 
                    alpha=0.8, edgecolor='black', linewidth=1.5)
    
    for bar, count in zip(bars2, counts):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}\n(n={count})',
            ha='center', va='bottom', fontsize=10, fontweight='bold'
        )
    
    ax2.set_xlabel('Event Size (pixels)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Mean Prediction Probability', fontsize=12, fontweight='bold')
    ax2.set_title('Mean Confidence by Event Size', fontsize=13, fontweight='bold', pad=10)
    ax2.set_ylim([0, 1.05])
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax2.axhline(y=0.5, color='black', linestyle='--', linewidth=2, alpha=0.5, label='Threshold')
    ax2.legend()
    
    plt.suptitle('South America - Performance by Deforestation Event Size',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_confusion_matrices(results_dict, save_path):
    """Plot confusion matrices for all datasets side by side"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, (dataset_name, results) in enumerate(results_dict.items()):
        cm = results['confusion_matrix']
        
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues', 
            ax=axes[idx],
            cbar=False,
            square=True
        )
        
        axes[idx].set_title(f'{dataset_name}\nF1: {results["f1"]:.3f}', 
                           fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('Actual', fontsize=10)
        axes[idx].set_xlabel('Predicted', fontsize=10)
        axes[idx].set_xticklabels(['Negative', 'Positive'])
        axes[idx].set_yticklabels(['Negative', 'Positive'])
    
    plt.suptitle('Confusion Matrices Across Tropical Rainforest Regions',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_roc_curves(results_dict, save_path):
    """Plot ROC curves for all datasets"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    for idx, (dataset_name, results) in enumerate(results_dict.items()):
        fpr, tpr, _ = roc_curve(results['labels'], results['probabilities'])
        
        ax.plot(
            fpr, tpr, 
            color=colors[idx], 
            linewidth=2.5,
            label=f'{dataset_name} (AUC = {results["roc_auc"]:.3f})'
        )
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, label='Random')
    
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('ROC Curves Across Tropical Rainforest Regions', 
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=11, frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_precision_recall_curves(results_dict, save_path):
    """Plot Precision-Recall curves for all datasets"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    for idx, (dataset_name, results) in enumerate(results_dict.items()):
        precision, recall, _ = precision_recall_curve(
            results['labels'], 
            results['probabilities']
        )
        
        ax.plot(
            recall, precision,
            color=colors[idx],
            linewidth=2.5,
            label=f'{dataset_name} (AP = {results["avg_precision"]:.3f})'
        )
    
    ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax.set_title('Precision-Recall Curves Across Tropical Rainforest Regions',
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='lower left', fontsize=11, frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_metrics_comparison(results_dict, save_path):
    """Bar chart comparing all metrics across datasets"""
    metrics = ['f1', 'precision', 'recall', 'roc_auc']
    metric_labels = ['F1 Score', 'Precision', 'Recall', 'ROC-AUC']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    dataset_names = list(results_dict.keys())
    
    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        values = [results_dict[name][metric] for name in dataset_names]
        
        bars = axes[idx].bar(dataset_names, values, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[idx].text(
                bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold'
            )
        
        axes[idx].set_ylabel(label, fontsize=12, fontweight='bold')
        axes[idx].set_ylim([0, 1.05])
        axes[idx].grid(True, alpha=0.3, linestyle='--', axis='y')
        axes[idx].set_title(label, fontsize=13, fontweight='bold', pad=10)
    
    plt.suptitle('Performance Metrics Across Tropical Rainforest Regions', 
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_error_analysis(results_dict, save_path):
    """Plot false positive and false negative rates"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    dataset_names = list(results_dict.keys())
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    # False Positive Rate
    fpr_values = [results_dict[name]['fpr'] for name in dataset_names]
    bars1 = ax1.bar(dataset_names, fpr_values, color=colors, alpha=0.8, edgecolor='black')
    
    for bar in bars1:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold'
        )
    
    ax1.set_ylabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax1.set_title('False Positive Rate by Region', fontsize=13, fontweight='bold', pad=10)
    ax1.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax1.set_ylim([0, max(fpr_values) * 1.2])
    
    # False Negative Rate
    fnr_values = [results_dict[name]['fnr'] for name in dataset_names]
    bars2 = ax2.bar(dataset_names, fnr_values, color=colors, alpha=0.8, edgecolor='black')
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold'
        )
    
    ax2.set_ylabel('False Negative Rate', fontsize=12, fontweight='bold')
    ax2.set_title('False Negative Rate by Region', fontsize=13, fontweight='bold', pad=10)
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax2.set_ylim([0, max(fnr_values) * 1.2])
    
    plt.suptitle('Error Analysis Across Tropical Rainforest Regions', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_probability_distributions(results_dict, save_path):
    """Plot prediction probability distributions for positive and negative classes"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    for idx, (dataset_name, results) in enumerate(results_dict.items()):
        ax = axes[idx]
        
        # Separate probabilities by true label
        pos_probs = results['probabilities'][results['labels'] == 1]
        neg_probs = results['probabilities'][results['labels'] == 0]
        
        ax.hist(neg_probs, bins=50, alpha=0.6, color='#E63946', 
                label=f'True Negative (n={len(neg_probs)})', edgecolor='black')
        ax.hist(pos_probs, bins=50, alpha=0.6, color='#06D6A0', 
                label=f'True Positive (n={len(pos_probs)})', edgecolor='black')
        
        ax.axvline(0.5, color='black', linestyle='--', linewidth=2, label='Threshold (0.5)')
        
        ax.set_xlabel('Predicted Probability', fontsize=11, fontweight='bold')
        ax.set_ylabel('Count', fontsize=11, fontweight='bold')
        ax.set_title(f'{dataset_name} - Prediction Distribution', 
                    fontsize=12, fontweight='bold')
        ax.legend(loc='upper center', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    plt.suptitle('Prediction Probability Distributions Across Tropical Rainforest Regions', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_sample_counts(results_dict, save_path):
    """Plot sample counts and class distribution"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    dataset_names = list(results_dict.keys())
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    # Total samples
    total_samples = [len(results_dict[name]['labels']) for name in dataset_names]
    bars1 = ax1.bar(dataset_names, total_samples, color=colors, alpha=0.8, edgecolor='black')
    
    for bar in bars1:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}',
            ha='center', va='bottom', fontsize=11, fontweight='bold'
        )
    
    ax1.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    ax1.set_title('Total Samples by Region', fontsize=13, fontweight='bold', pad=10)
    ax1.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Class distribution
    x = np.arange(len(dataset_names))
    width = 0.35
    
    pos_counts = [results_dict[name]['tp'] + results_dict[name]['fn'] for name in dataset_names]
    neg_counts = [results_dict[name]['tn'] + results_dict[name]['fp'] for name in dataset_names]
    
    bars2 = ax2.bar(x - width/2, pos_counts, width, label='Positive', 
                    color='#06D6A0', alpha=0.8, edgecolor='black')
    bars3 = ax2.bar(x + width/2, neg_counts, width, label='Negative', 
                    color='#E63946', alpha=0.8, edgecolor='black')
    
    # Add value labels
    for bars in [bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10, fontweight='bold'
            )
    
    ax2.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    ax2.set_title('Class Distribution by Region', fontsize=13, fontweight='bold', pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(dataset_names)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    plt.suptitle('Sample Distribution Across Tropical Rainforest Regions',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def print_detailed_results(dataset_name, results):
    """Print detailed metrics for a dataset"""
    print(f"\n{'='*70}")
    print(f"{dataset_name}")
    print(f"{'='*70}")
    
    print(f"\nClassification Metrics:")
    print(f"  F1 Score:       {results['f1']:.4f}")
    print(f"  Precision:      {results['precision']:.4f}")
    print(f"  Recall:         {results['recall']:.4f}")
    print(f"  ROC-AUC:        {results['roc_auc']:.4f}")
    print(f"  Avg Precision:  {results['avg_precision']:.4f}")
    
    print(f"\nConfusion Matrix:")
    cm = results['confusion_matrix']
    print(f"                Predicted")
    print(f"              Neg    Pos")
    print(f"  Actual Neg [{cm[0,0]:4d}  {cm[0,1]:4d}]")
    print(f"         Pos [{cm[1,0]:4d}  {cm[1,1]:4d}]")
    
    print(f"\nError Analysis:")
    print(f"  False Positive Rate: {results['fpr']:.4f} ({results['fp']}/{results['fp']+results['tn']})")
    print(f"  False Negative Rate: {results['fnr']:.4f} ({results['fn']}/{results['fn']+results['tp']})")

    print(f"\nSample Distribution:")
    print(f"  Total Samples: {len(results['labels'])}")
    print(f"  Positive: {results['tp'] + results['fn']}")
    print(f"  Negative: {results['tn'] + results['fp']}")


def save_results_to_json(results_dict, model_config, save_path):
    """Save all results to JSON file"""
    output = {
        'model_config': model_config,
        'datasets': {}
    }
    
    for dataset_name, results in results_dict.items():
        output['datasets'][dataset_name] = {
            'f1': float(results['f1']),
            'precision': float(results['precision']),
            'recall': float(results['recall']),
            'roc_auc': float(results['roc_auc']),
            'avg_precision': float(results['avg_precision']),
            'confusion_matrix': results['confusion_matrix'].tolist(),
            'fpr': float(results['fpr']),
            'fnr': float(results['fnr']),
            'tp': int(results['tp']),
            'fp': int(results['fp']),
            'tn': int(results['tn']),
            'fn': int(results['fn']),
            'total_samples': int(len(results['labels'])),
            'positive_samples': int(results['tp'] + results['fn']),
            'negative_samples': int(results['tn'] + results['fp'])
        }
    
    with open(save_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nSaved: {save_path}")


def save_size_analysis_to_json(results_by_size, save_path):
    """Save size analysis results to JSON"""
    output = {}
    
    for size_bin, results in results_by_size.items():
        output[size_bin] = {
            'count': int(results['count']),
            'accuracy': float(results['accuracy']),
            'mean_probability': float(results['mean_probability']),
            'correct': int(results['correct']),
            'incorrect': int(results['incorrect'])
        }
    
    with open(save_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nSaved: {save_path}")


def create_summary_table(results_dict, save_path):
    """Create a summary table comparing all datasets"""
    data = []
    
    for dataset_name, results in results_dict.items():
        data.append({
            'Dataset': dataset_name,
            'F1': f"{results['f1']:.4f}",
            'Precision': f"{results['precision']:.4f}",
            'Recall': f"{results['recall']:.4f}",
            'ROC-AUC': f"{results['roc_auc']:.4f}",
            'Avg Precision': f"{results['avg_precision']:.4f}",
            'FPR': f"{results['fpr']:.4f}",
            'FNR': f"{results['fnr']:.4f}",
            'Total Samples': len(results['labels']),
            'Positive': results['tp'] + results['fn'],
            'Negative': results['tn'] + results['fp']
        })
    
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)
    
    print(f"\nSaved: {save_path}")
    print("\nSummary Table:")
    print(df.to_string(index=False))


def create_size_analysis_table(results_by_size, save_path):
    """Create a summary table for size analysis"""
    data = []
    
    for size_bin, results in results_by_size.items():
        data.append({
            'Size Bin (pixels)': size_bin,
            'Count': results['count'],
            'Accuracy': f"{results['accuracy']:.4f}",
            'Mean Probability': f"{results['mean_probability']:.4f}",
            'Correct': results['correct'],
            'Incorrect': results['incorrect']
        })
    
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)
    
    print(f"\nSaved: {save_path}")
    print("\nSize Analysis Table:")
    print(df.to_string(index=False))


def main():
    """Main validation report generation"""
    print("="*70)
    print("VALIDATION REPORT GENERATION")
    print("="*70)
    
    # Setup
    device = setup_device(gpu_index=1, memory_fraction=1.0)
    print(f"Using device: {device}\n")
    
    # Load model config
    model_path = root_path / "models" / "final_classifier" / "best_model.pt"
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    print("Model Configuration:")
    print(json.dumps(config, indent=2))
    
    # Create output directory
    output_dir = root_path / "reports" / "figures" / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Determine actual input dimension from checkpoint
    first_layer_weight = checkpoint['model_state_dict']['model.0.weight']
    actual_input_dim = first_layer_weight.shape[1]
    print(f"\nDetected input dimension from checkpoint: {actual_input_dim}")
    
    # Load model with correct dimensions
    print("\nLoading trained model...")
    model = get_model(
        input_dim=actual_input_dim,
        hidden_dims=config['hidden_dims'],
        dropout=config['dropout'],
        device=device
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']} with Val F1: {checkpoint['val_f1']:.4f}")
    
    # Load datasets
    print(f"\n{'='*70}")
    print("Loading Datasets")
    print(f"{'='*70}")
    
    # Test set - South America (from training)
    from deforestation_embedder.classifier.dataset import create_datasets
    _, _, test_dataset = create_datasets(
        test_size=config['test_size'],
        val_split=config['val_split'],
        random_state=config['seed'],
        input_type=config['input_type'],
        augment=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4
    )
    print(f"South America (Test): {len(test_dataset)} samples")
    
    # Southeast Asia
    sea_loader, sea_labels, sea_paths = load_continent_dataset(
        'southeast_asia',
        input_type=config['input_type'],
        batch_size=config['batch_size']
    )
    
    # Africa
    africa_loader, africa_labels, africa_paths = load_continent_dataset(
        'africa',
        input_type=config['input_type'],
        batch_size=config['batch_size']
    )
    
    # Evaluate on all datasets
    print(f"\n{'='*70}")
    print("Running Evaluations")
    print(f"{'='*70}")
    
    print("\nEvaluating on South America (Test Set)...")
    test_results = evaluate_model(model, test_loader, device)
    
    print("\nEvaluating on Southeast Asia...")
    sea_results = evaluate_model(model, sea_loader, device)
    
    print("\nEvaluating on Africa...")
    africa_results = evaluate_model(model, africa_loader, device)
    
    # Evaluate by event size for South America
    print("\nAnalyzing South America by deforestation event size...")
    sa_size_results = evaluate_by_event_size(model, 'south_america', config, device)
    
    # Organize results
    results_dict = {
        'South America\n(Test)': test_results,
        'Southeast Asia': sea_results,
        'Africa': africa_results
    }
    
    # Print detailed results
    for dataset_name, results in results_dict.items():
        print_detailed_results(dataset_name, results)
    
    # Print size analysis
    print(f"\n{'='*70}")
    print("South America - Performance by Event Size")
    print(f"{'='*70}")
    for size_bin, results in sa_size_results.items():
        print(f"\nSize: {size_bin} pixels")
        print(f"  Count:           {results['count']}")
        print(f"  Accuracy:        {results['accuracy']:.4f}")
        print(f"  Mean Prob:       {results['mean_probability']:.4f}")
        print(f"  Correct:         {results['correct']}")
        print(f"  Incorrect:       {results['incorrect']}")
    
    # Generate visualizations
    print(f"\n{'='*70}")
    print("Generating Visualizations")
    print(f"{'='*70}\n")
    
    # South America summary
    plot_south_america_summary(
        test_results,
        output_dir / "south_america_summary.png"
    )
    
    # Size analysis
    if sa_size_results:
        plot_performance_by_size(
            sa_size_results,
            output_dir / "south_america_performance_by_size.png"
        )
    
    # Cross-region comparisons
    plot_confusion_matrices(
        results_dict,
        output_dir / "confusion_matrices.png"
    )
    
    plot_roc_curves(
        results_dict,
        output_dir / "roc_curves.png"
    )
    
    plot_precision_recall_curves(
        results_dict,
        output_dir / "precision_recall_curves.png"
    )
    
    plot_metrics_comparison(
        results_dict,
        output_dir / "metrics_comparison.png"
    )
    
    plot_error_analysis(
        results_dict,
        output_dir / "error_analysis.png"
    )
    
    plot_probability_distributions(
        results_dict,
        output_dir / "probability_distributions.png"
    )
    
    plot_sample_counts(
        results_dict,
        output_dir / "sample_counts.png"
    )
    
    # Save results
    print(f"\n{'='*70}")
    print("Saving Results")
    print(f"{'='*70}")
    
    save_results_to_json(
        results_dict,
        config,
        output_dir / "validation_results.json"
    )
    
    if sa_size_results:
        save_size_analysis_to_json(
            sa_size_results,
            output_dir / "south_america_size_analysis.json"
        )
    
    create_summary_table(
        results_dict,
        output_dir / "validation_summary.csv"
    )
    
    if sa_size_results:
        create_size_analysis_table(
            sa_size_results,
            output_dir / "south_america_size_analysis.csv"
        )
    
    print(f"\n{'='*70}")
    print("VALIDATION REPORT COMPLETE")
    print(f"{'='*70}")
    print(f"\nAll outputs saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - south_america_summary.png")
    print("  - south_america_performance_by_size.png")
    print("  - confusion_matrices.png")
    print("  - roc_curves.png")
    print("  - precision_recall_curves.png")
    print("  - metrics_comparison.png")
    print("  - error_analysis.png")
    print("  - probability_distributions.png")
    print("  - sample_counts.png")
    print("  - validation_results.json")
    print("  - south_america_size_analysis.json")
    print("  - validation_summary.csv")
    print("  - south_america_size_analysis.csv")


if __name__ == "__main__":
    main()