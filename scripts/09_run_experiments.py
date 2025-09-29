# scripts/run_experiments.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import json
from pathlib import Path
import rootutils
import sys
from datetime import datetime

root_path = rootutils.find_root()
sys.path.append(str(root_path))

from burned_embedder.classifier.dataset import create_datasets
from burned_embedder.classifier.model import get_model
from burned_embedder.classifier.model_utils import (
    train_epoch, evaluate, EarlyStopping, print_metrics
)
from burned_embedder.utils import setup_device

# Import experiment configurations
sys.path.append(str(root_path / 'configs'))
from experiment_configs import EXPERIMENTS


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def run_single_experiment(exp_name, config, device):
    """Run a single experiment with given configuration"""
    print(f"\n{'='*80}")
    print(f"Running Experiment: {exp_name}")
    print(f"{'='*80}")
    print(f"Config: {json.dumps(config, indent=2)}")
    
    # Set seed
    set_seed(config['seed'])
    
    # Create datasets with specified input type
    train_dataset, val_dataset, test_dataset = create_datasets(
        test_size=config['test_size'],
        val_split=config['val_split'],
        random_state=config['seed'],
        input_type=config['input_type']
    )
    
    # Get input dimension
    sample_embedding, _ = train_dataset[0]
    input_dim = sample_embedding.shape[0]
    print(f"\nInput dimension: {input_dim}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)
    
    # Create model
    model = get_model(
        input_dim=input_dim,
        hidden_dims=config['hidden_dims'],
        dropout=config['dropout'],
        device=device
    )
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config['lr'],
        weight_decay=config.get('weight_decay', 0.0)
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=config['patience'], mode='max')
    
    # Create output directory
    exp_dir = root_path / "experiments" / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    print(f"\n{'='*80}")
    print("Training")
    print(f"{'='*80}")
    
    best_val_f1 = 0
    history = {
        'train_loss': [], 'train_f1': [],
        'val_loss': [], 'val_f1': [], 'val_precision': [], 'val_recall': [], 'val_auc': []
    }
    
    for epoch in range(1, config['epochs'] + 1):
        # Train
        train_loss, train_f1, train_precision, train_recall = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_f1, val_precision, val_recall, val_auc, val_cm = evaluate(
            model, val_loader, criterion, device
        )
        
        # Store history
        history['train_loss'].append(train_loss)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_auc'].append(val_auc)
        
        # Print metrics
        print_metrics(epoch, train_loss, train_f1, val_loss, val_f1, val_precision, val_recall, val_auc)
        
        # Learning rate scheduling
        scheduler.step(val_f1)
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'config': config
            }, exp_dir / "best_model.pt")
        
        # Early stopping check
        if early_stopping(val_f1):
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break
    
    # Save training history
    with open(exp_dir / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    # Load best model for final evaluation
    checkpoint = torch.load(exp_dir / "best_model.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test evaluation
    test_loss, test_f1, test_precision, test_recall, test_auc, test_cm = evaluate(
        model, test_loader, criterion, device
    )
    
    # Compute overfitting gap
    best_train_f1 = max(history['train_f1'])
    overfit_gap = best_train_f1 - test_f1
    
    results = {
        'experiment_name': exp_name,
        'config': config,
        'best_epoch': checkpoint['epoch'],
        'best_val_f1': float(best_val_f1),
        'test_loss': float(test_loss),
        'test_f1': float(test_f1),
        'test_precision': float(test_precision),
        'test_recall': float(test_recall),
        'test_roc_auc': float(test_auc),
        'confusion_matrix': test_cm.tolist(),
        'best_train_f1': float(best_train_f1),
        'overfit_gap': float(overfit_gap)
    }
    
    # Save results
    with open(exp_dir / "test_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Experiment {exp_name} Complete")
    print(f"{'='*80}")
    print(f"Best Val F1:    {best_val_f1:.4f} (epoch {checkpoint['epoch']})")
    print(f"Test F1:        {test_f1:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall:    {test_recall:.4f}")
    print(f"Test ROC-AUC:   {test_auc:.4f}")
    print(f"Overfit Gap:    {overfit_gap:.4f} (train F1 - test F1)")
    print(f"Results saved to {exp_dir}")
    
    return results


def main():
    """Run all experiments"""
    print("="*80)
    print("SYSTEMATIC EXPERIMENT SUITE")
    print("="*80)
    
    device = setup_device(gpu_index=1, memory_fraction=1.0)
    print(f"Using device: {device}\n")
    
    # Run all experiments
    all_results = []
    
    for exp_name, config in EXPERIMENTS.items():
        try:
            results = run_single_experiment(exp_name, config, device)
            all_results.append(results)
        except Exception as e:
            print(f"\n✗ Experiment {exp_name} failed with error: {e}")
            continue
    
    # Save summary
    summary_dir = root_path / "experiments_more"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    with open(summary_dir / f"summary_{timestamp}.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print comparison table
    print("\n" + "="*80)
    print("EXPERIMENT COMPARISON")
    print("="*80)
    print(f"{'Experiment':<30} {'Test F1':<10} {'Precision':<12} {'Recall':<10} {'AUC':<10} {'Overfit':<10}")
    print("-"*80)
    
    for result in sorted(all_results, key=lambda x: x['test_f1'], reverse=True):
        print(f"{result['experiment_name']:<30} "
              f"{result['test_f1']:<10.4f} "
              f"{result['test_precision']:<12.4f} "
              f"{result['test_recall']:<10.4f} "
              f"{result['test_roc_auc']:<10.4f} "
              f"{result['overfit_gap']:<10.4f}")
    
    print("\n✓ All experiments complete!")
    print(f"Summary saved to {summary_dir / f'summary_{timestamp}.json'}")


if __name__ == "__main__":
    main()