# scripts/train_final_model.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import json
from pathlib import Path
import rootutils
import sys

root_path = rootutils.find_root()
sys.path.append(str(root_path))

from burned_embedder.classifier.dataset import create_datasets
from burned_embedder.classifier.model import get_model
from burned_embedder.classifier.model_utils import (
    train_epoch, evaluate, EarlyStopping, print_metrics
)
from burned_embedder.utils import setup_device


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def main():
    """Train final production model using exact experiment configuration"""
    print("="*80)
    print("TRAINING FINAL PRODUCTION MODEL")
    print("="*80)
    
    # Exact configuration from best experiments
    config = {
        'test_size': 0.2,
        'val_split': 0.1,
        'seed': 2,
        'input_type': 'concat',
        'hidden_dims': [1400, 700, 350],
        'dropout': 0.4,
        'weight_decay': 1e-4,
        'batch_size': 32,
        'lr': 0.001,
        'epochs': 100,
        'patience': 15,
    }
    
    print(f"\nConfiguration:")
    print(json.dumps(config, indent=2))
    
    # Set seed
    set_seed(config['seed'])
    
    # Setup device
    device = setup_device(gpu_index=1, memory_fraction=1.0)
    print(f"\nUsing device: {device}")
    
    # Create datasets (no augmentation, same as experiments)
    print("\nCreating datasets...")
    train_dataset, val_dataset, test_dataset = create_datasets(
        test_size=config['test_size'],
        val_split=config['val_split'],
        random_state=config['seed'],
        input_type=config['input_type'],
        augment=False  # No augmentation -> better result
    )
    
    # Get input dimension
    sample_embedding, _ = train_dataset[0]
    input_dim = sample_embedding.shape[0]
    print(f"Input dimension: {input_dim}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=4
    )
    
    # Create model (same as experiments)
    model = get_model(
        input_dim=input_dim,
        hidden_dims=config['hidden_dims'],
        dropout=config['dropout'],
        device=device
    )
    
    # Loss and optimizer (same as experiments)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler (same as experiments)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # Early stopping (same as experiments)
    early_stopping = EarlyStopping(patience=config['patience'], mode='max')
    
    # Create output directory
    save_dir = root_path / "models" / "final_classifier"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    print(f"\n{'='*80}")
    print("Training")
    print(f"{'='*80}")
    
    best_val_f1 = 0
    history = {
        'train_loss': [], 'train_f1': [],
        'val_loss': [], 'val_f1': [], 'val_precision': [], 
        'val_recall': [], 'val_auc': []
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
        print_metrics(epoch, train_loss, train_f1, val_loss, val_f1, 
                     val_precision, val_recall, val_auc)
        
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
            }, save_dir / "best_model.pt")
            print(f"  → Saved new best model (F1: {val_f1:.4f})")
        
        # Early stopping check
        if early_stopping(val_f1):
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break
    
    # Save training history
    with open(save_dir / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    # Load best model for final evaluation
    print(f"\n{'='*80}")
    print("Final Test Evaluation")
    print(f"{'='*80}")
    
    checkpoint = torch.load(save_dir / "best_model.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test evaluation
    test_loss, test_f1, test_precision, test_recall, test_auc, test_cm = evaluate(
        model, test_loader, criterion, device
    )
    
    # Compute overfitting gap
    best_train_f1 = max(history['train_f1'])
    overfit_gap = best_train_f1 - test_f1
    
    # Print results
    print(f"\nTest Results:")
    print(f"  F1:        {test_f1:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall:    {test_recall:.4f}")
    print(f"  ROC-AUC:   {test_auc:.4f}")
    print(f"\nGeneralization:")
    print(f"  Best Val F1:   {best_val_f1:.4f} (epoch {checkpoint['epoch']})")
    print(f"  Best Train F1: {best_train_f1:.4f}")
    print(f"  Overfit Gap:   {overfit_gap:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"              Predicted")
    print(f"            Neg    Pos")
    print(f"Actual Neg [{test_cm[0,0]:4d}  {test_cm[0,1]:4d}]")
    print(f"       Pos [{test_cm[1,0]:4d}  {test_cm[1,1]:4d}]")
    
    # Calculate metrics
    tn, fp, fn, tp = test_cm[0,0], test_cm[0,1], test_cm[1,0], test_cm[1,1]
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    print(f"\nError Analysis:")
    print(f"  False Positive Rate: {fpr:.4f} ({fp}/{fp+tn})")
    print(f"  False Negative Rate: {fnr:.4f} ({fn}/{fn+tp})")
    
    # Save results
    results = {
        'experiment_name': 'final_production_model',
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
    
    with open(save_dir / "test_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Training Complete!")
    print(f"{'='*80}")
    print(f"Model saved to: {save_dir}")
    print(f"Expected F1 range: 0.865-0.872 (based on seed variance)")


if __name__ == "__main__":
    main()