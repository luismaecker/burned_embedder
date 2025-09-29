# scripts/train_optimal_model.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import json
from pathlib import Path
import rootutils


root_path = rootutils.find_root()

from burned_embedder.utils import setup_device
from burned_embedder.classifier.dataset import create_datasets
from burned_embedder.classifier.model import OptimalMLP
from burned_embedder.classifier.model_utils import train_epoch, evaluate, EarlyStopping, print_metrics

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def main():
    print("="*80)
    print("Training OPTIMAL Deforestation Classifier")
    print("="*80)
    
    # Configuration
    config = {
        'test_size': 0.2,
        'val_split': 0.1,
        'seed': 42,
        'input_type': 'concat',
        'hidden_dims': [128],
        'dropout': 0.6,
        'weight_decay': 1e-3,
        'batch_size': 64,
        'lr': 0.001,
        'epochs': 100,
        'patience': 10,
        'use_batch_norm': True,
        'gradient_clip': 1.0,
    }
    
    print(f"\nOptimal Configuration:")
    print(json.dumps(config, indent=2))
    
    set_seed(config['seed'])
    device = setup_device(gpu_index=1, memory_fraction=1.0)
    print(f"\nDevice: {device}")
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset, val_dataset, test_dataset = create_datasets(
        test_size=config['test_size'],
        val_split=config['val_split'],
        random_state=config['seed'],
        input_type=config['input_type'],
        augment=True  # Only augment training data
    )
    
    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)
    
    # Get input dimension
    sample_embedding, _ = train_dataset[0]
    input_dim = sample_embedding.shape[0]
    print(f"Input dimension: {input_dim}")
    
    # Create model
    model = OptimalMLP(
        input_dim=input_dim,
        hidden_dims=config['hidden_dims'],
        dropout=config['dropout'],
        use_batch_norm=config['use_batch_norm']
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Optimizer and loss
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=config['patience'], mode='max')
    
    # Output directory
    save_dir = root_path / "models" / "optimal_classifier"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    print("\n" + "="*80)
    print("Training")
    print("="*80)
    
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
        
        # Print
        print_metrics(epoch, train_loss, train_f1, val_loss, val_f1, val_precision, val_recall, val_auc)
        
        # LR scheduling
        scheduler.step(val_f1)
        
        # Save best
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
        
        # Early stopping
        if early_stopping(val_f1):
            print(f"\nEarly stopping at epoch {epoch}")
            break
    
    # Save history
    with open(save_dir / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    # Final evaluation
    print("\n" + "="*80)
    print("Final Test Evaluation")
    print("="*80)
    
    checkpoint = torch.load(save_dir / "best_model.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_f1, test_precision, test_recall, test_auc, test_cm = evaluate(
        model, test_loader, criterion, device
    )
    
    print(f"\nTest Results:")
    print(f"  F1:        {test_f1:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall:    {test_recall:.4f}")
    print(f"  ROC-AUC:   {test_auc:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"              Predicted")
    print(f"            Neg    Pos")
    print(f"Actual Neg [{test_cm[0,0]:4d}  {test_cm[0,1]:4d}]")
    print(f"       Pos [{test_cm[1,0]:4d}  {test_cm[1,1]:4d}]")
    
    # Calculate improvement over baseline
    baseline_f1 = 0.7713  # From your original training
    improvement = ((test_f1 - baseline_f1) / baseline_f1) * 100
    print(f"\nImprovement over baseline: {improvement:+.1f}%")
    
    # Save results
    results = {
        'test_f1': float(test_f1),
        'test_precision': float(test_precision),
        'test_recall': float(test_recall),
        'test_auc': float(test_auc),
        'confusion_matrix': test_cm.tolist(),
        'config': config,
        'best_epoch': checkpoint['epoch']
    }
    
    with open(save_dir / "test_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Model saved to {save_dir}")

if __name__ == "__main__":
    main()