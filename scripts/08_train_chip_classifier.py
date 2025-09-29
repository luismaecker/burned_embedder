import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import argparse
from pathlib import Path
import rootutils
import json

root_path = rootutils.find_root()

from burned_embedder.classifier.dataset import DeforestationDataset, create_datasets
from burned_embedder.classifier.model import DifferenceMLP, get_model
from burned_embedder.classifier.model_utils import (
    train_epoch, evaluate, EarlyStopping, print_metrics, print_confusion_matrix
)

from burned_embedder.utils import setup_device


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def save_training_history(history, save_dir):
    """Save training history to JSON"""
    history_file = save_dir / "training_history.json"
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history to {history_file}")


def main(args):
    print("="*70)
    print("Deforestation Chip Classifier Training")
    print("="*70)
    
    # Set random seed
    set_seed(args.seed)
    
    # Setup device
    device = setup_device(gpu_index=1, memory_fraction=1.0)
    print(f"\nUsing device: {device}")
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset, val_dataset, test_dataset = create_datasets(
        test_size=args.test_size,
        val_split=args.val_split,
        random_state=args.seed
    )
    
    # Get input dimension from first sample
    sample_embedding, _ = train_dataset[0]
    input_dim = sample_embedding.shape[0]
    print(f"\nEmbedding dimension: {input_dim}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Create model
    model = get_model(
        input_dim=input_dim,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        device=device
    )
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience, mode='max')
    
    # Create output directory
    save_dir = root_path / "models" / "chip_classifier"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    print("\n" + "="*70)
    print("Starting Training")
    print("="*70)
    
    best_val_f1 = 0
    history = {
        'train_loss': [], 'train_f1': [],
        'val_loss': [], 'val_f1': [], 'val_precision': [], 'val_recall': [], 'val_auc': []
    }
    
    for epoch in range(1, args.epochs + 1):
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
                'config': vars(args)
            }, save_dir / "best_model.pt")
            print(f"  → Saved new best model (F1: {val_f1:.4f})")
        
        # Early stopping check
        if early_stopping(val_f1):
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break
    
    # Save training history
    save_training_history(history, save_dir)
    
    # Load best model for final evaluation
    print("\n" + "="*70)
    print("Final Evaluation on Test Set")
    print("="*70)
    
    checkpoint = torch.load(save_dir / "best_model.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']} (Val F1: {checkpoint['val_f1']:.4f})")
    
    # Test evaluation
    test_loss, test_f1, test_precision, test_recall, test_auc, test_cm = evaluate(
        model, test_loader, criterion, device
    )
    
    print(f"\nTest Results:")
    print(f"  Loss:      {test_loss:.4f}")
    print(f"  F1 Score:  {test_f1:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall:    {test_recall:.4f}")
    print(f"  ROC-AUC:   {test_auc:.4f}")
    
    print_confusion_matrix(test_cm, title="Test Set Confusion Matrix")
    
    # Save test results
    test_results = {
        'test_loss': float(test_loss),
        'test_f1': float(test_f1),
        'test_precision': float(test_precision),
        'test_recall': float(test_recall),
        'test_roc_auc': float(test_auc),
        'confusion_matrix': test_cm.tolist()
    }
    
    with open(save_dir / "test_results.json", 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\n✓ Training complete! Model saved to {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train chip-level deforestation classifier')
    
    # Data parameters
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set fraction')
    parser.add_argument('--val_split', type=float, default=0.1, help='Validation split from train')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Model parameters
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[256, 128], help='Hidden layer dimensions')
    parser.add_argument('--dropout', type=float, default=0.4, help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Maximum epochs')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    
    args = parser.parse_args()
    main(args)