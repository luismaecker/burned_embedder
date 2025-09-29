import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from sklearn.model_selection import train_test_split
import rootutils

root_path = rootutils.find_root()


class DeforestationDataset(Dataset):
    """Dataset for deforestation classification using embedding differences"""
    
    def __init__(self, embedding_paths, labels, input_type='difference', normalize=True, augment=False):
        """
        Args:
            embedding_paths: List of tuples (before_path, after_path)
            labels: List of labels (0 or 1)
            input_type: 'difference', 'concat', or 'combined'
            normalize: Whether to L2 normalize the input
            augment: Whether to apply augmentation (training only)
        """
        self.embedding_paths = embedding_paths
        self.labels = labels
        self.input_type = input_type
        self.normalize = normalize
        self.augment = augment
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        before_path, after_path = self.embedding_paths[idx]
        
        # Load embeddings
        embedd_before = np.load(before_path)
        embedd_after = np.load(after_path)
        
        # Get input representation
        input_embedding = get_input_representation(
            embedd_before, embedd_after, 
            self.input_type, self.normalize
        )
        
        # Convert to tensor
        input_embedding = torch.from_numpy(input_embedding).float()
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        # Apply augmentation if enabled
        if self.augment:
            noise = torch.randn_like(input_embedding) * 0.002
            input_embedding = input_embedding + noise
        
        return input_embedding, label



def load_embedding_paths_and_labels(sample_type):
    """Load all embedding paths for a given sample type"""
    embeddings_dir = root_path / "data" / "processed" / "embeddings" / sample_type
    
    embedding_paths = []
    event_dirs = sorted([d for d in embeddings_dir.iterdir() if d.is_dir() and d.name.startswith('event_')])
    
    for event_dir in event_dirs:
        before_path = event_dir / "embedd_before.npy"
        after_path = event_dir / "embedd_after.npy"
        
        # Only include if both embeddings exist
        if before_path.exists() and after_path.exists():
            embedding_paths.append((before_path, after_path))
    
    # Label: 1 for positive, 0 for negative
    label = 1 if sample_type == 'positive' else 0
    labels = [label] * len(embedding_paths)
    
    return embedding_paths, labels


def create_datasets(test_size=0.2, val_split=0.1, random_state=42, input_type='difference', augment=False):
    """
    Create train/val/test datasets with stratified splits
    
    Args:
        test_size: Fraction for test set (default 0.2 = 20%)
        val_split: Fraction of remaining data for validation (default 0.1)
        random_state: Random seed for reproducibility
        input_type: 'difference', 'concat', or 'combined'
    
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    # Load positive and negative samples
    pos_paths, pos_labels = load_embedding_paths_and_labels('positive')
    neg_paths, neg_labels = load_embedding_paths_and_labels('negative')
    
    print(f"Loaded {len(pos_paths)} positive samples")
    print(f"Loaded {len(neg_paths)} negative samples")
    
    # Combine
    all_paths = pos_paths + neg_paths
    all_labels = pos_labels + neg_labels
    
    # First split: train+val vs test (80/20)
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        all_paths, all_labels, test_size=test_size, random_state=random_state, stratify=all_labels
    )
    
    # Second split: train vs val from the train+val set
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels, test_size=val_split, random_state=random_state, stratify=train_val_labels
    )
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(train_paths)} samples (pos: {sum(train_labels)}, neg: {len(train_labels) - sum(train_labels)})")
    print(f"  Val:   {len(val_paths)} samples (pos: {sum(val_labels)}, neg: {len(val_labels) - sum(val_labels)})")
    print(f"  Test:  {len(test_paths)} samples (pos: {sum(test_labels)}, neg: {len(test_labels) - sum(test_labels)})")
    print(f"  Input type: {input_type}")
    
     # Create datasets
    train_dataset = DeforestationDataset(train_paths, train_labels, input_type=input_type, normalize=True, augment=augment)
    val_dataset = DeforestationDataset(val_paths, val_labels, input_type=input_type, normalize=True, augment=False)
    test_dataset = DeforestationDataset(test_paths, test_labels, input_type=input_type, normalize=True, augment=False)
    
    return train_dataset, val_dataset, test_dataset


def get_input_representation(embedd_before, embedd_after, input_type='difference', normalize=True):
    """
    Create different input representations from before/after embeddings
    
    Args:
        embedd_before: Before embedding
        embedd_after: After embedding
        input_type: 'difference', 'concat', or 'combined'
        normalize: Whether to L2 normalize
    
    Returns:
        Input tensor for the model
    """
    if input_type == 'difference':
        # Just the difference
        result = embedd_after - embedd_before
        
    elif input_type == 'concat':
        # Concatenate both embeddings
        result = np.concatenate([embedd_before, embedd_after])
        
    elif input_type == 'combined':
        # Concatenate before, after, and difference
        diff = embedd_after - embedd_before
        result = np.concatenate([embedd_before, embedd_after, diff])
        
    else:
        raise ValueError(f"Unknown input_type: {input_type}")
    
    # L2 normalize if requested
    if normalize:
        norm = np.linalg.norm(result)
        if norm > 1e-8:
            result = result / norm
        else:
            result = np.zeros_like(result)
    
    # Check for NaN or Inf
    if not np.isfinite(result).all():
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    
    return result