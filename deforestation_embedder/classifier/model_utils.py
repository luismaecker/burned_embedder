import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix


class EarlyStopping:
    """Early stopping to stop training when validation metric stops improving"""
    
    def __init__(self, patience=15, min_delta=0.0, mode='max'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for metrics to maximize (F1), 'min' for loss
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self.mode == 'max':
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:  # mode == 'min'
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
        
        return self.early_stop


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for embeddings, labels in dataloader:
        embeddings = embeddings.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(embeddings)
        
        # Clamp outputs to avoid numerical issues with BCELoss
        outputs = torch.clamp(outputs, min=1e-7, max=1-1e-7)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Store predictions
        preds = (outputs > 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    
    # Calculate metrics
    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    
    return avg_loss, f1, precision, recall

def evaluate(model, dataloader, criterion, device):
    """Evaluate model on validation/test set"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for embeddings, labels in dataloader:
            embeddings = embeddings.to(device)
            labels = labels.to(device)
            
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            # Store predictions and probabilities
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    
    # Calculate metrics
    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    roc_auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)
    
    return avg_loss, f1, precision, recall, roc_auc, cm


def print_metrics(epoch, train_loss, train_f1, val_loss, val_f1, val_precision, val_recall, val_auc):
    """Print training metrics in a readable format"""
    print(f"Epoch {epoch:3d} | "
          f"Train Loss: {train_loss:.4f} F1: {train_f1:.4f} | "
          f"Val Loss: {val_loss:.4f} F1: {val_f1:.4f} P: {val_precision:.4f} R: {val_recall:.4f} AUC: {val_auc:.4f}")


def print_confusion_matrix(cm, title="Confusion Matrix"):
    """Print confusion matrix in readable format"""
    print(f"\n{title}:")
    print(f"                Predicted")
    print(f"              Neg    Pos")
    print(f"Actual Neg  [{cm[0,0]:4d}  {cm[0,1]:4d}]")
    print(f"       Pos  [{cm[1,0]:4d}  {cm[1,1]:4d}]")