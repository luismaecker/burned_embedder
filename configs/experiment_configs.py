# configs/experiment_configs.py

"""
Configuration file for systematic model experiments
"""

# Base configuration
BASE_CONFIG = {
    'test_size': 0.2,
    'val_split': 0.1,
    'seed': 42,
    'batch_size': 64,
    'lr': 0.001,
    'epochs': 100,
    'patience': 15,
}

# Experiment configurations
EXPERIMENTS = {
    # Baseline
    'baseline': {
        **BASE_CONFIG,
        'input_type': 'difference',
        'hidden_dims': [256, 128],
        'dropout': 0.4,
        'weight_decay': 0.0,
    },
    
    # Input representation experiments
    'exp_concat': {
        **BASE_CONFIG,
        'input_type': 'concat',
        'hidden_dims': [512, 256, 128],  # Larger for 2x input
        'dropout': 0.4,
        'weight_decay': 0.0,
    },
    
    'exp_combined': {
        **BASE_CONFIG,
        'input_type': 'combined',
        'hidden_dims': [768, 384, 192],  # Larger for 3x input
        'dropout': 0.4,
        'weight_decay': 0.0,
    },
    
    # Architecture size experiments (with difference input)
    'exp_shallow': {
        **BASE_CONFIG,
        'input_type': 'difference',
        'hidden_dims': [128],
        'dropout': 0.4,
        'weight_decay': 0.0,
    },
    
    'exp_medium': {
        **BASE_CONFIG,
        'input_type': 'difference',
        'hidden_dims': [512, 256, 128],
        'dropout': 0.4,
        'weight_decay': 0.0,
    },
    
    'exp_deep': {
        **BASE_CONFIG,
        'input_type': 'difference',
        'hidden_dims': [512, 256, 128, 64],
        'dropout': 0.4,
        'weight_decay': 0.0,
    },
    
    # Regularization experiments
    'exp_low_dropout': {
        **BASE_CONFIG,
        'input_type': 'difference',
        'hidden_dims': [256, 128],
        'dropout': 0.2,
        'weight_decay': 0.0,
    },
    
    'exp_high_dropout': {
        **BASE_CONFIG,
        'input_type': 'difference',
        'hidden_dims': [256, 128],
        'dropout': 0.6,
        'weight_decay': 0.0,
    },
    
    'exp_weight_decay': {
        **BASE_CONFIG,
        'input_type': 'difference',
        'hidden_dims': [256, 128],
        'dropout': 0.4,
        'weight_decay': 1e-4,
    },
    
    'exp_weight_decay_strong': {
        **BASE_CONFIG,
        'input_type': 'difference',
        'hidden_dims': [256, 128],
        'dropout': 0.5,
        'weight_decay': 1e-3,
    },
    
    # Learning rate experiments
    'exp_lr_low': {
        **BASE_CONFIG,
        'input_type': 'difference',
        'hidden_dims': [256, 128],
        'dropout': 0.4,
        'weight_decay': 0.0,
        'lr': 0.0001,
    },
    
    'exp_lr_high': {
        **BASE_CONFIG,
        'input_type': 'difference',
        'hidden_dims': [256, 128],
        'dropout': 0.4,
        'weight_decay': 0.0,
        'lr': 0.01,
    },
    
    # Batch size experiments
    'exp_batch_small': {
        **BASE_CONFIG,
        'input_type': 'difference',
        'hidden_dims': [256, 128],
        'dropout': 0.4,
        'weight_decay': 0.0,
        'batch_size': 32,
    },
    
    'exp_batch_large': {
        **BASE_CONFIG,
        'input_type': 'difference',
        'hidden_dims': [256, 128],
        'dropout': 0.4,
        'weight_decay': 0.0,
        'batch_size': 128,
    },
    
    # Combined best practices
    'exp_regularized': {
        **BASE_CONFIG,
        'input_type': 'difference',
        'hidden_dims': [512, 256, 128],
        'dropout': 0.5,
        'weight_decay': 1e-4,
        'lr': 0.0001,
    },
    
    'exp_concat_regularized': {
        **BASE_CONFIG,
        'input_type': 'concat',
        'hidden_dims': [512, 256, 128],
        'dropout': 0.5,
        'weight_decay': 1e-4,
        'lr': 0.0001,
    },
}