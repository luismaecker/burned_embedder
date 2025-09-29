# # configs/experiment_configs.py

# """
# Extended systematic experiment suite based on preliminary findings
# """

# # Base configuration
BASE_CONFIG = {
    'test_size': 0.2,
    'val_split': 0.1,
    'seed': 42,
    'batch_size': 64,
    'lr': 0.001,
    'epochs': 100,
    'patience': 15,
}

# New experiments to add:

EXPERIMENTS = {

        'best_model_seed1': {
        **BASE_CONFIG,
        'input_type': 'concat',
        'hidden_dims': [512, 256, 128],
        'dropout': 0.5,
        'lr': 0.0005,
        'weight_decay': 0.0001,
        'batch_size': 64,
        'seed': 1,
    },

    'best_model_seed2': {
        **BASE_CONFIG,
        'input_type': 'concat',
        'hidden_dims': [512, 256, 128],
        'dropout': 0.5,
        'lr': 0.0005,
        'weight_decay': 0.0001,
        'batch_size': 64,
        'seed': 42,
    },

    'best_model_seed3': {
        **BASE_CONFIG,
        'input_type': 'concat',
        'hidden_dims': [512, 256, 128],
        'dropout': 0.5,
        'lr': 0.0005,
        'weight_decay': 0.0001,
        'batch_size': 64,
        'seed': 123,
    },

    'best_model_seed4': {
        **BASE_CONFIG,
        'input_type': 'concat',
        'hidden_dims': [512, 256, 128],
        'dropout': 0.5,
        'lr': 0.0005,
        'weight_decay': 0.0001,
        'batch_size': 64,
        'seed': 999,
    },

    # Second best: concat_minimal_compression config
    'second_best_seed1': {
        **BASE_CONFIG,
        'input_type': 'concat',
        'hidden_dims': [1400, 700, 350],
        'dropout': 0.4,
        'lr': 0.001,
        'weight_decay': 0.0001,
        'batch_size': 64,
        'seed': 1,
    },

    'second_best_seed2': {
        **BASE_CONFIG,
        'input_type': 'concat',
        'hidden_dims': [1400, 700, 350],
        'dropout': 0.4,
        'lr': 0.001,
        'weight_decay': 0.0001,
        'batch_size': 64,
        'seed': 42,
    },

    'second_best_seed3': {
        **BASE_CONFIG,
        'input_type': 'concat',
        'hidden_dims': [1400, 700, 350],
        'dropout': 0.4,
        'lr': 0.001,
        'weight_decay': 0.0001,
        'batch_size': 64,
        'seed': 123,
    },

    'second_best_seed4': {
        **BASE_CONFIG,
        'input_type': 'concat',
        'hidden_dims': [1400, 700, 350],
        'dropout': 0.4,
        'lr': 0.001,
        'weight_decay': 0.0001,
        'batch_size': 64,
        'seed': 999,
    },
}

# EXPERIMENTS = {
#     # ========================================
#     # SECTION 1: BASELINE REPLICATION
#     # ========================================
#     'baseline_v2': {
#         **BASE_CONFIG,
#         'input_type': 'difference',
#         'hidden_dims': [256, 128],
#         'dropout': 0.4,
#         'weight_decay': 0.0,
#     },
    
#     # ========================================
#     # SECTION 2: DEEP DIVE ON CONCAT INPUT
#     # (Since concat won in prelim results)
#     # ========================================
    
#     # Architecture variations for concat
#     'concat_tiny': {
#         **BASE_CONFIG,
#         'input_type': 'concat',
#         'hidden_dims': [256],
#         'dropout': 0.4,
#         'weight_decay': 0.0,
#     },
    
#     'concat_small': {
#         **BASE_CONFIG,
#         'input_type': 'concat',
#         'hidden_dims': [384, 192],
#         'dropout': 0.4,
#         'weight_decay': 0.0,
#     },
    
#     'concat_baseline': {  # Replicate best performer
#         **BASE_CONFIG,
#         'input_type': 'concat',
#         'hidden_dims': [512, 256, 128],
#         'dropout': 0.4,
#         'weight_decay': 0.0,
#     },
    
#     'concat_large': {
#         **BASE_CONFIG,
#         'input_type': 'concat',
#         'hidden_dims': [768, 384, 192],
#         'dropout': 0.4,
#         'weight_decay': 0.0,
#     },
    
#     'concat_extra_large': {
#         **BASE_CONFIG,
#         'input_type': 'concat',
#         'hidden_dims': [1024, 512, 256],
#         'dropout': 0.4,
#         'weight_decay': 0.0,
#     },
    
#     'concat_very_deep': {
#         **BASE_CONFIG,
#         'input_type': 'concat',
#         'hidden_dims': [512, 384, 256, 128, 64],
#         'dropout': 0.4,
#         'weight_decay': 0.0,
#     },
    
#     # Regularization sweep for concat (to fight overfitting)
#     'concat_dropout_03': {
#         **BASE_CONFIG,
#         'input_type': 'concat',
#         'hidden_dims': [512, 256, 128],
#         'dropout': 0.3,
#         'weight_decay': 0.0,
#     },
    
#     'concat_dropout_05': {
#         **BASE_CONFIG,
#         'input_type': 'concat',
#         'hidden_dims': [512, 256, 128],
#         'dropout': 0.5,
#         'weight_decay': 0.0,
#     },
    
#     'concat_dropout_06': {
#         **BASE_CONFIG,
#         'input_type': 'concat',
#         'hidden_dims': [512, 256, 128],
#         'dropout': 0.6,
#         'weight_decay': 0.0,
#     },
    
#     'concat_dropout_07': {
#         **BASE_CONFIG,
#         'input_type': 'concat',
#         'hidden_dims': [512, 256, 128],
#         'dropout': 0.7,
#         'weight_decay': 0.0,
#     },
    
#     # Weight decay sweep for concat
#     'concat_wd_1e5': {
#         **BASE_CONFIG,
#         'input_type': 'concat',
#         'hidden_dims': [512, 256, 128],
#         'dropout': 0.4,
#         'weight_decay': 1e-5,
#     },
    
#     'concat_wd_5e5': {
#         **BASE_CONFIG,
#         'input_type': 'concat',
#         'hidden_dims': [512, 256, 128],
#         'dropout': 0.4,
#         'weight_decay': 5e-5,
#     },
    
#     'concat_wd_1e4': {
#         **BASE_CONFIG,
#         'input_type': 'concat',
#         'hidden_dims': [512, 256, 128],
#         'dropout': 0.4,
#         'weight_decay': 1e-4,
#     },
    
#     'concat_wd_5e4': {
#         **BASE_CONFIG,
#         'input_type': 'concat',
#         'hidden_dims': [512, 256, 128],
#         'dropout': 0.4,
#         'weight_decay': 5e-4,
#     },
    
#     'concat_wd_1e3': {
#         **BASE_CONFIG,
#         'input_type': 'concat',
#         'hidden_dims': [512, 256, 128],
#         'dropout': 0.4,
#         'weight_decay': 1e-3,
#     },
    
#     # Combined regularization strategies
#     'concat_heavy_reg_v1': {
#         **BASE_CONFIG,
#         'input_type': 'concat',
#         'hidden_dims': [512, 256, 128],
#         'dropout': 0.6,
#         'weight_decay': 1e-4,
#     },
    
#     'concat_heavy_reg_v2': {
#         **BASE_CONFIG,
#         'input_type': 'concat',
#         'hidden_dims': [512, 256, 128],
#         'dropout': 0.7,
#         'weight_decay': 5e-4,
#     },
    
#     'concat_heavy_reg_v3': {
#         **BASE_CONFIG,
#         'input_type': 'concat',
#         'hidden_dims': [384, 192],  # Smaller + strong reg
#         'dropout': 0.6,
#         'weight_decay': 1e-4,
#     },
    
#     # Learning rate variations for concat
#     'concat_lr_5e5': {
#         **BASE_CONFIG,
#         'input_type': 'concat',
#         'hidden_dims': [512, 256, 128],
#         'dropout': 0.4,
#         'weight_decay': 0.0,
#         'lr': 5e-5,
#     },
    
#     'concat_lr_1e4': {
#         **BASE_CONFIG,
#         'input_type': 'concat',
#         'hidden_dims': [512, 256, 128],
#         'dropout': 0.4,
#         'weight_decay': 0.0,
#         'lr': 1e-4,
#     },
    
#     'concat_lr_5e4': {
#         **BASE_CONFIG,
#         'input_type': 'concat',
#         'hidden_dims': [512, 256, 128],
#         'dropout': 0.4,
#         'weight_decay': 0.0,
#         'lr': 5e-4,
#     },
    
#     'concat_lr_5e3': {
#         **BASE_CONFIG,
#         'input_type': 'concat',
#         'hidden_dims': [512, 256, 128],
#         'dropout': 0.4,
#         'weight_decay': 0.0,
#         'lr': 5e-3,
#     },
    
#     # ========================================
#     # SECTION 3: COMBINED INPUT EXPLORATION
#     # (Since it came 2nd in prelims)
#     # ========================================
    
#     'combined_small': {
#         **BASE_CONFIG,
#         'input_type': 'combined',
#         'hidden_dims': [512, 256],
#         'dropout': 0.5,
#         'weight_decay': 0.0,
#     },
    
#     'combined_baseline': {
#         **BASE_CONFIG,
#         'input_type': 'combined',
#         'hidden_dims': [768, 384, 192],
#         'dropout': 0.4,
#         'weight_decay': 0.0,
#     },
    
#     'combined_large': {
#         **BASE_CONFIG,
#         'input_type': 'combined',
#         'hidden_dims': [1024, 512, 256, 128],
#         'dropout': 0.4,
#         'weight_decay': 0.0,
#     },
    
#     'combined_regularized': {
#         **BASE_CONFIG,
#         'input_type': 'combined',
#         'hidden_dims': [768, 384, 192],
#         'dropout': 0.6,
#         'weight_decay': 1e-4,
#     },
    
#     'combined_heavy_reg': {
#         **BASE_CONFIG,
#         'input_type': 'combined',
#         'hidden_dims': [768, 384, 192],
#         'dropout': 0.7,
#         'weight_decay': 5e-4,
#     },
    
#     # ========================================
#     # SECTION 4: DIFFERENCE INPUT WITH BETTER REG
#     # (Try to salvage difference input)
#     # ========================================
    
#     'diff_optimized_v1': {
#         **BASE_CONFIG,
#         'input_type': 'difference',
#         'hidden_dims': [384, 192, 96],
#         'dropout': 0.5,
#         'weight_decay': 1e-4,
#         'lr': 5e-4,
#     },
    
#     'diff_optimized_v2': {
#         **BASE_CONFIG,
#         'input_type': 'difference',
#         'hidden_dims': [512, 256],
#         'dropout': 0.6,
#         'weight_decay': 1e-4,
#         'lr': 1e-4,
#     },
    
#     'diff_minimal_overfit': {
#         **BASE_CONFIG,
#         'input_type': 'difference',
#         'hidden_dims': [256, 128],
#         'dropout': 0.7,
#         'weight_decay': 5e-4,
#         'lr': 1e-4,
#     },
    
#     # ========================================
#     # SECTION 5: BATCH SIZE EFFECTS
#     # ========================================
    
#     'concat_batch_16': {
#         **BASE_CONFIG,
#         'input_type': 'concat',
#         'hidden_dims': [512, 256, 128],
#         'dropout': 0.4,
#         'weight_decay': 0.0,
#         'batch_size': 16,
#     },
    
#     'concat_batch_32': {
#         **BASE_CONFIG,
#         'input_type': 'concat',
#         'hidden_dims': [512, 256, 128],
#         'dropout': 0.4,
#         'weight_decay': 0.0,
#         'batch_size': 32,
#     },
    
#     'concat_batch_128': {
#         **BASE_CONFIG,
#         'input_type': 'concat',
#         'hidden_dims': [512, 256, 128],
#         'dropout': 0.4,
#         'weight_decay': 0.0,
#         'batch_size': 128,
#     },
    
#     'concat_batch_256': {
#         **BASE_CONFIG,
#         'input_type': 'concat',
#         'hidden_dims': [512, 256, 128],
#         'dropout': 0.4,
#         'weight_decay': 0.0,
#         'batch_size': 256,
#     },
    
#     # ========================================
#     # SECTION 6: EARLY STOPPING PATIENCE
#     # (Maybe stopping too early or too late?)
#     # ========================================
    
#     'concat_patience_5': {
#         **BASE_CONFIG,
#         'input_type': 'concat',
#         'hidden_dims': [512, 256, 128],
#         'dropout': 0.4,
#         'weight_decay': 0.0,
#         'patience': 5,
#     },
    
#     'concat_patience_10': {
#         **BASE_CONFIG,
#         'input_type': 'concat',
#         'hidden_dims': [512, 256, 128],
#         'dropout': 0.4,
#         'weight_decay': 0.0,
#         'patience': 10,
#     },
    
#     'concat_patience_25': {
#         **BASE_CONFIG,
#         'input_type': 'concat',
#         'hidden_dims': [512, 256, 128],
#         'dropout': 0.4,
#         'weight_decay': 0.0,
#         'patience': 25,
#     },
    
#     'concat_patience_30': {
#         **BASE_CONFIG,
#         'input_type': 'concat',
#         'hidden_dims': [512, 256, 128],
#         'dropout': 0.4,
#         'weight_decay': 0.0,
#         'patience': 30,
#     },
    
#     # ========================================
#     # SECTION 7: OPTIMIZER VARIATIONS
#     # (Add these if you modify run_experiments.py)
#     # ========================================
    
#     # Placeholder for different optimizers
#     # Would need code changes to support:
#     # - SGD with momentum
#     # - AdamW
#     # - Different beta values
    
#     # ========================================
#     # SECTION 8: TARGETED BEST PERFORMERS
#     # ========================================
    
#     'concat_ultra_optimized_v1': {
#         **BASE_CONFIG,
#         'input_type': 'concat',
#         'hidden_dims': [512, 256, 128],
#         'dropout': 0.55,
#         'weight_decay': 1e-4,
#         'lr': 5e-4,
#         'batch_size': 64,
#         'patience': 20,
#     },
    
#     'concat_ultra_optimized_v2': {
#         **BASE_CONFIG,
#         'input_type': 'concat',
#         'hidden_dims': [512, 384, 256],
#         'dropout': 0.5,
#         'weight_decay': 5e-5,
#         'lr': 1e-4,
#         'batch_size': 32,
#         'patience': 20,
#     },
    
#     'concat_ultra_optimized_v3': {
#         **BASE_CONFIG,
#         'input_type': 'concat',
#         'hidden_dims': [768, 512, 256],
#         'dropout': 0.6,
#         'weight_decay': 1e-4,
#         'lr': 2e-4,
#         'batch_size': 64,
#         'patience': 20,
#     },
    
#     # High recall configuration (for catching all deforestation)
#     'concat_high_recall': {
#         **BASE_CONFIG,
#         'input_type': 'concat',
#         'hidden_dims': [512, 256, 128],
#         'dropout': 0.3,  # Lower dropout = more sensitive
#         'weight_decay': 0.0,
#         'lr': 1e-3,
#         'batch_size': 32,  # Smaller batches = noisier gradients
#     },
    
#     # High precision configuration (for reducing false positives)
#     'concat_high_precision': {
#         **BASE_CONFIG,
#         'input_type': 'concat',
#         'hidden_dims': [768, 384, 192],
#         'dropout': 0.7,  # Heavy regularization
#         'weight_decay': 5e-4,
#         'lr': 5e-5,  # Very conservative
#         'batch_size': 128,
#     },
    
#     # Balanced configuration
#     'concat_balanced_f1': {
#         **BASE_CONFIG,
#         'input_type': 'concat',
#         'hidden_dims': [640, 320, 160],
#         'dropout': 0.55,
#         'weight_decay': 2e-4,
#         'lr': 3e-4,
#         'batch_size': 64,
#         'patience': 20,
#     },
    
#     # ========================================
#     # SECTION 9: ENSEMBLE DIVERSITY
#     # (Train multiple models with different seeds)
#     # ========================================
    
#     'concat_best_seed1': {
#         **BASE_CONFIG,
#         'input_type': 'concat',
#         'hidden_dims': [512, 256, 128],
#         'dropout': 0.5,
#         'weight_decay': 1e-4,
#         'lr': 5e-4,
#         'seed': 1,
#     },
    
#     'concat_best_seed2': {
#         **BASE_CONFIG,
#         'input_type': 'concat',
#         'hidden_dims': [512, 256, 128],
#         'dropout': 0.5,
#         'weight_decay': 1e-4,
#         'lr': 5e-4,
#         'seed': 2,
#     },
    
#     'concat_best_seed3': {
#         **BASE_CONFIG,
#         'input_type': 'concat',
#         'hidden_dims': [512, 256, 128],
#         'dropout': 0.5,
#         'weight_decay': 1e-4,
#         'lr': 5e-4,
#         'seed': 123,
#     },
    
#     'concat_best_seed4': {
#         **BASE_CONFIG,
#         'input_type': 'concat',
#         'hidden_dims': [512, 256, 128],
#         'dropout': 0.5,
#         'weight_decay': 1e-4,
#         'lr': 5e-4,
#         'seed': 999,
#     },
    
#     # ========================================
#     # SECTION 10: ABLATION ON TOP PERFORMER
#     # ========================================
    
#     # Starting from exp_concat (F1: 0.8125)
#     # Remove one thing at a time to see what matters
    
#     'concat_ablate_smaller': {
#         **BASE_CONFIG,
#         'input_type': 'concat',
#         'hidden_dims': [256, 128],  # Much smaller
#         'dropout': 0.4,
#         'weight_decay': 0.0,
#     },
    
#     'concat_ablate_no_dropout': {
#         **BASE_CONFIG,
#         'input_type': 'concat',
#         'hidden_dims': [512, 256, 128],
#         'dropout': 0.0,  # No regularization
#         'weight_decay': 0.0,
#     },
    
#     'concat_ablate_wider': {
#         **BASE_CONFIG,
#         'input_type': 'concat',
#         'hidden_dims': [1024, 1024],  # Very wide
#         'dropout': 0.4,
#         'weight_decay': 0.0,
#     },
#      'concat_wide_v1': {
#         **BASE_CONFIG,
#         'input_type': 'concat',
#         'hidden_dims': [1024, 512, 256],  # Less aggressive compression
#         'dropout': 0.4,
#         'weight_decay': 1e-4,
#         'lr': 0.001,
#     },
    
#     'concat_wide_v2': {
#         **BASE_CONFIG,
#         'input_type': 'concat',
#         'hidden_dims': [1280, 640, 320],  # Gradual 2x reductions
#         'dropout': 0.4,
#         'weight_decay': 1e-4,
#         'lr': 0.001,
#     },
    
#     'concat_very_wide': {
#         **BASE_CONFIG,
#         'input_type': 'concat',
#         'hidden_dims': [1536, 768, 384],  # No initial compression
#         'dropout': 0.4,
#         'weight_decay': 1e-4,
#         'lr': 0.001,
#     },
    
#     'concat_minimal_compression': {
#         **BASE_CONFIG,
#         'input_type': 'concat',
#         'hidden_dims': [1400, 700, 350],  # Very gradual
#         'dropout': 0.4,
#         'weight_decay': 1e-4,
#          'lr': 0.001,
#     }
# }
