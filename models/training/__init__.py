"""
Training Module for Crop Yield Prediction.

This package contains the training pipeline components:
- data_prep.py: Data loading and preprocessing
- trainer.py: Model training and checkpointing

Usage:
    # Run training from command line
    python -m models.training.trainer
    
    # Or import and use programmatically
    from models.training import CropYieldTrainer
    trainer = CropYieldTrainer(epochs=50)
    metrics = trainer.train()
"""

from .trainer import CropYieldTrainer, main as train_main
from .data_prep import (
    load_kaggle_data,
    preprocess_data,
    create_tensors,
    create_dataloaders
)

__all__ = [
    'CropYieldTrainer',
    'train_main',
    'load_kaggle_data',
    'preprocess_data',
    'create_tensors',
    'create_dataloaders'
]
