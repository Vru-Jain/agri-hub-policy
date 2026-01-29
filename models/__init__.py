"""
Models Package for Agri-Policy Hub.

This package contains the machine learning models for crop yield prediction.

Subpackages:
    lstm/       - LSTM neural network architecture and predictor
    training/   - Data preparation and training pipeline
    checkpoints/ - Saved model weights (git-ignored)

Quick Usage:
    # Get predictions
    from models import get_trained_model
    predictor = get_trained_model()
    yield_kg = predictor.predict(state='Maharashtra', crop='Rice', season='Kharif', ndvi=0.5, soil_moisture=0.3)
    
    # Train a new model
    from models.training import CropYieldTrainer
    trainer = CropYieldTrainer(epochs=50)
    trainer.train()
"""

# Re-export main components for convenience
from .lstm import (
    CropYieldLSTM,
    CropYieldPredictor,
    get_trained_model,
    reset_predictor
)

__all__ = [
    # LSTM components
    'CropYieldLSTM',
    'CropYieldPredictor',
    'get_trained_model',
    'reset_predictor',
]
