"""
ML Models for Crop Yield Prediction.

This module provides PyTorch-based LSTM models trained on historical
crop yield data from Kaggle datasets.
"""

from .crop_yield_model import CropYieldLSTM, get_trained_model
