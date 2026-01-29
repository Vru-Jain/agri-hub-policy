"""
LSTM Architecture for Crop Yield Prediction.

This module defines the neural network architecture used for
predicting crop yields based on agricultural and environmental features.
"""

import torch
import torch.nn as nn
from typing import Tuple


class CropYieldLSTM(nn.Module):
    """
    LSTM-based neural network for crop yield prediction.
    
    Architecture:
        - Embedding layers for categorical features (state, crop, season)
        - LSTM layers for learning temporal patterns
        - Dense layers for final yield prediction
    
    Input Features:
        - Categorical: state_idx, crop_idx, season_idx
        - Continuous: NDVI, soil_moisture, rainfall, area, year
    
    Output:
        - Predicted yield value (raw, requires post-processing)
    """
    
    # Default architecture parameters
    DEFAULT_NUM_STATES = 30
    DEFAULT_NUM_CROPS = 20
    DEFAULT_NUM_SEASONS = 2
    DEFAULT_EMBEDDING_DIM = 16
    DEFAULT_HIDDEN_SIZE = 64
    DEFAULT_NUM_LAYERS = 2
    DEFAULT_DROPOUT = 0.2
    
    def __init__(
        self,
        num_states: int = DEFAULT_NUM_STATES,
        num_crops: int = DEFAULT_NUM_CROPS,
        num_seasons: int = DEFAULT_NUM_SEASONS,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
        num_layers: int = DEFAULT_NUM_LAYERS,
        dropout: float = DEFAULT_DROPOUT
    ):
        """
        Initialize the LSTM model.
        
        Args:
            num_states: Number of unique states in the dataset
            num_crops: Number of unique crops in the dataset
            num_seasons: Number of seasons (typically 2: Kharif, Rabi)
            embedding_dim: Dimension of embedding vectors
            hidden_size: LSTM hidden layer size
            num_layers: Number of LSTM layers
            dropout: Dropout probability for regularization
        """
        super(CropYieldLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Embedding layers for categorical features
        self.state_embedding = nn.Embedding(num_states, embedding_dim)
        self.crop_embedding = nn.Embedding(num_crops, embedding_dim)
        self.season_embedding = nn.Embedding(num_seasons, embedding_dim)
        
        # Number of continuous features: NDVI, soil_moisture, rainfall, area, year
        self.num_continuous = 5
        
        # Total input size = embeddings + continuous features
        input_size = (3 * embedding_dim) + self.num_continuous
        
        # LSTM layers for sequence modeling
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected output layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
    
    def forward(
        self,
        state_idx: torch.Tensor,
        crop_idx: torch.Tensor,
        season_idx: torch.Tensor,
        continuous_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            state_idx: State indices tensor (batch_size,)
            crop_idx: Crop indices tensor (batch_size,)
            season_idx: Season indices tensor (batch_size,)
            continuous_features: Continuous features tensor (batch_size, 5)
            
        Returns:
            Predicted yield tensor (batch_size, 1)
        """
        # Get embeddings for categorical features
        state_emb = self.state_embedding(state_idx)
        crop_emb = self.crop_embedding(crop_idx)
        season_emb = self.season_embedding(season_idx)
        
        # Concatenate all features
        x = torch.cat([state_emb, crop_emb, season_emb, continuous_features], dim=-1)
        
        # Add sequence dimension for LSTM: (batch, seq_len=1, features)
        x = x.unsqueeze(1)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Take the last output
        out = lstm_out[:, -1, :]
        
        # Final prediction through dense layers
        yield_pred = self.fc(out)
        
        return yield_pred
    
    def get_config(self) -> dict:
        """Return model configuration for saving/loading."""
        return {
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'num_continuous': self.num_continuous
        }
