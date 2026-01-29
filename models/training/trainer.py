"""
LSTM Model Trainer for Crop Yield Prediction.

This module handles the training loop, optimization, and model checkpointing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Tuple, Optional
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.lstm import CropYieldLSTM
from .data_prep import (
    load_kaggle_data,
    preprocess_data,
    create_tensors,
    create_dataloaders
)


class CropYieldTrainer:
    """
    Trainer class for the LSTM crop yield prediction model.
    
    Handles the full training pipeline:
    - Data loading and preprocessing
    - Model initialization
    - Training loop with validation
    - Checkpoint saving
    """
    
    def __init__(
        self,
        hidden_size: int = 64,
        num_layers: int = 2,
        embedding_dim: int = 16,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
        device: Optional[str] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            hidden_size: LSTM hidden layer size
            num_layers: Number of LSTM layers
            embedding_dim: Dimension of embedding vectors
            dropout: Dropout probability
            learning_rate: Optimizer learning rate
            batch_size: Training batch size
            epochs: Number of training epochs
            device: Device to train on ('cuda' or 'cpu')
        """
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        
        self.device = torch.device(
            device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        )
        
        self.model: Optional[CropYieldLSTM] = None
        self.encoders: Dict[str, Dict] = {}
        self.scalers: Dict[str, Tuple[float, float]] = {}
        
        # Checkpoint directory
        self.checkpoint_dir = Path(__file__).parent.parent / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)
    
    def train(self) -> Dict[str, float]:
        """
        Run the full training pipeline.
        
        Returns:
            Dictionary with training metrics
        """
        # Load and preprocess data
        df = load_kaggle_data()
        clean_df, self.encoders, self.scalers = preprocess_data(df)
        features, target = create_tensors(clean_df, self.encoders, self.scalers)
        train_loader, val_loader = create_dataloaders(
            features, target, self.batch_size
        )
        
        # Initialize model
        model_params = self._get_model_params()
        self.model = CropYieldLSTM(**model_params).to(self.device)
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.5
        )
        
        # Training loop
        best_val_loss = float('inf')
        history = {'train_loss': [], 'val_loss': []}
        
        print(f"\nTraining on {self.device}...")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print("-" * 50)
        
        for epoch in range(self.epochs):
            train_loss = self._train_epoch(train_loader, criterion, optimizer)
            val_loss = self._validate_epoch(val_loader, criterion)
            
            scheduler.step(val_loss)
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint(model_params)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}/{self.epochs} | "
                      f"Train: {train_loss:.4f} | Val: {val_loss:.4f}")
        
        print("-" * 50)
        print(f"Training complete! Best val loss: {best_val_loss:.4f}")
        
        return {
            'best_val_loss': best_val_loss,
            'final_train_loss': history['train_loss'][-1],
            'final_val_loss': history['val_loss'][-1]
        }
    
    def _train_epoch(
        self,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer
    ) -> float:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        
        for batch in loader:
            state, crop, season, continuous, y = [b.to(self.device) for b in batch]
            
            optimizer.zero_grad()
            y_pred = self.model(state, crop, season, continuous)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(loader)
    
    def _validate_epoch(self, loader: DataLoader, criterion: nn.Module) -> float:
        """Run validation epoch."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in loader:
                state, crop, season, continuous, y = [b.to(self.device) for b in batch]
                y_pred = self.model(state, crop, season, continuous)
                loss = criterion(y_pred, y)
                total_loss += loss.item()
        
        return total_loss / len(loader)
    
    def _get_model_params(self) -> Dict:
        """Get model architecture parameters."""
        return {
            'num_states': len(self.encoders['state']),
            'num_crops': len(self.encoders['crop']),
            'num_seasons': len(self.encoders['season']),
            'embedding_dim': self.embedding_dim,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout': self.dropout
        }
    
    def _save_checkpoint(self, model_params: Dict) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'model_params': model_params,
            'encoders': self.encoders,
            'scalers': self.scalers
        }
        
        path = self.checkpoint_dir / 'crop_yield_model.pt'
        torch.save(checkpoint, path)


def main():
    """Main entry point for training."""
    print("=" * 60)
    print("Crop Yield Prediction Model - Training")
    print("=" * 60)
    
    trainer = CropYieldTrainer(
        hidden_size=64,
        num_layers=2,
        embedding_dim=16,
        dropout=0.2,
        learning_rate=0.001,
        batch_size=32,
        epochs=50
    )
    
    metrics = trainer.train()
    
    print("\n" + "=" * 60)
    print("Training Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    print("=" * 60)
    print("\nModel saved to: models/checkpoints/crop_yield_model.pt")


if __name__ == "__main__":
    main()
