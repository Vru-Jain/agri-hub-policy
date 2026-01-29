"""
Training Script for Crop Yield Prediction Model.

This script:
1. Loads crop yield data from Kaggle
2. Preprocesses and engineers features
3. Trains the LSTM model
4. Saves the trained model checkpoint
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
import json
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.crop_yield_model import CropYieldLSTM
from services.kaggle_service import get_kaggle_service


class CropYieldTrainer:
    """Trainer class for the crop yield prediction model."""
    
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
        
        # Paths
        self.checkpoint_dir = Path(__file__).parent / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)
    
    def load_data(self) -> pd.DataFrame:
        """Load crop yield data from Kaggle."""
        print("Loading data from Kaggle...")
        kaggle = get_kaggle_service()
        
        # Get crop yield data
        df = kaggle.get_crop_yield_data()
        print(f"Loaded {len(df)} records")
        
        return df
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Preprocess data for training.
        
        Args:
            df: Raw DataFrame with crop yield data
            
        Returns:
            Tuple of (feature tensors dict, target tensor)
        """
        print("Preprocessing data...")
        
        # Standardize column names
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Handle different column naming conventions
        state_col = next((c for c in df.columns if 'state' in c), None)
        crop_col = next((c for c in df.columns if 'crop' in c), None)
        year_col = next((c for c in df.columns if 'year' in c), None)
        yield_col = next((c for c in df.columns if 'yield' in c), None)
        area_col = next((c for c in df.columns if 'area' in c), None)
        
        if not all([state_col, crop_col, yield_col]):
            raise ValueError(f"Required columns not found. Available: {df.columns.tolist()}")
        
        # Create a clean dataframe
        clean_df = pd.DataFrame({
            'state': df[state_col].astype(str),
            'crop': df[crop_col].astype(str),
            'year': df[year_col] if year_col else 2020,
            'yield': df[yield_col].astype(float),
            'area': df[area_col].astype(float) if area_col else 100000.0
        })
        
        # Add simulated features (in real scenario, would join with actual data)
        np.random.seed(42)
        clean_df['ndvi'] = np.random.uniform(0.3, 0.9, len(clean_df))
        clean_df['soil_moisture'] = np.random.uniform(0.2, 0.8, len(clean_df))
        clean_df['rainfall'] = np.random.uniform(400, 1500, len(clean_df))
        
        # Determine season based on crop type
        kharif_crops = ['rice', 'maize', 'cotton', 'groundnut', 'soyabean', 'sugarcane']
        clean_df['season'] = clean_df['crop'].str.lower().apply(
            lambda x: 'Kharif' if any(k in x.lower() for k in kharif_crops) else 'Rabi'
        )
        
        # Remove invalid yields
        clean_df = clean_df[clean_df['yield'] > 0].dropna()
        
        print(f"Clean data: {len(clean_df)} records")
        
        # Create encoders for categorical variables
        self.encoders['state'] = {v: i for i, v in enumerate(clean_df['state'].unique())}
        self.encoders['crop'] = {v: i for i, v in enumerate(clean_df['crop'].unique())}
        self.encoders['season'] = {'Kharif': 0, 'Rabi': 1}
        
        # Encode categorical features
        state_encoded = clean_df['state'].map(self.encoders['state']).values
        crop_encoded = clean_df['crop'].map(self.encoders['crop']).values
        season_encoded = clean_df['season'].map(self.encoders['season']).values
        
        # Scale continuous features
        continuous_features = ['ndvi', 'soil_moisture', 'rainfall', 'area', 'year']
        continuous_data = []
        
        for feat in continuous_features:
            values = clean_df[feat].values.astype(float)
            mean, std = values.mean(), values.std()
            self.scalers[feat] = (float(mean), float(std))
            scaled = (values - mean) / (std + 1e-8)
            continuous_data.append(scaled)
        
        continuous_array = np.column_stack(continuous_data)
        
        # Target
        target = clean_df['yield'].values
        
        # Convert to tensors
        features = {
            'state': torch.tensor(state_encoded, dtype=torch.long),
            'crop': torch.tensor(crop_encoded, dtype=torch.long),
            'season': torch.tensor(season_encoded, dtype=torch.long),
            'continuous': torch.tensor(continuous_array, dtype=torch.float32)
        }
        
        target_tensor = torch.tensor(target, dtype=torch.float32).unsqueeze(1)
        
        return features, target_tensor
    
    def create_dataloaders(
        self,
        features: Dict[str, torch.Tensor],
        target: torch.Tensor,
        train_split: float = 0.8
    ) -> Tuple[DataLoader, DataLoader]:
        """Create train and validation dataloaders."""
        
        n_samples = len(target)
        indices = torch.randperm(n_samples)
        train_size = int(n_samples * train_split)
        
        train_idx = indices[:train_size]
        val_idx = indices[train_size:]
        
        # Create datasets
        train_dataset = TensorDataset(
            features['state'][train_idx],
            features['crop'][train_idx],
            features['season'][train_idx],
            features['continuous'][train_idx],
            target[train_idx]
        )
        
        val_dataset = TensorDataset(
            features['state'][val_idx],
            features['crop'][val_idx],
            features['season'][val_idx],
            features['continuous'][val_idx],
            target[val_idx]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        
        return train_loader, val_loader
    
    def train(self) -> Dict[str, float]:
        """
        Main training loop.
        
        Returns:
            Dictionary with training metrics
        """
        # Load and preprocess data
        df = self.load_data()
        features, target = self.preprocess_data(df)
        train_loader, val_loader = self.create_dataloaders(features, target)
        
        # Initialize model
        model_params = {
            'num_states': len(self.encoders['state']),
            'num_crops': len(self.encoders['crop']),
            'num_seasons': len(self.encoders['season']),
            'embedding_dim': self.embedding_dim,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout': self.dropout
        }
        
        self.model = CropYieldLSTM(**model_params).to(self.device)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        best_val_loss = float('inf')
        history = {'train_loss': [], 'val_loss': []}
        
        print(f"\nTraining on {self.device}...")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print("-" * 50)
        
        for epoch in range(self.epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for batch in train_loader:
                state, crop, season, continuous, y = [b.to(self.device) for b in batch]
                
                optimizer.zero_grad()
                y_pred = self.model(state, crop, season, continuous)
                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch in val_loader:
                    state, crop, season, continuous, y = [b.to(self.device) for b in batch]
                    y_pred = self.model(state, crop, season, continuous)
                    loss = criterion(y_pred, y)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            scheduler.step(val_loss)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint(model_params)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}/{self.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        print("-" * 50)
        print(f"Training complete! Best validation loss: {best_val_loss:.4f}")
        
        return {
            'best_val_loss': best_val_loss,
            'final_train_loss': history['train_loss'][-1],
            'final_val_loss': history['val_loss'][-1]
        }
    
    def _save_checkpoint(self, model_params: Dict):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'model_params': model_params,
            'encoders': self.encoders,
            'scalers': self.scalers
        }
        
        checkpoint_path = self.checkpoint_dir / 'crop_yield_model.pt'
        torch.save(checkpoint, checkpoint_path)


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
        epochs=50  # Reduced for faster training
    )
    
    metrics = trainer.train()
    
    print("\n" + "=" * 60)
    print("Training Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    print("=" * 60)
    
    print(f"\nModel saved to: models/checkpoints/crop_yield_model.pt")


if __name__ == "__main__":
    main()
