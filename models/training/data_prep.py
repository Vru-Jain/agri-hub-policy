"""
Data Preparation for Crop Yield Model Training.

This module handles:
- Loading data from Kaggle
- Preprocessing and feature engineering
- Creating encoders and scalers
- Converting to PyTorch tensors
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from typing import Dict, Tuple, Optional
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_kaggle_data() -> pd.DataFrame:
    """
    Load crop yield data from Kaggle service.
    
    Returns:
        DataFrame with raw crop yield data
    
    Raises:
        ImportError: If Kaggle service is not available
    """
    from services.kaggle_service import get_kaggle_service
    
    print("Loading data from Kaggle...")
    kaggle = get_kaggle_service()
    df = kaggle.get_crop_yield_data()
    print(f"Loaded {len(df)} records")
    
    return df


def preprocess_data(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, Dict], Dict[str, Tuple[float, float]]]:
    """
    Preprocess raw data for training.
    
    Args:
        df: Raw DataFrame with crop yield data
        
    Returns:
        Tuple of (clean DataFrame, encoders dict, scalers dict)
    """
    print("Preprocessing data...")
    
    # Standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    
    # Find required columns
    state_col = next((c for c in df.columns if 'state' in c), None)
    crop_col = next((c for c in df.columns if 'crop' in c), None)
    year_col = next((c for c in df.columns if 'year' in c), None)
    yield_col = next((c for c in df.columns if 'yield' in c), None)
    area_col = next((c for c in df.columns if 'area' in c), None)
    
    if not all([state_col, crop_col, yield_col]):
        raise ValueError(f"Required columns not found. Available: {df.columns.tolist()}")
    
    # Create clean dataframe
    clean_df = pd.DataFrame({
        'state': df[state_col].astype(str),
        'crop': df[crop_col].astype(str),
        'year': df[year_col] if year_col else 2020,
        'yield': df[yield_col].astype(float),
        'area': df[area_col].astype(float) if area_col else 100000.0
    })
    
    # Add simulated features (would be joined from real data in production)
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
    
    # Create encoders
    encoders = {
        'state': {v: i for i, v in enumerate(clean_df['state'].unique())},
        'crop': {v: i for i, v in enumerate(clean_df['crop'].unique())},
        'season': {'Kharif': 0, 'Rabi': 1}
    }
    
    # Create scalers for continuous features
    scalers = {}
    for feat in ['ndvi', 'soil_moisture', 'rainfall', 'area', 'year']:
        values = clean_df[feat].values.astype(float)
        scalers[feat] = (float(values.mean()), float(values.std()))
    
    return clean_df, encoders, scalers


def create_tensors(
    df: pd.DataFrame,
    encoders: Dict[str, Dict],
    scalers: Dict[str, Tuple[float, float]]
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """
    Convert preprocessed data to PyTorch tensors.
    
    Args:
        df: Clean preprocessed DataFrame
        encoders: Categorical encoders
        scalers: Feature scalers (mean, std)
        
    Returns:
        Tuple of (features dict, target tensor)
    """
    # Encode categorical features
    state_encoded = df['state'].map(encoders['state']).values
    crop_encoded = df['crop'].map(encoders['crop']).values
    season_encoded = df['season'].map(encoders['season']).values
    
    # Scale continuous features
    continuous_features = ['ndvi', 'soil_moisture', 'rainfall', 'area', 'year']
    continuous_data = []
    
    for feat in continuous_features:
        values = df[feat].values.astype(float)
        mean, std = scalers[feat]
        scaled = (values - mean) / (std + 1e-8)
        continuous_data.append(scaled)
    
    continuous_array = np.column_stack(continuous_data)
    
    # Create feature tensors
    features = {
        'state': torch.tensor(state_encoded, dtype=torch.long),
        'crop': torch.tensor(crop_encoded, dtype=torch.long),
        'season': torch.tensor(season_encoded, dtype=torch.long),
        'continuous': torch.tensor(continuous_array, dtype=torch.float32)
    }
    
    target = torch.tensor(df['yield'].values, dtype=torch.float32).unsqueeze(1)
    
    return features, target


def create_dataloaders(
    features: Dict[str, torch.Tensor],
    target: torch.Tensor,
    batch_size: int = 32,
    train_split: float = 0.8
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders.
    
    Args:
        features: Feature tensors dict
        target: Target tensor
        batch_size: Batch size for training
        train_split: Fraction of data for training
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
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
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    return train_loader, val_loader
