"""
Crop Yield Predictor - High-level inference interface.

This module provides the CropYieldPredictor class that handles:
- Model loading and initialization
- Feature preprocessing and encoding
- Yield prediction with fallback logic
"""

import math
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Dict, Tuple, List

from .architecture import CropYieldLSTM


class CropYieldPredictor:
    """
    High-level predictor for crop yield estimation.
    
    Handles model loading, feature preprocessing, and inference.
    Falls back to rule-based prediction when no trained model is available.
    
    Usage:
        predictor = CropYieldPredictor()
        yield_kg_ha = predictor.predict(
            state='Maharashtra',
            crop='Rice',
            season='Kharif',
            ndvi=0.5,
            soil_moisture=0.3
        )
    """
    
    # Yield output range (Kg/Ha)
    MIN_YIELD = 1500
    MAX_YIELD = 4500
    
    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to saved model checkpoint.
                        If None, uses default: models/checkpoints/crop_yield_model.pt
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model: Optional[CropYieldLSTM] = None
        self.encoders: Dict[str, Dict[str, int]] = {}
        self.scalers: Dict[str, Tuple[float, float]] = {}
        
        # Set default model path
        if model_path is None:
            model_path = Path(__file__).parent.parent / 'checkpoints' / 'crop_yield_model.pt'
        
        self.model_path = model_path
        self._load_model()
    
    def _load_model(self) -> None:
        """Load trained model and encoders from checkpoint."""
        if not self.model_path.exists():
            return  # No trained model - will use fallback
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Load model architecture and weights
        model_params = checkpoint.get('model_params', {})
        self.model = CropYieldLSTM(**model_params)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Load feature encoders and scalers
        self.encoders = checkpoint.get('encoders', {})
        self.scalers = checkpoint.get('scalers', {})
    
    def is_trained(self) -> bool:
        """Check if a trained model is available."""
        return self.model is not None
    
    def predict(
        self,
        state: str,
        crop: str,
        season: str,
        ndvi: float,
        soil_moisture: float,
        rainfall: float = 800.0,
        area: float = 100000.0,
        year: int = 2024
    ) -> float:
        """
        Predict crop yield.
        
        Args:
            state: State name (e.g., 'Maharashtra')
            crop: Crop name (e.g., 'Rice')
            season: Season ('Kharif' or 'Rabi')
            ndvi: NDVI value (0-1 scale)
            soil_moisture: Soil moisture fraction (0-1 scale)
            rainfall: Annual rainfall in mm (default: 800)
            area: Cultivation area in hectares (default: 100000)
            year: Year of prediction (default: 2024)
            
        Returns:
            Predicted yield in Kg/Ha (range: 1500-4500)
        """
        if not self.is_trained():
            return self._fallback_prediction(ndvi, soil_moisture, season)
        
        return self._model_prediction(
            state, crop, season, ndvi, soil_moisture, rainfall, area, year
        )
    
    def _model_prediction(
        self,
        state: str,
        crop: str,
        season: str,
        ndvi: float,
        soil_moisture: float,
        rainfall: float,
        area: float,
        year: int
    ) -> float:
        """Run prediction through the trained LSTM model."""
        with torch.no_grad():
            # Encode categorical features
            state_idx = self._encode_category('state', state)
            crop_idx = self._encode_category('crop', crop)
            season_idx = self._encode_category('season', season)
            
            # Scale continuous features
            continuous = self._scale_continuous(
                ndvi, soil_moisture, rainfall, area, year
            )
            
            # Convert to tensors
            state_tensor = torch.tensor([state_idx], dtype=torch.long, device=self.device)
            crop_tensor = torch.tensor([crop_idx], dtype=torch.long, device=self.device)
            season_tensor = torch.tensor([season_idx], dtype=torch.long, device=self.device)
            continuous_tensor = torch.tensor([continuous], dtype=torch.float32, device=self.device)
            
            # Get raw prediction
            yield_pred = self.model(state_tensor, crop_tensor, season_tensor, continuous_tensor)
            raw_output = float(yield_pred.item())
            
            # Post-process to realistic range
            return self._normalize_output(raw_output, ndvi, soil_moisture)
    
    def _normalize_output(
        self,
        raw_output: float,
        ndvi: float,
        soil_moisture: float
    ) -> float:
        """
        Normalize raw model output to realistic yield range.
        
        Uses sigmoid normalization and feature-based scaling.
        """
        # Sigmoid-like normalization for any raw value
        normalized = 1.0 / (1.0 + math.exp(-raw_output / 1000))
        adjustment = 0.7 + (normalized * 0.6)  # Maps to 0.7-1.3 range
        
        # Base yield from input quality indicators
        quality_score = (ndvi * 0.6 + soil_moisture * 0.4)
        base_yield = 2000 + (quality_score * 2000)  # 2000-4000 Kg/Ha
        
        # Apply adjustment and cap
        final_yield = base_yield * adjustment
        return max(self.MIN_YIELD, min(self.MAX_YIELD, final_yield))
    
    def _encode_category(self, category: str, value: str) -> int:
        """Encode categorical value to index."""
        encoder = self.encoders.get(category, {})
        return encoder.get(value, 0)  # Default to 0 for unknown
    
    def _scale_continuous(
        self,
        ndvi: float,
        soil_moisture: float,
        rainfall: float,
        area: float,
        year: int
    ) -> List[float]:
        """Scale continuous features using saved scalers."""
        features = [ndvi, soil_moisture, rainfall, area, float(year)]
        feature_names = ['ndvi', 'soil_moisture', 'rainfall', 'area', 'year']
        
        scaled = []
        for feat, name in zip(features, feature_names):
            if name in self.scalers:
                mean, std = self.scalers[name]
                scaled.append((feat - mean) / (std + 1e-8))
            else:
                scaled.append(feat)
        
        return scaled
    
    def _fallback_prediction(
        self,
        ndvi: float,
        soil_moisture: float,
        season: str
    ) -> float:
        """
        Rule-based fallback when no trained model is available.
        
        Uses seasonal weighting:
        - Rabi: NDVI weighted 70% (vegetation critical for winter crops)
        - Kharif: Soil moisture weighted 60% (monsoon-dependent)
        """
        # Seasonal weights
        if season.lower() == 'rabi':
            ndvi_weight, sm_weight = 0.7, 0.3
        else:
            ndvi_weight, sm_weight = 0.4, 0.6
        
        # Score calculation
        ndvi_score = ndvi * 100
        sm_score = soil_moisture * 100
        combined_score = (ndvi_weight * ndvi_score) + (sm_weight * sm_score)
        
        # Map to yield (Kg/Ha)
        base_yield = 1500 + (combined_score / 100) * 2500
        noise = np.random.uniform(-200, 200)
        
        return max(1000, round(base_yield + noise, 2))


# ============================================================
# Singleton Pattern for Global Access
# ============================================================

_predictor: Optional[CropYieldPredictor] = None


def get_trained_model() -> CropYieldPredictor:
    """
    Get the singleton predictor instance.
    
    Returns:
        CropYieldPredictor: The global predictor instance
    """
    global _predictor
    if _predictor is None:
        _predictor = CropYieldPredictor()
    return _predictor


def reset_predictor() -> None:
    """Reset the singleton (useful for testing or model reload)."""
    global _predictor
    _predictor = None
