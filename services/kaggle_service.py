"""
Kaggle Service for fetching agricultural datasets.

Provides access to Kaggle datasets for:
- Historical crop yield data
- Soil quality information
- Agricultural commodity prices
- Weather/climate data for agriculture
"""

import os
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any
import zipfile
import tempfile
import shutil

from .config import get_kaggle_credentials


# Popular agricultural datasets on Kaggle
AGRICULTURAL_DATASETS = {
    'crop_yield_india': {
        'dataset': 'akshatgupta7/crop-yield-in-indian-states-dataset',
        'description': 'Crop yield data across Indian states (1997-2020)',
        'files': ['crop_yield.csv']
    },
    'crop_production': {
        'dataset': 'abhinand05/crop-production-in-india',
        'description': 'Crop production statistics in India',
        'files': ['crop_production.csv']
    },
    'agriculture_india': {
        'dataset': 'srinivas1/agriculture-crops-production-in-india',
        'description': 'Agricultural crops production dataset',
        'files': ['datafile.csv']
    },
    'soil_quality': {
        'dataset': 'cdminix/us-drought-meteorological-data',
        'description': 'Meteorological data for drought prediction',
        'files': ['train_timeseries.csv']
    },
    'rainfall_india': {
        'dataset': 'rajanand/rainfall-in-india',
        'description': 'Historical rainfall data in India (1901-2015)',
        'files': ['rainfall in india 1901-2015.csv']
    }
}


class KaggleService:
    """Service for interacting with Kaggle datasets."""
    
    def __init__(self):
        """Initialize Kaggle service with credentials."""
        self.credentials = get_kaggle_credentials()
        self.is_configured = self.credentials is not None
        self.cache_dir = Path(__file__).parent.parent / 'data' / 'kaggle_cache'
        
        if self.is_configured:
            # Set environment variables for Kaggle API
            os.environ['KAGGLE_USERNAME'] = self.credentials['username']
            os.environ['KAGGLE_KEY'] = self.credentials['key']
            
            # Ensure cache directory exists
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_kaggle_api(self):
        """Get authenticated Kaggle API instance."""
        if not self.is_configured:
            raise ValueError("Kaggle credentials not configured")
        
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            return api
        except ImportError:
            raise ImportError("Kaggle package not installed. Run: pip install kaggle")
    
    def list_available_datasets(self) -> List[Dict[str, str]]:
        """
        List pre-configured agricultural datasets.
        
        Returns:
            List of dataset info dictionaries
        """
        return [
            {
                'id': key,
                'dataset': info['dataset'],
                'description': info['description']
            }
            for key, info in AGRICULTURAL_DATASETS.items()
        ]
    
    def search_datasets(self, query: str = "agriculture india", max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search for datasets on Kaggle.
        
        Args:
            query: Search query string
            max_results: Maximum number of results
            
        Returns:
            List of matching datasets
        """
        if not self.is_configured:
            return self._get_demo_search_results()
        
        try:
            api = self._get_kaggle_api()
            datasets = api.dataset_list(search=query, page=1, page_size=max_results)
            
            return [
                {
                    'ref': str(ds.ref),
                    'title': ds.title,
                    'size': ds.size,
                    'last_updated': str(ds.lastUpdated),
                    'download_count': ds.downloadCount,
                    'vote_count': ds.voteCount
                }
                for ds in datasets
            ]
        except Exception as e:
            print(f"Error searching Kaggle: {e}")
            return self._get_demo_search_results()
    
    def download_dataset(self, dataset_id: str, force: bool = False) -> Optional[Path]:
        """
        Download a dataset from Kaggle.
        
        Args:
            dataset_id: Either a key from AGRICULTURAL_DATASETS or a Kaggle dataset ref
            force: Force re-download even if cached
            
        Returns:
            Path to downloaded dataset directory, or None on failure
        """
        if not self.is_configured:
            return None
        
        # Resolve dataset reference
        if dataset_id in AGRICULTURAL_DATASETS:
            dataset_ref = AGRICULTURAL_DATASETS[dataset_id]['dataset']
        else:
            dataset_ref = dataset_id
        
        # Create dataset-specific cache directory
        safe_name = dataset_ref.replace('/', '_')
        dataset_dir = self.cache_dir / safe_name
        
        # Check if already cached
        if dataset_dir.exists() and not force:
            return dataset_dir
        
        try:
            api = self._get_kaggle_api()
            
            # Download to temporary directory first
            with tempfile.TemporaryDirectory() as tmp_dir:
                api.dataset_download_files(dataset_ref, path=tmp_dir, unzip=True)
                
                # Move to cache
                if dataset_dir.exists():
                    shutil.rmtree(dataset_dir)
                shutil.move(tmp_dir, dataset_dir)
            
            return dataset_dir
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            return None
    
    def load_dataset(self, dataset_id: str, filename: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Load a dataset as a pandas DataFrame.
        
        Args:
            dataset_id: Either a key from AGRICULTURAL_DATASETS or a Kaggle dataset ref
            filename: Specific file to load (optional, uses first CSV if not specified)
            
        Returns:
            DataFrame or None on failure
        """
        if not self.is_configured:
            return self._get_demo_data(dataset_id)
        
        dataset_dir = self.download_dataset(dataset_id)
        if not dataset_dir:
            return self._get_demo_data(dataset_id)
        
        # Find the file to load
        if filename:
            file_path = dataset_dir / filename
        else:
            # Use predefined file or find first CSV
            if dataset_id in AGRICULTURAL_DATASETS:
                files = AGRICULTURAL_DATASETS[dataset_id].get('files', [])
                if files:
                    file_path = dataset_dir / files[0]
                else:
                    csv_files = list(dataset_dir.glob('*.csv'))
                    file_path = csv_files[0] if csv_files else None
            else:
                csv_files = list(dataset_dir.glob('*.csv'))
                file_path = csv_files[0] if csv_files else None
        
        if not file_path or not file_path.exists():
            return None
        
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return None
    
    def get_crop_yield_data(self) -> pd.DataFrame:
        """
        Get Indian crop yield data.
        
        Returns:
            DataFrame with crop yield statistics
        """
        df = self.load_dataset('crop_yield_india')
        if df is None:
            df = self._get_demo_crop_yield()
        return df
    
    def get_rainfall_data(self) -> pd.DataFrame:
        """
        Get historical rainfall data for India.
        
        Returns:
            DataFrame with rainfall statistics
        """
        df = self.load_dataset('rainfall_india')
        if df is None:
            df = self._get_demo_rainfall()
        return df
    
    def _get_demo_search_results(self) -> List[Dict[str, Any]]:
        """Return demo search results when API is not available."""
        return [
            {
                'ref': 'akshatgupta7/crop-yield-in-indian-states-dataset',
                'title': 'Crop Yield in Indian States Dataset',
                'size': '15KB',
                'last_updated': '2023-01-15',
                'download_count': 5000,
                'vote_count': 45
            },
            {
                'ref': 'abhinand05/crop-production-in-india',
                'title': 'Crop Production in India',
                'size': '1.2MB',
                'last_updated': '2022-08-20',
                'download_count': 12000,
                'vote_count': 89
            },
            {
                'ref': 'rajanand/rainfall-in-india',
                'title': 'Rainfall in India 1901-2015',
                'size': '500KB',
                'last_updated': '2021-05-10',
                'download_count': 8000,
                'vote_count': 67
            }
        ]
    
    def _get_demo_data(self, dataset_id: str) -> Optional[pd.DataFrame]:
        """Return demo data based on dataset type."""
        if 'crop' in dataset_id.lower() or 'yield' in dataset_id.lower():
            return self._get_demo_crop_yield()
        elif 'rain' in dataset_id.lower():
            return self._get_demo_rainfall()
        return None
    
    def _get_demo_crop_yield(self) -> pd.DataFrame:
        """Generate demo crop yield data."""
        import numpy as np
        
        states = ['Maharashtra', 'Punjab', 'Uttar Pradesh', 'Madhya Pradesh', 'Karnataka']
        crops = ['Rice', 'Wheat', 'Cotton', 'Sugarcane', 'Maize']
        years = list(range(2015, 2024))
        
        data = []
        for state in states:
            for crop in crops:
                for year in years:
                    base_yield = np.random.uniform(1500, 4000)
                    yield_val = base_yield * (1 + np.random.uniform(-0.1, 0.15))
                    data.append({
                        'State': state,
                        'Crop': crop,
                        'Year': year,
                        'Yield_Kg_per_Ha': round(yield_val, 2),
                        'Area_Ha': np.random.randint(50000, 500000),
                        'Production_Tonnes': round(yield_val * np.random.randint(50000, 500000) / 1000, 2)
                    })
        
        return pd.DataFrame(data)
    
    def _get_demo_rainfall(self) -> pd.DataFrame:
        """Generate demo rainfall data."""
        import numpy as np
        
        subdivisions = ['Coastal Andhra Pradesh', 'Telangana', 'Rayalaseema', 
                        'Tamil Nadu', 'Karnataka', 'Kerala', 'Maharashtra']
        years = list(range(2015, 2024))
        months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 
                  'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
        
        data = []
        for subdiv in subdivisions:
            for year in years:
                row = {'SUBDIVISION': subdiv, 'YEAR': year}
                annual = 0
                for month in months:
                    # Monsoon months get more rainfall
                    if month in ['JUN', 'JUL', 'AUG', 'SEP']:
                        rainfall = np.random.uniform(150, 400)
                    else:
                        rainfall = np.random.uniform(10, 100)
                    row[month] = round(rainfall, 1)
                    annual += rainfall
                row['ANNUAL'] = round(annual, 1)
                data.append(row)
        
        return pd.DataFrame(data)


# Singleton instance
_kaggle_service: Optional[KaggleService] = None


def get_kaggle_service() -> KaggleService:
    """Get the Kaggle service singleton instance."""
    global _kaggle_service
    if _kaggle_service is None:
        _kaggle_service = KaggleService()
    return _kaggle_service
