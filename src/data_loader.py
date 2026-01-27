"""
Data Loader Module
Handles loading datasets from various file formats (CSV, Excel, JSON)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import io


class DataLoader:
    """Utility class for loading and validating datasets."""
    
    SUPPORTED_FORMATS = {
        'csv': ['.csv'],
        'excel': ['.xlsx', '.xls'],
        'json': ['.json']
    }
    
    @staticmethod
    def load_from_file(file_obj, filename: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Load a dataset from an uploaded file object.
        
        Args:
            file_obj: File-like object (from Streamlit uploader)
            filename: Original filename to determine format
            
        Returns:
            Tuple of (DataFrame, metadata dict)
        """
        ext = Path(filename).suffix.lower()
        
        try:
            if ext in DataLoader.SUPPORTED_FORMATS['csv']:
                df = DataLoader._load_csv(file_obj)
            elif ext in DataLoader.SUPPORTED_FORMATS['excel']:
                df = DataLoader._load_excel(file_obj)
            elif ext in DataLoader.SUPPORTED_FORMATS['json']:
                df = DataLoader._load_json(file_obj)
            else:
                raise ValueError(f"Unsupported file format: {ext}")
            
            metadata = DataLoader._generate_metadata(df)
            return df, metadata
            
        except Exception as e:
            raise ValueError(f"Error loading file: {str(e)}")
    
    @staticmethod
    def _load_csv(file_obj) -> pd.DataFrame:
        """Load CSV file with encoding detection."""
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        
        for encoding in encodings:
            try:
                file_obj.seek(0)
                return pd.read_csv(file_obj, encoding=encoding)
            except UnicodeDecodeError:
                continue
        
        raise ValueError("Could not decode CSV file with common encodings")
    
    @staticmethod
    def _load_excel(file_obj) -> pd.DataFrame:
        """Load Excel file."""
        return pd.read_excel(file_obj, engine='openpyxl')
    
    @staticmethod
    def _load_json(file_obj) -> pd.DataFrame:
        """Load JSON file."""
        return pd.read_json(file_obj)
    
    @staticmethod
    def _generate_metadata(df: pd.DataFrame) -> Dict[str, Any]:
        """Generate metadata summary for a DataFrame."""
        
        # Detect column types
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        bool_cols = df.select_dtypes(include=['bool']).columns.tolist()
        
        # Calculate missing values
        missing_counts = df.isnull().sum().to_dict()
        missing_percentages = (df.isnull().sum() / len(df) * 100).to_dict()
        
        # Unique value counts
        unique_counts = {col: df[col].nunique() for col in df.columns}
        
        # Memory usage
        memory_usage = df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
        
        return {
            'n_rows': len(df),
            'n_cols': len(df.columns),
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'numeric_columns': numeric_cols,
            'categorical_columns': categorical_cols,
            'datetime_columns': datetime_cols,
            'boolean_columns': bool_cols,
            'missing_counts': missing_counts,
            'missing_percentages': missing_percentages,
            'unique_counts': unique_counts,
            'memory_mb': round(memory_usage, 2),
            'has_duplicates': df.duplicated().any(),
            'duplicate_count': df.duplicated().sum()
        }
    
    @staticmethod
    def get_column_stats(df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """Get detailed statistics for a specific column."""
        col_data = df[column]
        stats = {
            'name': column,
            'dtype': str(col_data.dtype),
            'count': len(col_data),
            'missing': col_data.isnull().sum(),
            'missing_pct': round(col_data.isnull().sum() / len(col_data) * 100, 2),
            'unique': col_data.nunique()
        }
        
        if pd.api.types.is_numeric_dtype(col_data):
            stats.update({
                'mean': round(col_data.mean(), 4) if not col_data.isnull().all() else None,
                'std': round(col_data.std(), 4) if not col_data.isnull().all() else None,
                'min': col_data.min() if not col_data.isnull().all() else None,
                'max': col_data.max() if not col_data.isnull().all() else None,
                'median': col_data.median() if not col_data.isnull().all() else None,
                'q25': col_data.quantile(0.25) if not col_data.isnull().all() else None,
                'q75': col_data.quantile(0.75) if not col_data.isnull().all() else None,
                'skewness': round(col_data.skew(), 4) if not col_data.isnull().all() else None,
                'kurtosis': round(col_data.kurtosis(), 4) if not col_data.isnull().all() else None
            })
        else:
            # Categorical stats
            value_counts = col_data.value_counts().head(10).to_dict()
            stats.update({
                'top_values': value_counts,
                'mode': col_data.mode().iloc[0] if len(col_data.mode()) > 0 else None
            })
        
        return stats
    
    @staticmethod
    def detect_target_column(df: pd.DataFrame) -> Optional[str]:
        """
        Attempt to auto-detect the target column based on common naming conventions.
        
        Returns:
            Column name if detected, None otherwise
        """
        common_target_names = [
            'target', 'label', 'class', 'y', 'outcome', 'result',
            'survived', 'churn', 'default', 'fraud', 'click'
        ]
        
        for col in df.columns:
            if col.lower() in common_target_names:
                return col
        
        # Check for columns at the end that might be targets
        last_col = df.columns[-1]
        if df[last_col].nunique() <= 10:  # Likely a classification target
            return last_col
        
        return None
