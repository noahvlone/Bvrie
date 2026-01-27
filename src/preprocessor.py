"""
Preprocessor Module
Handles data cleaning, transformation, and preparation for feature analysis
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional, Any
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import warnings

warnings.filterwarnings('ignore')


class Preprocessor:
    """Data preprocessing and cleaning utilities."""
    
    def __init__(self):
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scalers: Dict[str, Any] = {}
        self.imputers: Dict[str, SimpleImputer] = {}
        self.feature_names: List[str] = []
        self.original_dtypes: Dict[str, str] = {}
    
    def prepare_for_analysis(
        self,
        df: pd.DataFrame,
        target_column: str,
        handle_missing: str = 'auto',
        encode_categorical: bool = True,
        scale_features: bool = False,
        remove_high_cardinality: bool = True,
        cardinality_threshold: int = 50
    ) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        """
        Prepare dataset for feature importance analysis.
        
        Args:
            df: Input DataFrame
            target_column: Name of the target column
            handle_missing: Strategy for missing values ('auto', 'drop', 'mean', 'median', 'mode')
            encode_categorical: Whether to encode categorical variables
            scale_features: Whether to scale numerical features
            remove_high_cardinality: Whether to remove high-cardinality categorical columns
            cardinality_threshold: Max unique values for categorical columns
            
        Returns:
            Tuple of (processed features DataFrame, target Series, preprocessing info dict)
        """
        df_processed = df.copy()
        preprocessing_info = {
            'original_shape': df.shape,
            'dropped_columns': [],
            'encoded_columns': [],
            'imputed_columns': [],
            'scaled_columns': [],
            'warnings': []
        }
        
        # Store original dtypes
        self.original_dtypes = df.dtypes.astype(str).to_dict()
        
        # Separate target
        if target_column not in df_processed.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        y = df_processed[target_column].copy()
        X = df_processed.drop(columns=[target_column])
        
        # Handle target encoding if categorical
        if y.dtype == 'object' or y.dtype.name == 'category':
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y.astype(str)), name=target_column)
            self.label_encoders['__target__'] = le
            preprocessing_info['target_encoded'] = True
            preprocessing_info['target_classes'] = list(le.classes_)
        else:
            preprocessing_info['target_encoded'] = False
        
        # Remove constant columns
        constant_cols = [col for col in X.columns if X[col].nunique() <= 1]
        if constant_cols:
            X = X.drop(columns=constant_cols)
            preprocessing_info['dropped_columns'].extend(constant_cols)
            preprocessing_info['warnings'].append(
                f"Removed {len(constant_cols)} constant columns: {constant_cols[:5]}..."
            )
        
        # Remove high cardinality categorical columns
        if remove_high_cardinality:
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns
            high_card_cols = [
                col for col in categorical_cols 
                if X[col].nunique() > cardinality_threshold
            ]
            if high_card_cols:
                X = X.drop(columns=high_card_cols)
                preprocessing_info['dropped_columns'].extend(high_card_cols)
                preprocessing_info['warnings'].append(
                    f"Removed {len(high_card_cols)} high-cardinality columns: {high_card_cols[:5]}..."
                )
        
        # Handle missing values
        X, imputed_cols = self._handle_missing_values(X, handle_missing)
        preprocessing_info['imputed_columns'] = imputed_cols
        
        # Handle missing in target
        y_missing_mask = y.isnull()
        if y_missing_mask.any():
            X = X[~y_missing_mask]
            y = y[~y_missing_mask]
            preprocessing_info['warnings'].append(
                f"Removed {y_missing_mask.sum()} rows with missing target values"
            )
        
        # Encode categorical variables
        if encode_categorical:
            X, encoded_cols = self._encode_categorical(X)
            preprocessing_info['encoded_columns'] = encoded_cols
        
        # Scale features
        if scale_features:
            X, scaled_cols = self._scale_features(X)
            preprocessing_info['scaled_columns'] = scaled_cols
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        preprocessing_info['final_shape'] = X.shape
        preprocessing_info['final_features'] = self.feature_names
        
        # Reset indices
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        
        return X, y, preprocessing_info
    
    def _handle_missing_values(
        self, 
        df: pd.DataFrame, 
        strategy: str
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Handle missing values in the dataset."""
        imputed_cols = []
        
        if strategy == 'drop':
            df = df.dropna()
            return df, imputed_cols
        
        # Get columns with missing values
        missing_cols = df.columns[df.isnull().any()].tolist()
        
        if not missing_cols:
            return df, imputed_cols
        
        for col in missing_cols:
            if df[col].isnull().all():
                df = df.drop(columns=[col])
                continue
            
            if pd.api.types.is_numeric_dtype(df[col]):
                # Numeric: use mean/median
                if strategy == 'auto' or strategy == 'median':
                    fill_value = df[col].median()
                else:
                    fill_value = df[col].mean()
                df[col] = df[col].fillna(fill_value)
            else:
                # Categorical: use mode
                mode_value = df[col].mode()
                if len(mode_value) > 0:
                    df[col] = df[col].fillna(mode_value.iloc[0])
                else:
                    df[col] = df[col].fillna('Unknown')
            
            imputed_cols.append(col)
        
        return df, imputed_cols
    
    def _encode_categorical(
        self, 
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Encode categorical variables using label encoding."""
        encoded_cols = []
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le
            encoded_cols.append(col)
        
        return df, encoded_cols
    
    def _scale_features(
        self, 
        df: pd.DataFrame,
        method: str = 'standard'
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Scale numerical features."""
        scaled_cols = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if method == 'standard':
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()
        
        for col in numeric_cols:
            df[col] = scaler.fit_transform(df[[col]])
            self.scalers[col] = scaler
            scaled_cols.append(col)
        
        return df, scaled_cols
    
    def get_feature_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate a summary DataFrame with information about each feature.
        """
        info_data = []
        
        for col in df.columns:
            col_data = df[col]
            info = {
                'Feature': col,
                'Type': str(col_data.dtype),
                'Missing': col_data.isnull().sum(),
                'Missing %': round(col_data.isnull().sum() / len(col_data) * 100, 2),
                'Unique': col_data.nunique(),
                'Unique %': round(col_data.nunique() / len(col_data) * 100, 2)
            }
            
            if pd.api.types.is_numeric_dtype(col_data):
                info['Mean'] = round(col_data.mean(), 2)
                info['Std'] = round(col_data.std(), 2)
                info['Min'] = col_data.min()
                info['Max'] = col_data.max()
            else:
                info['Mean'] = '-'
                info['Std'] = '-'
                info['Min'] = '-'
                info['Max'] = '-'
            
            info_data.append(info)
        
        return pd.DataFrame(info_data)
    
    @staticmethod
    def detect_problem_type(y: pd.Series) -> str:
        """
        Detect whether the problem is classification or regression.
        
        Args:
            y: Target series
            
        Returns:
            'classification' or 'regression'
        """
        unique_values = y.nunique()
        
        if unique_values <= 20 or y.dtype == 'object':
            return 'classification'
        
        # If numeric but has few unique values relative to size, likely classification
        if unique_values / len(y) < 0.05:
            return 'classification'
        
        return 'regression'
