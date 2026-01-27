"""
Feature Analyzer Module
Multi-method feature importance analysis including statistical, tree-based, and SHAP methods
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# Conditional imports for optional dependencies
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


class FeatureAnalyzer:
    """
    Comprehensive feature importance analyzer using multiple methods.
    """
    
    def __init__(self, problem_type: str = 'classification', random_state: int = 42):
        """
        Initialize the feature analyzer.
        
        Args:
            problem_type: 'classification' or 'regression'
            random_state: Random seed for reproducibility
        """
        self.problem_type = problem_type
        self.random_state = random_state
        self.results: Dict[str, pd.DataFrame] = {}
        self.models: Dict[str, Any] = {}
        self.shap_values = None
        self.feature_names: List[str] = []
    
    def analyze_all(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        methods: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Run all feature importance analysis methods.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            methods: List of methods to run. If None, runs all available methods.
                    Options: 'correlation', 'mutual_info', 'random_forest', 'xgboost', 
                            'permutation', 'shap'
        
        Returns:
            Dictionary mapping method names to importance DataFrames
        """
        self.feature_names = X.columns.tolist()
        
        if methods is None:
            methods = ['correlation', 'mutual_info', 'random_forest']
            if HAS_XGBOOST:
                methods.append('xgboost')
            methods.append('permutation')
            if HAS_SHAP:
                methods.append('shap')
        
        results = {}
        
        for method in methods:
            try:
                if method == 'correlation':
                    results['correlation'] = self.correlation_importance(X, y)
                elif method == 'mutual_info':
                    results['mutual_info'] = self.mutual_information(X, y)
                elif method == 'random_forest':
                    results['random_forest'] = self.random_forest_importance(X, y)
                elif method == 'xgboost' and HAS_XGBOOST:
                    results['xgboost'] = self.xgboost_importance(X, y)
                elif method == 'permutation':
                    results['permutation'] = self.permutation_importance(X, y)
                elif method == 'shap' and HAS_SHAP:
                    results['shap'] = self.shap_importance(X, y)
            except Exception as e:
                print(f"Warning: {method} analysis failed: {str(e)}")
                continue
        
        self.results = results
        return results
    
    def correlation_importance(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> pd.DataFrame:
        """
        Calculate correlation-based feature importance.
        Uses Pearson correlation for numeric targets, Point-biserial for binary.
        """
        importance_data = []
        
        for col in X.columns:
            try:
                if pd.api.types.is_numeric_dtype(X[col]):
                    # Pearson correlation
                    corr, p_value = stats.pearsonr(X[col].fillna(0), y)
                    
                    # Spearman correlation (rank-based, captures non-linear relationships)
                    spearman_corr, spearman_p = stats.spearmanr(X[col].fillna(0), y)
                    
                    importance_data.append({
                        'feature': col,
                        'importance': abs(corr),
                        'correlation': corr,
                        'spearman_corr': spearman_corr,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    })
                else:
                    importance_data.append({
                        'feature': col,
                        'importance': 0,
                        'correlation': 0,
                        'spearman_corr': 0,
                        'p_value': 1.0,
                        'significant': False
                    })
            except Exception:
                importance_data.append({
                    'feature': col,
                    'importance': 0,
                    'correlation': 0,
                    'spearman_corr': 0,
                    'p_value': 1.0,
                    'significant': False
                })
        
        df = pd.DataFrame(importance_data)
        df = df.sort_values('importance', ascending=False).reset_index(drop=True)
        df['rank'] = range(1, len(df) + 1)
        
        return df
    
    def mutual_information(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> pd.DataFrame:
        """
        Calculate Mutual Information scores.
        Captures non-linear dependencies between features and target.
        """
        X_filled = X.fillna(0)
        
        if self.problem_type == 'classification':
            mi_scores = mutual_info_classif(
                X_filled, y, 
                random_state=self.random_state,
                n_neighbors=5
            )
        else:
            mi_scores = mutual_info_regression(
                X_filled, y,
                random_state=self.random_state,
                n_neighbors=5
            )
        
        df = pd.DataFrame({
            'feature': X.columns,
            'importance': mi_scores,
            'mi_score': mi_scores
        })
        
        # Normalize to 0-1 range
        if df['importance'].max() > 0:
            df['importance_normalized'] = df['importance'] / df['importance'].max()
        else:
            df['importance_normalized'] = 0
        
        df = df.sort_values('importance', ascending=False).reset_index(drop=True)
        df['rank'] = range(1, len(df) + 1)
        
        return df
    
    def random_forest_importance(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        n_estimators: int = 100
    ) -> pd.DataFrame:
        """
        Calculate feature importance using Random Forest.
        Uses Gini importance (mean decrease in impurity).
        """
        X_filled = X.fillna(0)
        
        if self.problem_type == 'classification':
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=self.random_state,
                n_jobs=-1,
                max_depth=10
            )
        else:
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                random_state=self.random_state,
                n_jobs=-1,
                max_depth=10
            )
        
        model.fit(X_filled, y)
        self.models['random_forest'] = model
        
        importance = model.feature_importances_
        
        df = pd.DataFrame({
            'feature': X.columns,
            'importance': importance,
            'gini_importance': importance
        })
        
        df = df.sort_values('importance', ascending=False).reset_index(drop=True)
        df['rank'] = range(1, len(df) + 1)
        df['cumulative_importance'] = df['importance'].cumsum()
        
        return df
    
    def xgboost_importance(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        n_estimators: int = 100
    ) -> pd.DataFrame:
        """
        Calculate feature importance using XGBoost.
        """
        if not HAS_XGBOOST:
            raise ImportError("XGBoost is not installed")
        
        X_filled = X.fillna(0)
        
        if self.problem_type == 'classification':
            n_classes = y.nunique()
            if n_classes == 2:
                model = xgb.XGBClassifier(
                    n_estimators=n_estimators,
                    random_state=self.random_state,
                    n_jobs=-1,
                    max_depth=6,
                    eval_metric='logloss'
                )
            else:
                model = xgb.XGBClassifier(
                    n_estimators=n_estimators,
                    random_state=self.random_state,
                    n_jobs=-1,
                    max_depth=6,
                    eval_metric='mlogloss'
                )
        else:
            model = xgb.XGBRegressor(
                n_estimators=n_estimators,
                random_state=self.random_state,
                n_jobs=-1,
                max_depth=6
            )
        
        model.fit(X_filled, y)
        self.models['xgboost'] = model
        
        # Get different importance types
        importance_gain = model.feature_importances_
        
        df = pd.DataFrame({
            'feature': X.columns,
            'importance': importance_gain,
            'gain_importance': importance_gain
        })
        
        df = df.sort_values('importance', ascending=False).reset_index(drop=True)
        df['rank'] = range(1, len(df) + 1)
        
        return df
    
    def permutation_importance(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        n_repeats: int = 10
    ) -> pd.DataFrame:
        """
        Calculate permutation importance (model-agnostic).
        Measures the decrease in model performance when feature values are shuffled.
        """
        X_filled = X.fillna(0)
        
        # Use Random Forest as base model if not already fitted
        if 'random_forest' not in self.models:
            if self.problem_type == 'classification':
                model = RandomForestClassifier(
                    n_estimators=50,
                    random_state=self.random_state,
                    n_jobs=-1,
                    max_depth=8
                )
            else:
                model = RandomForestRegressor(
                    n_estimators=50,
                    random_state=self.random_state,
                    n_jobs=-1,
                    max_depth=8
                )
            model.fit(X_filled, y)
        else:
            model = self.models['random_forest']
        
        perm_importance = permutation_importance(
            model, X_filled, y,
            n_repeats=n_repeats,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        df = pd.DataFrame({
            'feature': X.columns,
            'importance': perm_importance.importances_mean,
            'importance_std': perm_importance.importances_std,
            'importance_min': [min(imp) for imp in perm_importance.importances],
            'importance_max': [max(imp) for imp in perm_importance.importances]
        })
        
        df = df.sort_values('importance', ascending=False).reset_index(drop=True)
        df['rank'] = range(1, len(df) + 1)
        
        return df
    
    def shap_importance(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        max_samples: int = 1000
    ) -> pd.DataFrame:
        """
        Calculate SHAP-based feature importance.
        Provides both global importance and local explanations.
        """
        if not HAS_SHAP:
            raise ImportError("SHAP is not installed")
        
        X_filled = X.fillna(0)
        
        # Subsample for large datasets
        if len(X_filled) > max_samples:
            indices = np.random.choice(len(X_filled), max_samples, replace=False)
            X_sample = X_filled.iloc[indices]
        else:
            X_sample = X_filled
        
        # Use XGBoost model if available, otherwise Random Forest
        if 'xgboost' in self.models:
            model = self.models['xgboost']
            explainer = shap.TreeExplainer(model)
        elif 'random_forest' in self.models:
            model = self.models['random_forest']
            explainer = shap.TreeExplainer(model)
        else:
            # Train a quick model
            if self.problem_type == 'classification':
                model = RandomForestClassifier(
                    n_estimators=50,
                    random_state=self.random_state,
                    max_depth=8
                )
            else:
                model = RandomForestRegressor(
                    n_estimators=50,
                    random_state=self.random_state,
                    max_depth=8
                )
            model.fit(X_filled, y)
            explainer = shap.TreeExplainer(model)
        
        shap_values = explainer.shap_values(X_sample)
        
        # Handle multi-class case
        if isinstance(shap_values, list):
            # Take mean absolute across classes
            shap_values = np.abs(np.array(shap_values)).mean(axis=0)
        
        self.shap_values = shap_values
        
        # Calculate mean absolute SHAP values per feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        df = pd.DataFrame({
            'feature': X.columns,
            'importance': mean_abs_shap,
            'mean_shap': mean_abs_shap,
            'std_shap': np.abs(shap_values).std(axis=0)
        })
        
        df = df.sort_values('importance', ascending=False).reset_index(drop=True)
        df['rank'] = range(1, len(df) + 1)
        
        return df
    
    def get_aggregated_importance(
        self,
        weights: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Aggregate importance scores from all methods into a single ranking.
        
        Args:
            weights: Optional weights for each method. If None, equal weights used.
            
        Returns:
            DataFrame with aggregated importance scores and rankings
        """
        if not self.results:
            raise ValueError("No analysis results available. Run analyze_all() first.")
        
        # Default equal weights
        if weights is None:
            weights = {method: 1.0 for method in self.results.keys()}
        
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}
        
        # Initialize aggregated scores
        aggregated = pd.DataFrame({'feature': self.feature_names})
        aggregated['weighted_score'] = 0.0
        
        method_ranks = {}
        
        for method, df in self.results.items():
            if method not in weights:
                continue
            
            # Normalize importance to 0-1
            max_imp = df['importance'].max()
            if max_imp > 0:
                normalized = df[['feature', 'importance']].copy()
                normalized['normalized_importance'] = normalized['importance'] / max_imp
            else:
                normalized = df[['feature', 'importance']].copy()
                normalized['normalized_importance'] = 0
            
            # Merge with aggregated
            merged = aggregated.merge(
                normalized[['feature', 'normalized_importance']], 
                on='feature', 
                how='left'
            )
            aggregated['weighted_score'] += (
                merged['normalized_importance'].fillna(0) * weights[method]
            )
            
            # Store ranks
            method_ranks[f'{method}_rank'] = df.set_index('feature')['rank']
        
        # Add method-specific ranks
        for method_name, ranks in method_ranks.items():
            aggregated[method_name] = aggregated['feature'].map(ranks)
        
        # Calculate final ranking
        aggregated = aggregated.sort_values('weighted_score', ascending=False)
        aggregated['final_rank'] = range(1, len(aggregated) + 1)
        aggregated = aggregated.reset_index(drop=True)
        
        # Calculate consensus score (how many methods agree on top ranking)
        rank_cols = [col for col in aggregated.columns if col.endswith('_rank')]
        aggregated['avg_rank'] = aggregated[rank_cols].mean(axis=1)
        aggregated['rank_std'] = aggregated[rank_cols].std(axis=1)
        
        return aggregated
    
    def get_feature_explanations(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Generate natural language explanations for top features.
        
        Returns:
            List of explanation dictionaries for each top feature
        """
        aggregated = self.get_aggregated_importance()
        top_features = aggregated.head(top_n)
        
        explanations = []
        
        for _, row in top_features.iterrows():
            feature = row['feature']
            explanation = {
                'feature': feature,
                'rank': int(row['final_rank']),
                'score': round(row['weighted_score'], 4),
                'reasons': []
            }
            
            # Add correlation-based explanation
            if 'correlation' in self.results:
                corr_df = self.results['correlation']
                corr_row = corr_df[corr_df['feature'] == feature].iloc[0]
                corr_val = corr_row['correlation']
                if abs(corr_val) > 0.3:
                    direction = "positive" if corr_val > 0 else "negative"
                    explanation['reasons'].append(
                        f"Has {direction} correlation ({corr_val:.3f}) with target"
                    )
                if corr_row.get('significant', False):
                    explanation['reasons'].append("Statistically significant (p < 0.05)")
            
            # Add mutual information explanation
            if 'mutual_info' in self.results:
                mi_df = self.results['mutual_info']
                mi_row = mi_df[mi_df['feature'] == feature].iloc[0]
                mi_val = mi_row.get('mi_score', mi_row['importance'])
                if mi_val > 0.1:
                    explanation['reasons'].append(
                        f"High mutual information ({mi_val:.3f}) indicating non-linear dependencies"
                    )
            
            # Add tree-based explanation
            if 'random_forest' in self.results:
                rf_df = self.results['random_forest']
                rf_row = rf_df[rf_df['feature'] == feature].iloc[0]
                if rf_row['rank'] <= 10:
                    explanation['reasons'].append(
                        f"Ranked #{rf_row['rank']} in Random Forest importance"
                    )
            
            # Add XGBoost explanation
            if 'xgboost' in self.results:
                xgb_df = self.results['xgboost']
                xgb_row = xgb_df[xgb_df['feature'] == feature].iloc[0]
                if xgb_row['rank'] <= 10:
                    explanation['reasons'].append(
                        f"Ranked #{xgb_row['rank']} in XGBoost gradient boosting"
                    )
            
            # Add SHAP explanation
            if 'shap' in self.results:
                shap_df = self.results['shap']
                shap_row = shap_df[shap_df['feature'] == feature].iloc[0]
                if shap_row['rank'] <= 10:
                    explanation['reasons'].append(
                        f"High SHAP impact (mean |SHAP| = {shap_row['mean_shap']:.4f})"
                    )
            
            # Add statistical summary
            if feature in X.columns:
                col_data = X[feature]
                if pd.api.types.is_numeric_dtype(col_data):
                    explanation['stats'] = {
                        'mean': round(col_data.mean(), 4),
                        'std': round(col_data.std(), 4),
                        'min': round(col_data.min(), 4),
                        'max': round(col_data.max(), 4)
                    }
            
            # Generate summary
            if not explanation['reasons']:
                explanation['summary'] = f"Feature contributes to model predictions"
            else:
                explanation['summary'] = "; ".join(explanation['reasons'][:3])
            
            explanations.append(explanation)
        
        return explanations
