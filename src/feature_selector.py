"""
Feature Selector Module
Provides algorithms for selecting optimal feature subsets
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from sklearn.feature_selection import (
    SelectKBest, 
    RFE, 
    VarianceThreshold,
    f_classif,
    f_regression,
    chi2
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import warnings

warnings.filterwarnings('ignore')


class FeatureSelector:
    """
    Feature selection algorithms and utilities.
    """
    
    def __init__(self, problem_type: str = 'classification', random_state: int = 42):
        """
        Initialize the feature selector.
        
        Args:
            problem_type: 'classification' or 'regression'
            random_state: Random seed for reproducibility
        """
        self.problem_type = problem_type
        self.random_state = random_state
        self.selected_features: List[str] = []
        self.selection_history: List[Dict] = []
    
    def select_top_k(
        self,
        importance_df: pd.DataFrame,
        k: int = 10,
        min_importance: float = 0.01
    ) -> List[str]:
        """
        Select top K features based on importance scores.
        
        Args:
            importance_df: DataFrame with 'feature' and 'importance' columns
            k: Number of features to select
            min_importance: Minimum importance threshold
            
        Returns:
            List of selected feature names
        """
        # Filter by minimum importance
        filtered_df = importance_df[importance_df['importance'] >= min_importance]
        
        # Select top K
        top_k = filtered_df.nsmallest(k, 'rank') if 'rank' in filtered_df.columns else filtered_df.nlargest(k, 'importance')
        
        self.selected_features = top_k['feature'].tolist()
        
        self.selection_history.append({
            'method': 'top_k',
            'k': k,
            'min_importance': min_importance,
            'n_selected': len(self.selected_features)
        })
        
        return self.selected_features
    
    def select_by_threshold(
        self,
        importance_df: pd.DataFrame,
        threshold: float = 0.05
    ) -> List[str]:
        """
        Select features above an importance threshold.
        
        Args:
            importance_df: DataFrame with 'feature' and 'importance' columns
            threshold: Minimum importance value (as fraction of max)
            
        Returns:
            List of selected feature names
        """
        max_importance = importance_df['importance'].max()
        threshold_value = max_importance * threshold
        
        selected = importance_df[importance_df['importance'] >= threshold_value]
        self.selected_features = selected['feature'].tolist()
        
        self.selection_history.append({
            'method': 'threshold',
            'threshold': threshold,
            'threshold_value': threshold_value,
            'n_selected': len(self.selected_features)
        })
        
        return self.selected_features
    
    def select_by_cumulative_importance(
        self,
        importance_df: pd.DataFrame,
        cumulative_threshold: float = 0.95
    ) -> List[str]:
        """
        Select features that cumulatively account for a certain percentage of total importance.
        
        Args:
            importance_df: DataFrame with 'feature' and 'importance' columns
            cumulative_threshold: Cumulative importance threshold (0-1)
            
        Returns:
            List of selected feature names
        """
        # Sort by importance
        sorted_df = importance_df.sort_values('importance', ascending=False).copy()
        
        # Calculate cumulative importance
        total_importance = sorted_df['importance'].sum()
        if total_importance == 0:
            self.selected_features = sorted_df['feature'].tolist()[:5]
            return self.selected_features
        
        sorted_df['cumulative'] = sorted_df['importance'].cumsum() / total_importance
        
        # Select features up to threshold
        selected = sorted_df[sorted_df['cumulative'] <= cumulative_threshold]
        
        # Ensure at least one feature is selected
        if len(selected) == 0:
            selected = sorted_df.head(1)
        
        self.selected_features = selected['feature'].tolist()
        
        self.selection_history.append({
            'method': 'cumulative_importance',
            'threshold': cumulative_threshold,
            'n_selected': len(self.selected_features)
        })
        
        return self.selected_features
    
    def variance_threshold_selection(
        self,
        X: pd.DataFrame,
        threshold: float = 0.01
    ) -> List[str]:
        """
        Remove low-variance features.
        
        Args:
            X: Feature DataFrame
            threshold: Variance threshold
            
        Returns:
            List of selected feature names
        """
        selector = VarianceThreshold(threshold=threshold)
        
        # Only apply to numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X_numeric = X[numeric_cols].fillna(0)
        
        try:
            selector.fit(X_numeric)
            selected_mask = selector.get_support()
            self.selected_features = [
                col for col, selected in zip(numeric_cols, selected_mask) if selected
            ]
        except Exception:
            self.selected_features = numeric_cols
        
        self.selection_history.append({
            'method': 'variance_threshold',
            'threshold': threshold,
            'n_original': len(numeric_cols),
            'n_selected': len(self.selected_features)
        })
        
        return self.selected_features
    
    def rfe_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_features: int = 10,
        step: int = 1
    ) -> Tuple[List[str], Dict[str, int]]:
        """
        Recursive Feature Elimination.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            n_features: Number of features to select
            step: Number of features to remove at each iteration
            
        Returns:
            Tuple of (selected feature names, feature rankings dict)
        """
        X_filled = X.fillna(0)
        
        if self.problem_type == 'classification':
            estimator = RandomForestClassifier(
                n_estimators=50,
                random_state=self.random_state,
                max_depth=6,
                n_jobs=-1
            )
        else:
            estimator = RandomForestRegressor(
                n_estimators=50,
                random_state=self.random_state,
                max_depth=6,
                n_jobs=-1
            )
        
        rfe = RFE(
            estimator=estimator,
            n_features_to_select=min(n_features, len(X.columns)),
            step=step
        )
        
        rfe.fit(X_filled, y)
        
        self.selected_features = X.columns[rfe.support_].tolist()
        rankings = dict(zip(X.columns, rfe.ranking_))
        
        self.selection_history.append({
            'method': 'rfe',
            'n_features': n_features,
            'n_selected': len(self.selected_features)
        })
        
        return self.selected_features, rankings
    
    def univariate_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        k: int = 10
    ) -> Tuple[List[str], pd.DataFrame]:
        """
        Univariate statistical feature selection.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            k: Number of features to select
            
        Returns:
            Tuple of (selected feature names, scores DataFrame)
        """
        X_filled = X.fillna(0)
        
        # Choose scoring function
        if self.problem_type == 'classification':
            score_func = f_classif
        else:
            score_func = f_regression
        
        selector = SelectKBest(score_func=score_func, k=min(k, len(X.columns)))
        selector.fit(X_filled, y)
        
        scores_df = pd.DataFrame({
            'feature': X.columns,
            'score': selector.scores_,
            'p_value': selector.pvalues_ if hasattr(selector, 'pvalues_') else [None] * len(X.columns)
        })
        scores_df = scores_df.sort_values('score', ascending=False)
        
        self.selected_features = X.columns[selector.get_support()].tolist()
        
        self.selection_history.append({
            'method': 'univariate',
            'k': k,
            'n_selected': len(self.selected_features)
        })
        
        return self.selected_features, scores_df
    
    def get_feature_recommendations(
        self,
        aggregated_importance: pd.DataFrame,
        X: pd.DataFrame,
        y: pd.Series,
        target_n_features: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate intelligent feature recommendations.
        
        Args:
            aggregated_importance: Aggregated importance DataFrame from FeatureAnalyzer
            X: Feature DataFrame
            y: Target Series
            target_n_features: Target number of features (if None, automatically determined)
            
        Returns:
            Dictionary with recommendations and rationale
        """
        n_total = len(X.columns)
        
        # Determine optimal number of features
        if target_n_features is None:
            # Use elbow method on cumulative importance
            sorted_imp = aggregated_importance.sort_values('weighted_score', ascending=False)
            sorted_imp['cumulative'] = sorted_imp['weighted_score'].cumsum()
            total = sorted_imp['weighted_score'].sum()
            
            if total > 0:
                sorted_imp['cumulative_pct'] = sorted_imp['cumulative'] / total
                
                # Find elbow (where 95% of importance is covered)
                optimal_n = len(sorted_imp[sorted_imp['cumulative_pct'] <= 0.95]) + 1
                optimal_n = max(5, min(optimal_n, n_total))
            else:
                optimal_n = min(10, n_total)
        else:
            optimal_n = target_n_features
        
        # Get recommended features
        recommended = aggregated_importance.nsmallest(optimal_n, 'final_rank')
        
        # Categorize features
        high_confidence = recommended[recommended['rank_std'] < 3]['feature'].tolist()
        medium_confidence = recommended[
            (recommended['rank_std'] >= 3) & (recommended['rank_std'] < 6)
        ]['feature'].tolist()
        low_confidence = recommended[recommended['rank_std'] >= 6]['feature'].tolist()
        
        # Features to consider dropping
        drop_candidates = aggregated_importance[
            aggregated_importance['weighted_score'] < 0.1
        ]['feature'].tolist()
        
        recommendations = {
            'recommended_features': recommended['feature'].tolist(),
            'n_recommended': len(recommended),
            'n_total': n_total,
            'reduction_pct': round((1 - len(recommended) / n_total) * 100, 1),
            'high_confidence': high_confidence,
            'medium_confidence': medium_confidence,
            'low_confidence': low_confidence,
            'drop_candidates': drop_candidates,
            'rationale': self._generate_rationale(
                recommended, high_confidence, medium_confidence, n_total
            )
        }
        
        return recommendations
    
    def _generate_rationale(
        self,
        recommended: pd.DataFrame,
        high_confidence: List[str],
        medium_confidence: List[str],
        n_total: int
    ) -> str:
        """Generate human-readable rationale for recommendations."""
        
        n_rec = len(recommended)
        
        rationale_parts = [
            f"Recommending {n_rec} out of {n_total} features ({round(n_rec/n_total*100)}% retention).",
        ]
        
        if high_confidence:
            rationale_parts.append(
                f"{len(high_confidence)} features have high confidence (consistent ranking across methods)."
            )
        
        if medium_confidence:
            rationale_parts.append(
                f"{len(medium_confidence)} features have medium confidence (some variation in rankings)."
            )
        
        top_feature = recommended.iloc[0]['feature'] if len(recommended) > 0 else None
        if top_feature:
            rationale_parts.append(
                f"Top feature: '{top_feature}' with highest aggregated importance score."
            )
        
        return " ".join(rationale_parts)
    
    def export_selected_features(
        self,
        X: pd.DataFrame,
        features: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Export dataset with only selected features.
        
        Args:
            X: Original feature DataFrame
            features: List of features to include (uses self.selected_features if None)
            
        Returns:
            DataFrame with only selected features
        """
        if features is None:
            features = self.selected_features
        
        if not features:
            raise ValueError("No features selected. Run a selection method first.")
        
        # Filter to features that exist in X
        available_features = [f for f in features if f in X.columns]
        
        return X[available_features].copy()
