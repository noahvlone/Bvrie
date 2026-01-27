"""
Visualizer Module
Creates rich visualizations for feature importance analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')

# Optional imports
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


class Visualizer:
    """
    Create visualizations for feature importance analysis.
    """
    
    # Color schemes
    COLORS = {
        'primary': '#6366F1',       # Indigo
        'secondary': '#8B5CF6',     # Violet
        'accent': '#EC4899',        # Pink
        'success': '#10B981',       # Emerald
        'warning': '#F59E0B',       # Amber
        'danger': '#EF4444',        # Red
        'info': '#3B82F6',          # Blue
        'gradient': ['#6366F1', '#8B5CF6', '#A855F7', '#C084FC', '#D8B4FE'],
        'heatmap': 'RdBu_r',
        'sequential': 'Viridis'
    }
    
    TEMPLATE = 'plotly_dark'
    
    def __init__(self, theme: str = 'dark'):
        """
        Initialize visualizer.
        
        Args:
            theme: 'dark' or 'light'
        """
        self.theme = theme
        self.template = 'plotly_dark' if theme == 'dark' else 'plotly_white'
        self.bg_color = '#0E1117' if theme == 'dark' else '#FFFFFF'
        self.text_color = '#FAFAFA' if theme == 'dark' else '#1F2937'
        self.grid_color = '#1F2937' if theme == 'dark' else '#E5E7EB'
    
    def feature_importance_bar(
        self,
        importance_df: pd.DataFrame,
        title: str = "Feature Importance",
        top_n: int = 20,
        method_name: str = "",
        show_values: bool = True
    ) -> go.Figure:
        """
        Create a horizontal bar chart for feature importance.
        
        Args:
            importance_df: DataFrame with 'feature' and 'importance' columns
            title: Chart title
            top_n: Number of top features to show
            method_name: Name of the importance method used
            show_values: Whether to show importance values on bars
        """
        # Get top N features
        df = importance_df.nlargest(top_n, 'importance').sort_values('importance')
        
        # Create color gradient based on importance
        colors = [self.COLORS['primary'] if i >= len(df) - 5 else self.COLORS['secondary'] 
                  for i in range(len(df))]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=df['importance'],
            y=df['feature'],
            orientation='h',
            marker=dict(
                color=df['importance'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Importance")
            ),
            text=df['importance'].round(4) if show_values else None,
            textposition='outside',
            textfont=dict(size=10),
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>"
        ))
        
        full_title = f"{title} ({method_name})" if method_name else title
        
        fig.update_layout(
            title=dict(text=full_title, font=dict(size=18)),
            xaxis_title="Importance Score",
            yaxis_title="Feature",
            template=self.template,
            height=max(400, top_n * 25),
            margin=dict(l=20, r=20, t=60, b=40),
            paper_bgcolor=self.bg_color,
            plot_bgcolor=self.bg_color,
            font=dict(color=self.text_color)
        )
        
        return fig
    
    def importance_comparison(
        self,
        results: Dict[str, pd.DataFrame],
        top_n: int = 15
    ) -> go.Figure:
        """
        Create a grouped bar chart comparing importance across methods.
        
        Args:
            results: Dictionary of method names to importance DataFrames
            top_n: Number of features to compare
        """
        # Get union of top features from all methods
        all_top_features = set()
        for df in results.values():
            all_top_features.update(df.nlargest(top_n, 'importance')['feature'].tolist())
        
        all_top_features = list(all_top_features)[:top_n]
        
        fig = go.Figure()
        
        colors = ['#6366F1', '#8B5CF6', '#EC4899', '#10B981', '#F59E0B', '#3B82F6']
        
        for i, (method, df) in enumerate(results.items()):
            # Normalize importance to 0-1 for comparison
            max_imp = df['importance'].max()
            df_norm = df.copy()
            df_norm['normalized'] = df_norm['importance'] / max_imp if max_imp > 0 else 0
            
            # Get values for top features
            feature_values = []
            for feat in all_top_features:
                val = df_norm[df_norm['feature'] == feat]['normalized'].values
                feature_values.append(val[0] if len(val) > 0 else 0)
            
            fig.add_trace(go.Bar(
                name=method.replace('_', ' ').title(),
                x=all_top_features,
                y=feature_values,
                marker_color=colors[i % len(colors)],
                hovertemplate=f"<b>%{{x}}</b><br>{method}: %{{y:.3f}}<extra></extra>"
            ))
        
        fig.update_layout(
            title="Feature Importance Comparison Across Methods",
            xaxis_title="Feature",
            yaxis_title="Normalized Importance",
            barmode='group',
            template=self.template,
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            paper_bgcolor=self.bg_color,
            plot_bgcolor=self.bg_color,
            font=dict(color=self.text_color)
        )
        
        fig.update_xaxes(tickangle=45)
        
        return fig
    
    def correlation_heatmap(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        top_n: int = 20,
        method: str = 'pearson'
    ) -> go.Figure:
        """
        Create a correlation heatmap.
        
        Args:
            X: Feature DataFrame
            y: Optional target series to include
            top_n: Number of features to include
            method: Correlation method ('pearson', 'spearman', 'kendall')
        """
        # Select numeric columns only
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) > top_n:
            # Select most variant features
            variances = X[numeric_cols].var().sort_values(ascending=False)
            numeric_cols = variances.head(top_n).index.tolist()
        
        df = X[numeric_cols].copy()
        
        if y is not None:
            df['Target'] = y.values
        
        # Calculate correlation matrix
        corr_matrix = df.corr(method=method)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu_r',
            zmid=0,
            text=corr_matrix.values.round(2),
            texttemplate="%{text}",
            textfont=dict(size=9),
            hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>",
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title=f"Feature Correlation Heatmap ({method.title()})",
            template=self.template,
            height=max(500, len(corr_matrix) * 25),
            width=max(600, len(corr_matrix) * 25),
            paper_bgcolor=self.bg_color,
            plot_bgcolor=self.bg_color,
            font=dict(color=self.text_color)
        )
        
        return fig
    
    def target_correlation_bar(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        top_n: int = 20
    ) -> go.Figure:
        """
        Create a bar chart showing correlations with target.
        """
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        correlations = []
        for col in numeric_cols:
            try:
                corr = X[col].corr(y)
                correlations.append({'feature': col, 'correlation': corr})
            except Exception:
                pass
        
        if not correlations:
            return self._empty_figure("No numeric features for correlation analysis")
        
        corr_df = pd.DataFrame(correlations)
        corr_df['abs_correlation'] = corr_df['correlation'].abs()
        corr_df = corr_df.nlargest(top_n, 'abs_correlation').sort_values('correlation')
        
        colors = ['#EF4444' if c < 0 else '#10B981' for c in corr_df['correlation']]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=corr_df['correlation'],
            y=corr_df['feature'],
            orientation='h',
            marker_color=colors,
            text=corr_df['correlation'].round(3),
            textposition='outside',
            hovertemplate="<b>%{y}</b><br>Correlation: %{x:.3f}<extra></extra>"
        ))
        
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        
        fig.update_layout(
            title="Feature Correlation with Target",
            xaxis_title="Correlation Coefficient",
            yaxis_title="Feature",
            template=self.template,
            height=max(400, top_n * 25),
            paper_bgcolor=self.bg_color,
            plot_bgcolor=self.bg_color,
            font=dict(color=self.text_color)
        )
        
        return fig
    
    def feature_distribution(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature: str,
        problem_type: str = 'classification'
    ) -> go.Figure:
        """
        Create distribution plot for a feature, colored by target.
        """
        if feature not in X.columns:
            return self._empty_figure(f"Feature '{feature}' not found")
        
        df = pd.DataFrame({'feature': X[feature], 'target': y})
        
        if problem_type == 'classification':
            fig = px.histogram(
                df,
                x='feature',
                color='target',
                barmode='overlay',
                opacity=0.7,
                title=f"Distribution of '{feature}' by Target Class",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
        else:
            fig = px.scatter(
                df,
                x='feature',
                y='target',
                trendline='ols',
                title=f"'{feature}' vs Target",
                color_discrete_sequence=[self.COLORS['primary']]
            )
        
        fig.update_layout(
            template=self.template,
            paper_bgcolor=self.bg_color,
            plot_bgcolor=self.bg_color,
            font=dict(color=self.text_color)
        )
        
        return fig
    
    def feature_distributions_grid(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        features: List[str],
        problem_type: str = 'classification',
        cols: int = 3
    ) -> go.Figure:
        """
        Create a grid of distribution plots for multiple features.
        """
        n_features = len(features)
        rows = (n_features + cols - 1) // cols
        
        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=[f[:20] + '...' if len(f) > 20 else f for f in features]
        )
        
        colors = px.colors.qualitative.Set2
        unique_targets = y.unique()
        
        for i, feature in enumerate(features):
            if feature not in X.columns:
                continue
            
            row = i // cols + 1
            col = i % cols + 1
            
            for j, target_val in enumerate(unique_targets):
                mask = y == target_val
                feature_data = X.loc[mask, feature].dropna()
                
                fig.add_trace(
                    go.Histogram(
                        x=feature_data,
                        name=f"Class {target_val}",
                        opacity=0.7,
                        marker_color=colors[j % len(colors)],
                        showlegend=(i == 0)
                    ),
                    row=row,
                    col=col
                )
        
        fig.update_layout(
            title="Feature Distributions by Target Class",
            barmode='overlay',
            height=300 * rows,
            template=self.template,
            paper_bgcolor=self.bg_color,
            plot_bgcolor=self.bg_color,
            font=dict(color=self.text_color),
            showlegend=True
        )
        
        return fig
    
    def cumulative_importance(
        self,
        importance_df: pd.DataFrame,
        threshold: float = 0.95
    ) -> go.Figure:
        """
        Create cumulative importance plot with threshold line.
        """
        df = importance_df.sort_values('importance', ascending=False).copy()
        
        total_importance = df['importance'].sum()
        if total_importance > 0:
            df['cumulative'] = df['importance'].cumsum() / total_importance
        else:
            df['cumulative'] = 0
        
        df['feature_num'] = range(1, len(df) + 1)
        
        # Find threshold point
        threshold_idx = len(df[df['cumulative'] <= threshold])
        
        fig = go.Figure()
        
        # Individual importance bars
        fig.add_trace(go.Bar(
            x=df['feature_num'],
            y=df['importance'],
            name='Individual Importance',
            marker_color=self.COLORS['primary'],
            opacity=0.7,
            yaxis='y'
        ))
        
        # Cumulative line
        fig.add_trace(go.Scatter(
            x=df['feature_num'],
            y=df['cumulative'],
            name='Cumulative Importance',
            mode='lines+markers',
            line=dict(color=self.COLORS['accent'], width=2),
            marker=dict(size=6),
            yaxis='y2'
        ))
        
        # Threshold line
        fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color=self.COLORS['warning'],
            annotation_text=f"{threshold*100}% threshold",
            yref='y2'
        )
        
        # Optimal feature count line
        if threshold_idx > 0:
            fig.add_vline(
                x=threshold_idx,
                line_dash="dot",
                line_color=self.COLORS['success'],
                annotation_text=f"Optimal: {threshold_idx} features"
            )
        
        fig.update_layout(
            title="Cumulative Feature Importance",
            xaxis_title="Number of Features",
            yaxis=dict(title="Individual Importance", side='left'),
            yaxis2=dict(title="Cumulative Importance", side='right', overlaying='y', range=[0, 1.05]),
            template=self.template,
            height=450,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            paper_bgcolor=self.bg_color,
            plot_bgcolor=self.bg_color,
            font=dict(color=self.text_color)
        )
        
        return fig
    
    def rank_comparison_heatmap(
        self,
        aggregated_df: pd.DataFrame,
        top_n: int = 15
    ) -> go.Figure:
        """
        Create a heatmap showing feature ranks across methods.
        """
        rank_cols = [col for col in aggregated_df.columns if col.endswith('_rank')]
        
        if not rank_cols:
            return self._empty_figure("No rank data available")
        
        df = aggregated_df.nsmallest(top_n, 'final_rank')[['feature'] + rank_cols]
        
        # Rename columns for display
        display_cols = [col.replace('_rank', '').replace('_', ' ').title() for col in rank_cols]
        
        z_data = df[rank_cols].values
        
        fig = go.Figure(data=go.Heatmap(
            z=z_data,
            x=display_cols,
            y=df['feature'],
            colorscale='YlOrRd_r',
            text=z_data.astype(int),
            texttemplate="%{text}",
            textfont=dict(size=10),
            hovertemplate="<b>%{y}</b><br>%{x}: Rank %{z}<extra></extra>",
            colorbar=dict(title="Rank")
        ))
        
        fig.update_layout(
            title="Feature Rank Comparison Across Methods",
            xaxis_title="Method",
            yaxis_title="Feature",
            template=self.template,
            height=max(400, top_n * 30),
            paper_bgcolor=self.bg_color,
            plot_bgcolor=self.bg_color,
            font=dict(color=self.text_color)
        )
        
        return fig
    
    def shap_summary_plotly(
        self,
        shap_values: np.ndarray,
        X: pd.DataFrame,
        top_n: int = 15
    ) -> go.Figure:
        """
        Create a SHAP-style summary plot using Plotly.
        """
        feature_importance = np.abs(shap_values).mean(axis=0)
        top_indices = np.argsort(feature_importance)[-top_n:][::-1]
        
        fig = go.Figure()
        
        for i, idx in enumerate(top_indices[::-1]):
            feature = X.columns[idx]
            shap_vals = shap_values[:, idx]
            feature_vals = X.iloc[:, idx].values
            
            # Normalize feature values for color
            fv_norm = (feature_vals - np.nanmin(feature_vals)) / (np.nanmax(feature_vals) - np.nanmin(feature_vals) + 1e-8)
            
            # Add jitter
            jitter = np.random.normal(0, 0.1, len(shap_vals))
            
            fig.add_trace(go.Scatter(
                x=shap_vals,
                y=[i + j for j in jitter],
                mode='markers',
                marker=dict(
                    size=5,
                    color=fv_norm,
                    colorscale='RdBu',
                    opacity=0.7
                ),
                name=feature,
                showlegend=False,
                hovertemplate=f"<b>{feature}</b><br>SHAP: %{{x:.3f}}<extra></extra>"
            ))
        
        # Add feature labels
        fig.update_layout(
            title="SHAP Summary Plot",
            xaxis_title="SHAP Value (impact on model output)",
            yaxis=dict(
                tickmode='array',
                tickvals=list(range(len(top_indices))),
                ticktext=[X.columns[idx] for idx in top_indices[::-1]]
            ),
            template=self.template,
            height=max(400, top_n * 35),
            paper_bgcolor=self.bg_color,
            plot_bgcolor=self.bg_color,
            font=dict(color=self.text_color)
        )
        
        fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        return fig
    
    def recommendation_summary(
        self,
        recommendations: Dict[str, Any],
        explanations: List[Dict]
    ) -> go.Figure:
        """
        Create a visual summary of feature recommendations.
        """
        n_features = len(explanations)
        
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "domain"}, {"type": "xy"}]],
            subplot_titles=["Feature Categories", "Top Features by Score"]
        )
        
        # Pie chart for confidence levels
        labels = ['High Confidence', 'Medium Confidence', 'Low Confidence']
        values = [
            len(recommendations.get('high_confidence', [])),
            len(recommendations.get('medium_confidence', [])),
            len(recommendations.get('low_confidence', []))
        ]
        colors = [self.COLORS['success'], self.COLORS['warning'], self.COLORS['danger']]
        
        fig.add_trace(go.Pie(
            labels=labels,
            values=values,
            marker_colors=colors,
            hole=0.4,
            textinfo='value+percent',
            hovertemplate="<b>%{label}</b><br>Count: %{value}<extra></extra>"
        ), row=1, col=1)
        
        # Bar chart for top features
        features = [exp['feature'][:15] for exp in explanations[:10]]
        scores = [exp['score'] for exp in explanations[:10]]
        
        fig.add_trace(go.Bar(
            x=scores,
            y=features,
            orientation='h',
            marker_color=self.COLORS['primary'],
            text=[f"{s:.3f}" for s in scores],
            textposition='outside',
            hovertemplate="<b>%{y}</b><br>Score: %{x:.4f}<extra></extra>"
        ), row=1, col=2)
        
        fig.update_layout(
            title=f"Feature Recommendation Summary ({recommendations.get('n_recommended', 0)} of {recommendations.get('n_total', 0)} features)",
            template=self.template,
            height=450,
            showlegend=False,
            paper_bgcolor=self.bg_color,
            plot_bgcolor=self.bg_color,
            font=dict(color=self.text_color)
        )
        
        return fig
    
    def _empty_figure(self, message: str) -> go.Figure:
        """Create an empty figure with a message."""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color=self.text_color)
        )
        fig.update_layout(
            template=self.template,
            paper_bgcolor=self.bg_color,
            plot_bgcolor=self.bg_color
        )
        return fig
    
    def create_shap_waterfall(
        self,
        shap_values: np.ndarray,
        X: pd.DataFrame,
        sample_idx: int = 0
    ) -> go.Figure:
        """
        Create a waterfall plot for a single prediction explanation.
        """
        sample_shap = shap_values[sample_idx]
        feature_names = X.columns.tolist()
        
        # Sort by absolute SHAP value
        sorted_idx = np.argsort(np.abs(sample_shap))[::-1][:15]
        
        values = sample_shap[sorted_idx]
        names = [feature_names[i] for i in sorted_idx]
        
        fig = go.Figure(go.Waterfall(
            orientation="h",
            y=names[::-1],
            x=values[::-1],
            connector={"line": {"color": "rgba(63, 63, 63, 0.5)"}},
            decreasing={"marker": {"color": self.COLORS['danger']}},
            increasing={"marker": {"color": self.COLORS['success']}},
            totals={"marker": {"color": self.COLORS['info']}}
        ))
        
        fig.update_layout(
            title=f"SHAP Explanation for Sample {sample_idx}",
            xaxis_title="SHAP Value",
            template=self.template,
            height=450,
            paper_bgcolor=self.bg_color,
            plot_bgcolor=self.bg_color,
            font=dict(color=self.text_color)
        )
        
        return fig
