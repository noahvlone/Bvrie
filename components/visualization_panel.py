"""
Visualization Panel Component
Interactive charts and plots for feature analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional

from src.visualizer import Visualizer


def render_visualization_panel(
    visualizer: Visualizer,
    results: Dict[str, pd.DataFrame],
    aggregated: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    recommendations: Dict[str, Any],
    explanations: List[Dict],
    shap_values: Optional[np.ndarray] = None
):
    """
    Render the visualization panel with all charts.
    
    Args:
        visualizer: Visualizer instance
        results: Dictionary of method results
        aggregated: Aggregated importance DataFrame
        X: Feature DataFrame (preprocessed)
        y: Target Series
        recommendations: Recommendations dict
        explanations: Feature explanations
        shap_values: Optional SHAP values array
    """
    st.markdown("## 📊 Visualizations")
    
    # Visualization selector
    viz_options = [
        "📊 Feature Importance",
        "🔥 Correlation Heatmap",
        "📈 Cumulative Importance",
        "🎯 Method Comparison",
        "📉 Feature Distributions",
        "🏆 Rank Comparison"
    ]
    
    if shap_values is not None:
        viz_options.append("🔮 SHAP Analysis")
    
    selected_viz = st.selectbox(
        "Select Visualization",
        options=viz_options,
        index=0
    )
    
    # Render selected visualization
    if selected_viz == "📊 Feature Importance":
        _render_importance_charts(visualizer, results, aggregated)
    
    elif selected_viz == "🔥 Correlation Heatmap":
        _render_correlation_heatmap(visualizer, X, y)
    
    elif selected_viz == "📈 Cumulative Importance":
        _render_cumulative_chart(visualizer, aggregated)
    
    elif selected_viz == "🎯 Method Comparison":
        _render_method_comparison_chart(visualizer, results)
    
    elif selected_viz == "📉 Feature Distributions":
        _render_distribution_charts(visualizer, X, y, aggregated)
    
    elif selected_viz == "🏆 Rank Comparison":
        _render_rank_heatmap(visualizer, aggregated)
    
    elif selected_viz == "🔮 SHAP Analysis" and shap_values is not None:
        _render_shap_charts(visualizer, shap_values, X)


def _render_importance_charts(
    visualizer: Visualizer,
    results: Dict[str, pd.DataFrame],
    aggregated: pd.DataFrame
):
    """Render feature importance bar charts."""
    
    st.markdown("### Feature Importance Scores")
    
    # Method selector
    col1, col2 = st.columns([2, 1])
    
    with col1:
        method_options = ['aggregated'] + list(results.keys())
        selected = st.selectbox(
            "Select Method",
            options=method_options,
            format_func=lambda x: x.replace('_', ' ').title() if x != 'aggregated' else '⭐ Aggregated (All Methods)'
        )
    
    with col2:
        top_n = st.slider("Top N Features", 5, 30, 15)
    
    # Get the appropriate dataframe
    if selected == 'aggregated':
        # Create a compatible dataframe from aggregated
        plot_df = aggregated[['feature', 'weighted_score']].copy()
        plot_df.columns = ['feature', 'importance']
        method_name = "Aggregated"
    else:
        plot_df = results[selected]
        method_name = selected.replace('_', ' ').title()
    
    # Create and display chart
    fig = visualizer.feature_importance_bar(
        plot_df,
        title="Feature Importance",
        top_n=top_n,
        method_name=method_name
    )
    
    st.plotly_chart(fig, use_container_width=True)


def _render_correlation_heatmap(
    visualizer: Visualizer,
    X: pd.DataFrame,
    y: pd.Series
):
    """Render correlation analysis."""
    
    st.markdown("### Correlation Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        corr_method = st.selectbox(
            "Correlation Method",
            options=['pearson', 'spearman', 'kendall'],
            index=0
        )
    
    with col2:
        n_features = st.slider("Max Features", 10, 30, 20)
    
    with col3:
        include_target = st.checkbox("Include Target", value=True)
    
    # Heatmap
    fig = visualizer.correlation_heatmap(
        X,
        y=y if include_target else None,
        top_n=n_features,
        method=corr_method
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Target correlation bar chart
    st.markdown("#### Feature Correlation with Target")
    
    fig2 = visualizer.target_correlation_bar(X, y, top_n=15)
    st.plotly_chart(fig2, use_container_width=True)


def _render_cumulative_chart(
    visualizer: Visualizer,
    aggregated: pd.DataFrame
):
    """Render cumulative importance chart."""
    
    st.markdown("### Cumulative Feature Importance")
    
    st.info(
        "📘 This chart shows how much of the total importance is captured by the top N features. "
        "Use this to determine the optimal number of features to keep."
    )
    
    # Create compatible dataframe
    plot_df = aggregated[['feature', 'weighted_score']].copy()
    plot_df.columns = ['feature', 'importance']
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        threshold = st.slider(
            "Importance Threshold",
            min_value=0.80,
            max_value=0.99,
            value=0.95,
            step=0.01
        )
    
    with col1:
        fig = visualizer.cumulative_importance(plot_df, threshold=threshold)
        st.plotly_chart(fig, use_container_width=True)
    
    # Analysis text
    total = plot_df['importance'].sum()
    if total > 0:
        plot_df_sorted = plot_df.sort_values('importance', ascending=False)
        plot_df_sorted['cumulative'] = plot_df_sorted['importance'].cumsum() / total
        n_threshold = len(plot_df_sorted[plot_df_sorted['cumulative'] <= threshold]) + 1
        
        st.success(
            f"📊 **{n_threshold} features** capture **{threshold*100:.0f}%** of total importance, "
            f"reducing from {len(plot_df)} features ({(1 - n_threshold/len(plot_df))*100:.0f}% reduction)"
        )


def _render_method_comparison_chart(
    visualizer: Visualizer,
    results: Dict[str, pd.DataFrame]
):
    """Render method comparison chart."""
    
    st.markdown("### Method Comparison")
    
    st.info(
        "📘 Compare how different methods rank feature importance. "
        "Features that score high across multiple methods are more reliable."
    )
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        n_features = st.slider("Features to Compare", 5, 20, 10)
    
    with col1:
        fig = visualizer.importance_comparison(results, top_n=n_features)
        st.plotly_chart(fig, use_container_width=True)


def _render_distribution_charts(
    visualizer: Visualizer,
    X: pd.DataFrame,
    y: pd.Series,
    aggregated: pd.DataFrame
):
    """Render feature distribution charts."""
    
    st.markdown("### Feature Distributions")
    
    # Feature selector
    top_features = aggregated.nsmallest(20, 'final_rank')['feature'].tolist()
    available_features = [f for f in top_features if f in X.columns]
    
    if not available_features:
        st.warning("No features available for distribution analysis.")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_feature = st.selectbox(
            "Select Feature",
            options=available_features
        )
    
    with col2:
        problem_type = st.radio(
            "Problem Type",
            options=['classification', 'regression'],
            index=0,
            horizontal=True
        )
    
    if selected_feature:
        fig = visualizer.feature_distribution(
            X, y, selected_feature, problem_type
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Grid view
    with st.expander("📊 View Multiple Distributions"):
        n_features_grid = st.slider("Number of Features", 3, 9, 6)
        
        grid_features = available_features[:n_features_grid]
        fig_grid = visualizer.feature_distributions_grid(
            X, y, grid_features, problem_type, cols=3
        )
        st.plotly_chart(fig_grid, use_container_width=True)


def _render_rank_heatmap(
    visualizer: Visualizer,
    aggregated: pd.DataFrame
):
    """Render rank comparison heatmap."""
    
    st.markdown("### Feature Rank Consistency")
    
    st.info(
        "📘 This heatmap shows how consistently features are ranked across different methods. "
        "Lower rank = higher importance. Consistent rankings (similar colors across methods) indicate reliable features."
    )
    
    n_features = st.slider("Number of Features", 10, 25, 15)
    
    fig = visualizer.rank_comparison_heatmap(aggregated, top_n=n_features)
    st.plotly_chart(fig, use_container_width=True)


def _render_shap_charts(
    visualizer: Visualizer,
    shap_values: np.ndarray,
    X: pd.DataFrame
):
    """Render SHAP analysis charts."""
    
    st.markdown("### SHAP Analysis")
    
    st.info(
        "📘 SHAP (SHapley Additive exPlanations) values show how each feature contributes to individual predictions. "
        "Positive values push predictions higher, negative values push them lower."
    )
    
    tab1, tab2 = st.tabs(["Summary Plot", "Individual Explanation"])
    
    with tab1:
        n_features = st.slider("Top N Features", 5, 20, 12)
        
        fig = visualizer.shap_summary_plotly(shap_values, X, top_n=n_features)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        sample_idx = st.slider(
            "Sample Index",
            0,
            min(len(X) - 1, 100),
            0,
            help="Select a sample to explain"
        )
        
        fig = visualizer.create_shap_waterfall(shap_values, X, sample_idx)
        st.plotly_chart(fig, use_container_width=True)


def render_recommendation_summary_chart(
    visualizer: Visualizer,
    recommendations: Dict[str, Any],
    explanations: List[Dict]
):
    """Render recommendation summary visualization."""
    
    fig = visualizer.recommendation_summary(recommendations, explanations)
    st.plotly_chart(fig, use_container_width=True)
