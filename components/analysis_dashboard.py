"""
Analysis Dashboard Component
Main dashboard for feature analysis results
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional


def render_analysis_dashboard(
    results: Dict[str, pd.DataFrame],
    aggregated: pd.DataFrame,
    explanations: List[Dict[str, Any]],
    recommendations: Dict[str, Any]
):
    """
    Render the main analysis dashboard.
    
    Args:
        results: Dictionary of method results
        aggregated: Aggregated importance DataFrame
        explanations: List of feature explanations
        recommendations: Feature recommendations dict
    """
    # Summary metrics
    _render_summary_metrics(results, aggregated, recommendations)
    
    st.markdown("---")
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs([
        "🏆 Top Features",
        "📊 Method Comparison",
        "💡 Recommendations"
    ])
    
    with tab1:
        _render_top_features(aggregated, explanations)
    
    with tab2:
        _render_method_comparison(results)
    
    with tab3:
        _render_recommendations(recommendations, explanations)


def _render_summary_metrics(
    results: Dict[str, pd.DataFrame],
    aggregated: pd.DataFrame,
    recommendations: Dict[str, Any]
):
    """Render summary metric cards."""
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "🔬 Methods Used",
            len(results),
            help="Number of importance methods applied"
        )
    
    with col2:
        st.metric(
            "📋 Total Features",
            len(aggregated),
            help="Total number of features analyzed"
        )
    
    with col3:
        n_recommended = recommendations.get('n_recommended', 0)
        reduction = recommendations.get('reduction_pct', 0)
        st.metric(
            "✅ Recommended",
            n_recommended,
            delta=f"-{reduction:.0f}% reduction",
            delta_color="normal"
        )
    
    with col4:
        high_conf = len(recommendations.get('high_confidence', []))
        st.metric(
            "🎯 High Confidence",
            high_conf,
            help="Features with consistent rankings across methods"
        )


def _render_top_features(aggregated: pd.DataFrame, explanations: List[Dict]):
    """Render top features with explanations."""
    
    st.markdown("### 🏆 Top Features by Aggregated Importance")
    
    # Top features table
    display_cols = ['final_rank', 'feature', 'weighted_score', 'avg_rank', 'rank_std']
    available_cols = [c for c in display_cols if c in aggregated.columns]
    
    top_df = aggregated.head(15)[available_cols].copy()
    top_df.columns = ['Rank', 'Feature', 'Score', 'Avg Rank', 'Rank Std'][:len(available_cols)]
    
    # Format numeric columns
    if 'Score' in top_df.columns:
        top_df['Score'] = top_df['Score'].round(4)
    if 'Avg Rank' in top_df.columns:
        top_df['Avg Rank'] = top_df['Avg Rank'].round(1)
    if 'Rank Std' in top_df.columns:
        top_df['Rank Std'] = top_df['Rank Std'].round(2)
    
    st.dataframe(
        top_df,
        use_container_width=True,
        hide_index=True,
        height=400
    )
    
    # Feature explanation cards
    st.markdown("### 📝 Feature Explanations")
    
    n_cols = 2
    for i in range(0, min(10, len(explanations)), n_cols):
        cols = st.columns(n_cols)
        for j, col in enumerate(cols):
            idx = i + j
            if idx < len(explanations):
                exp = explanations[idx]
                with col:
                    _render_feature_card(exp)


def _render_feature_card(explanation: Dict[str, Any]):
    """Render a single feature explanation card."""
    
    rank = explanation.get('rank', '-')
    feature = explanation.get('feature', 'Unknown')
    score = explanation.get('score', 0)
    summary = explanation.get('summary', '')
    reasons = explanation.get('reasons', [])
    
    # Card color based on rank
    if rank <= 3:
        border_color = "#10B981"  # Green
        badge_color = "#059669"
    elif rank <= 7:
        border_color = "#F59E0B"  # Amber
        badge_color = "#D97706"
    else:
        border_color = "#6366F1"  # Indigo
        badge_color = "#4F46E5"
    
    st.markdown(f"""
    <div style="
        border: 1px solid {border_color};
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.08) 0%, rgba(139, 92, 246, 0.08) 100%);
    ">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
            <span style="
                background: {badge_color};
                color: white;
                padding: 3px 10px;
                border-radius: 20px;
                font-size: 0.85em;
                font-weight: bold;
            ">#{rank}</span>
            <span style="color: #9CA3AF; font-size: 0.85em;">Score: {score:.4f}</span>
        </div>
        <h4 style="margin: 5px 0; color: #F3F4F6;">{feature}</h4>
        <p style="color: #9CA3AF; font-size: 0.9em; margin: 8px 0;">{summary}</p>
    </div>
    """, unsafe_allow_html=True)


def _render_method_comparison(results: Dict[str, pd.DataFrame]):
    """Render method-by-method comparison."""
    
    st.markdown("### 📊 Results by Method")
    
    method_names = list(results.keys())
    selected_method = st.selectbox(
        "Select Method",
        options=method_names,
        format_func=lambda x: x.replace('_', ' ').title()
    )
    
    if selected_method and selected_method in results:
        df = results[selected_method]
        
        # Display columns to show
        display_cols = ['rank', 'feature', 'importance']
        extra_cols = [c for c in df.columns if c not in display_cols and c != 'importance']
        display_cols.extend(extra_cols[:3])  # Add up to 3 extra cols
        
        available_cols = [c for c in display_cols if c in df.columns]
        
        st.dataframe(
            df[available_cols].head(20),
            use_container_width=True,
            hide_index=True,
            height=450
        )
        
        # Method stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Features Analyzed", len(df))
        with col2:
            top_feature = df.iloc[0]['feature'] if len(df) > 0 else '-'
            st.metric("Top Feature", top_feature)
        with col3:
            max_imp = df['importance'].max() if 'importance' in df.columns else 0
            st.metric("Max Importance", f"{max_imp:.4f}")


def _render_recommendations(
    recommendations: Dict[str, Any],
    explanations: List[Dict]
):
    """Render feature recommendations section."""
    
    st.markdown("### 💡 Feature Selection Recommendations")
    
    # Rationale
    rationale = recommendations.get('rationale', '')
    if rationale:
        st.info(f"📋 **Summary**: {rationale}")
    
    # Categories
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### ✅ High Confidence")
        high_conf = recommendations.get('high_confidence', [])
        if high_conf:
            for feat in high_conf[:10]:
                st.markdown(f"• `{feat}`")
        else:
            st.caption("No high confidence features")
    
    with col2:
        st.markdown("#### ⚠️ Medium Confidence")
        med_conf = recommendations.get('medium_confidence', [])
        if med_conf:
            for feat in med_conf[:10]:
                st.markdown(f"• `{feat}`")
        else:
            st.caption("No medium confidence features")
    
    with col3:
        st.markdown("#### ❌ Consider Dropping")
        drop_cand = recommendations.get('drop_candidates', [])
        if drop_cand:
            for feat in drop_cand[:10]:
                st.markdown(f"• `{feat}`")
        else:
            st.caption("No features recommended for dropping")
    
    # Export selected features
    st.markdown("---")
    
    recommended_features = recommendations.get('recommended_features', [])
    
    if recommended_features:
        st.markdown("#### 📥 Export Recommendations")
        
        export_df = pd.DataFrame({
            'Rank': range(1, len(recommended_features) + 1),
            'Feature': recommended_features,
            'Category': [
                'High Confidence' if f in recommendations.get('high_confidence', [])
                else 'Medium Confidence' if f in recommendations.get('medium_confidence', [])
                else 'Low Confidence'
                for f in recommended_features
            ]
        })
        
        csv = export_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Feature List (CSV)",
            data=csv,
            file_name="recommended_features.csv",
            mime="text/csv"
        )


def render_preprocessing_summary(preprocessing_info: Dict[str, Any]):
    """Render preprocessing summary."""
    
    with st.expander("🔧 Preprocessing Summary", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Data Shape**")
            st.write(f"- Original: {preprocessing_info.get('original_shape', '-')}")
            st.write(f"- Final: {preprocessing_info.get('final_shape', '-')}")
        
        with col2:
            st.markdown("**Processing Steps**")
            if preprocessing_info.get('encoded_columns'):
                st.write(f"- Encoded: {len(preprocessing_info['encoded_columns'])} columns")
            if preprocessing_info.get('imputed_columns'):
                st.write(f"- Imputed: {len(preprocessing_info['imputed_columns'])} columns")
            if preprocessing_info.get('dropped_columns'):
                st.write(f"- Dropped: {len(preprocessing_info['dropped_columns'])} columns")
        
        # Warnings
        warnings = preprocessing_info.get('warnings', [])
        if warnings:
            st.markdown("**⚠️ Warnings**")
            for warning in warnings:
                st.warning(warning)
