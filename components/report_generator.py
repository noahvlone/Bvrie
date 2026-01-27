"""
Report Generator Component
Export analysis results and generate reports
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import io


def render_export_section(
    df_original: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    results: Dict[str, pd.DataFrame],
    aggregated: pd.DataFrame,
    recommendations: Dict[str, Any],
    explanations: List[Dict],
    preprocessing_info: Dict[str, Any]
):
    """
    Render the export and report generation section.
    
    Args:
        df_original: Original uploaded DataFrame
        X: Preprocessed features DataFrame
        y: Target Series
        results: Dictionary of method results
        aggregated: Aggregated importance DataFrame
        recommendations: Feature recommendations
        explanations: Feature explanations
        preprocessing_info: Preprocessing information
    """
    st.markdown("## 📥 Export & Reports")
    
    tab1, tab2, tab3 = st.tabs(["📊 Export Data", "📄 Generate Report", "💾 Save Analysis"])
    
    with tab1:
        _render_data_export(
            df_original, X, y, results, aggregated, recommendations
        )
    
    with tab2:
        _render_report_generation(
            results, aggregated, recommendations, explanations, preprocessing_info
        )
    
    with tab3:
        _render_save_analysis(
            results, aggregated, recommendations, preprocessing_info
        )


def _render_data_export(
    df_original: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    results: Dict[str, pd.DataFrame],
    aggregated: pd.DataFrame,
    recommendations: Dict[str, Any]
):
    """Render data export options."""
    
    st.markdown("### Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📋 Feature Importance")
        
        # Aggregated importance
        csv_agg = aggregated.to_csv(index=False)
        st.download_button(
            label="📥 Aggregated Importance (CSV)",
            data=csv_agg,
            file_name="aggregated_feature_importance.csv",
            mime="text/csv"
        )
        
        # Individual method results
        with st.expander("Download by Method"):
            for method, result_df in results.items():
                csv_method = result_df.to_csv(index=False)
                st.download_button(
                    label=f"📥 {method.replace('_', ' ').title()} Results",
                    data=csv_method,
                    file_name=f"{method}_importance.csv",
                    mime="text/csv",
                    key=f"dl_{method}"
                )
    
    with col2:
        st.markdown("#### 📊 Dataset Export")
        
        # Recommended features only
        recommended = recommendations.get('recommended_features', [])
        if recommended:
            available = [f for f in recommended if f in X.columns]
            if available:
                reduced_df = X[available].copy()
                reduced_df['target'] = y.values
                
                csv_reduced = reduced_df.to_csv(index=False)
                st.download_button(
                    label="📥 Reduced Dataset (Recommended Features)",
                    data=csv_reduced,
                    file_name="reduced_dataset.csv",
                    mime="text/csv"
                )
                
                st.caption(f"Contains {len(available)} features + target")
        
        # Full preprocessed dataset
        full_df = X.copy()
        full_df['target'] = y.values
        csv_full = full_df.to_csv(index=False)
        st.download_button(
            label="📥 Full Preprocessed Dataset",
            data=csv_full,
            file_name="preprocessed_dataset.csv",
            mime="text/csv"
        )


def _render_report_generation(
    results: Dict[str, pd.DataFrame],
    aggregated: pd.DataFrame,
    recommendations: Dict[str, Any],
    explanations: List[Dict],
    preprocessing_info: Dict[str, Any]
):
    """Generate and export analysis report."""
    
    st.markdown("### Generate Analysis Report")
    
    report_format = st.radio(
        "Report Format",
        options=['HTML', 'Markdown', 'Text'],
        horizontal=True
    )
    
    include_options = st.multiselect(
        "Include in Report",
        options=[
            'Summary Statistics',
            'Preprocessing Details',
            'Top Features Table',
            'Feature Explanations',
            'Method Comparison',
            'Recommendations'
        ],
        default=[
            'Summary Statistics',
            'Top Features Table',
            'Feature Explanations',
            'Recommendations'
        ]
    )
    
    if st.button("📄 Generate Report", type="primary"):
        report_content = _generate_report(
            results,
            aggregated,
            recommendations,
            explanations,
            preprocessing_info,
            include_options,
            report_format
        )
        
        if report_format == 'HTML':
            ext = 'html'
            mime = 'text/html'
        elif report_format == 'Markdown':
            ext = 'md'
            mime = 'text/markdown'
        else:
            ext = 'txt'
            mime = 'text/plain'
        
        st.download_button(
            label=f"📥 Download Report (.{ext})",
            data=report_content,
            file_name=f"feature_analysis_report.{ext}",
            mime=mime
        )
        
        # Preview
        with st.expander("📖 Report Preview"):
            if report_format == 'HTML':
                st.markdown(report_content, unsafe_allow_html=True)
            else:
                st.text(report_content)


def _generate_report(
    results: Dict[str, pd.DataFrame],
    aggregated: pd.DataFrame,
    recommendations: Dict[str, Any],
    explanations: List[Dict],
    preprocessing_info: Dict[str, Any],
    include_sections: List[str],
    format: str
) -> str:
    """Generate report content."""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if format == 'HTML':
        return _generate_html_report(
            results, aggregated, recommendations, explanations,
            preprocessing_info, include_sections, timestamp
        )
    elif format == 'Markdown':
        return _generate_markdown_report(
            results, aggregated, recommendations, explanations,
            preprocessing_info, include_sections, timestamp
        )
    else:
        return _generate_text_report(
            results, aggregated, recommendations, explanations,
            preprocessing_info, include_sections, timestamp
        )


def _generate_html_report(
    results, aggregated, recommendations, explanations,
    preprocessing_info, include_sections, timestamp
) -> str:
    """Generate HTML format report."""
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Feature Analysis Report</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; background: #1a1a2e; color: #eee; }}
            h1 {{ color: #6366F1; border-bottom: 2px solid #6366F1; padding-bottom: 10px; }}
            h2 {{ color: #8B5CF6; margin-top: 30px; }}
            table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
            th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #333; }}
            th {{ background: #2a2a4e; color: #6366F1; }}
            tr:hover {{ background: #252545; }}
            .card {{ background: #252545; border-radius: 10px; padding: 15px; margin: 10px 0; }}
            .metric {{ display: inline-block; background: #2a2a4e; padding: 10px 20px; border-radius: 8px; margin: 5px; text-align: center; }}
            .metric-value {{ font-size: 24px; color: #6366F1; font-weight: bold; }}
            .metric-label {{ font-size: 12px; color: #888; }}
            .badge {{ display: inline-block; padding: 3px 10px; border-radius: 15px; font-size: 12px; }}
            .badge-success {{ background: #10B981; }}
            .badge-warning {{ background: #F59E0B; }}
        </style>
    </head>
    <body>
        <h1>🔬 Feature Analysis Report</h1>
        <p style="color: #888;">Generated: {timestamp}</p>
    """
    
    if 'Summary Statistics' in include_sections:
        n_methods = len(results)
        n_features = len(aggregated)
        n_recommended = recommendations.get('n_recommended', 0)
        
        html += f"""
        <h2>📊 Summary Statistics</h2>
        <div>
            <div class="metric"><div class="metric-value">{n_methods}</div><div class="metric-label">Methods Used</div></div>
            <div class="metric"><div class="metric-value">{n_features}</div><div class="metric-label">Features Analyzed</div></div>
            <div class="metric"><div class="metric-value">{n_recommended}</div><div class="metric-label">Recommended</div></div>
        </div>
        """
    
    if 'Top Features Table' in include_sections:
        html += "<h2>🏆 Top Features</h2><table><tr><th>Rank</th><th>Feature</th><th>Score</th></tr>"
        for _, row in aggregated.head(15).iterrows():
            html += f"<tr><td>{int(row['final_rank'])}</td><td>{row['feature']}</td><td>{row['weighted_score']:.4f}</td></tr>"
        html += "</table>"
    
    if 'Feature Explanations' in include_sections:
        html += "<h2>📝 Feature Explanations</h2>"
        for exp in explanations[:10]:
            html += f"""
            <div class="card">
                <strong>#{exp['rank']} {exp['feature']}</strong> (Score: {exp['score']:.4f})
                <p style="color: #aaa; margin: 5px 0;">{exp['summary']}</p>
            </div>
            """
    
    if 'Recommendations' in include_sections:
        html += f"""
        <h2>💡 Recommendations</h2>
        <p>{recommendations.get('rationale', '')}</p>
        <p><strong>Recommended Features ({n_recommended}):</strong></p>
        <ul>
        """
        for feat in recommendations.get('recommended_features', [])[:15]:
            html += f"<li>{feat}</li>"
        html += "</ul>"
    
    html += "</body></html>"
    return html


def _generate_markdown_report(
    results, aggregated, recommendations, explanations,
    preprocessing_info, include_sections, timestamp
) -> str:
    """Generate Markdown format report."""
    
    md = f"# 🔬 Feature Analysis Report\n\n*Generated: {timestamp}*\n\n"
    
    if 'Summary Statistics' in include_sections:
        md += "## 📊 Summary Statistics\n\n"
        md += f"- **Methods Used**: {len(results)}\n"
        md += f"- **Features Analyzed**: {len(aggregated)}\n"
        md += f"- **Recommended Features**: {recommendations.get('n_recommended', 0)}\n\n"
    
    if 'Top Features Table' in include_sections:
        md += "## 🏆 Top Features\n\n"
        md += "| Rank | Feature | Score |\n|------|---------|-------|\n"
        for _, row in aggregated.head(15).iterrows():
            md += f"| {int(row['final_rank'])} | {row['feature']} | {row['weighted_score']:.4f} |\n"
        md += "\n"
    
    if 'Feature Explanations' in include_sections:
        md += "## 📝 Feature Explanations\n\n"
        for exp in explanations[:10]:
            md += f"**#{exp['rank']} {exp['feature']}** (Score: {exp['score']:.4f})\n\n"
            md += f"> {exp['summary']}\n\n"
    
    if 'Recommendations' in include_sections:
        md += "## 💡 Recommendations\n\n"
        md += f"{recommendations.get('rationale', '')}\n\n"
        md += f"**Recommended Features ({recommendations.get('n_recommended', 0)}):**\n\n"
        for feat in recommendations.get('recommended_features', [])[:15]:
            md += f"- {feat}\n"
    
    return md


def _generate_text_report(
    results, aggregated, recommendations, explanations,
    preprocessing_info, include_sections, timestamp
) -> str:
    """Generate plain text format report."""
    
    txt = f"FEATURE ANALYSIS REPORT\n{'='*50}\nGenerated: {timestamp}\n\n"
    
    if 'Summary Statistics' in include_sections:
        txt += "SUMMARY STATISTICS\n" + "-"*30 + "\n"
        txt += f"Methods Used: {len(results)}\n"
        txt += f"Features Analyzed: {len(aggregated)}\n"
        txt += f"Recommended Features: {recommendations.get('n_recommended', 0)}\n\n"
    
    if 'Top Features Table' in include_sections:
        txt += "TOP FEATURES\n" + "-"*30 + "\n"
        for _, row in aggregated.head(15).iterrows():
            txt += f"{int(row['final_rank']):3d}. {row['feature']:30s} {row['weighted_score']:.4f}\n"
        txt += "\n"
    
    if 'Feature Explanations' in include_sections:
        txt += "FEATURE EXPLANATIONS\n" + "-"*30 + "\n"
        for exp in explanations[:10]:
            txt += f"\n#{exp['rank']} {exp['feature']} (Score: {exp['score']:.4f})\n"
            txt += f"   {exp['summary']}\n"
        txt += "\n"
    
    if 'Recommendations' in include_sections:
        txt += "RECOMMENDATIONS\n" + "-"*30 + "\n"
        txt += f"{recommendations.get('rationale', '')}\n\n"
        txt += "Recommended Features:\n"
        for feat in recommendations.get('recommended_features', [])[:15]:
            txt += f"  - {feat}\n"
    
    return txt


def _render_save_analysis(
    results: Dict[str, pd.DataFrame],
    aggregated: pd.DataFrame,
    recommendations: Dict[str, Any],
    preprocessing_info: Dict[str, Any]
):
    """Save complete analysis state."""
    
    st.markdown("### Save Complete Analysis")
    
    st.info(
        "💾 Save the complete analysis state as JSON. "
        "This includes all results, rankings, and recommendations."
    )
    
    # Prepare JSON data
    analysis_data = {
        'timestamp': datetime.now().isoformat(),
        'preprocessing': {
            'original_shape': list(preprocessing_info.get('original_shape', [])),
            'final_shape': list(preprocessing_info.get('final_shape', [])),
            'encoded_columns': preprocessing_info.get('encoded_columns', []),
            'dropped_columns': preprocessing_info.get('dropped_columns', [])
        },
        'methods_used': list(results.keys()),
        'aggregated_ranking': aggregated.head(30).to_dict(orient='records'),
        'recommendations': {
            'n_recommended': recommendations.get('n_recommended', 0),
            'recommended_features': recommendations.get('recommended_features', []),
            'high_confidence': recommendations.get('high_confidence', []),
            'medium_confidence': recommendations.get('medium_confidence', []),
            'rationale': recommendations.get('rationale', '')
        }
    }
    
    json_str = json.dumps(analysis_data, indent=2)
    
    st.download_button(
        label="💾 Download Analysis State (JSON)",
        data=json_str,
        file_name="feature_analysis_state.json",
        mime="application/json"
    )
