"""
AutoFeature - Automated Feature Engineering Pipeline
Main Streamlit Application
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import DataLoader
from src.preprocessor import Preprocessor
from src.feature_analyzer import FeatureAnalyzer
from src.feature_selector import FeatureSelector
from src.visualizer import Visualizer
from components.upload import (
    render_upload_section, 
    render_target_selector,
    render_preprocessing_options
)
from components.analysis_dashboard import (
    render_analysis_dashboard,
    render_preprocessing_summary
)
from components.visualization_panel import (
    render_visualization_panel,
    render_recommendation_summary_chart
)
from components.report_generator import render_export_section


# Page configuration
st.set_page_config(
    page_title="AutoFeature | Feature Engineering Pipeline",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Main theme */
    .stApp {
        background: linear-gradient(180deg, #0E1117 0%, #1A1A2E 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 50%, #EC4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0;
    }
    
    .sub-header {
        color: #9CA3AF;
        text-align: center;
        font-size: 1.1rem;
        margin-top: 0;
    }
    
    /* Card styling */
    .feature-card {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
    }
    
    /* Metric cards */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(139, 92, 246, 0.15) 100%);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 10px;
        padding: 15px;
    }
    
    div[data-testid="metric-container"] label {
        color: #9CA3AF !important;
    }
    
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #6366F1 !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 25px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(99, 102, 241, 0.4);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(99, 102, 241, 0.1);
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(99, 102, 241, 0.1);
        border-radius: 8px;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #6366F1, #8B5CF6, #EC4899);
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1A1A2E 0%, #16162A 100%);
    }
    
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #8B5CF6;
    }
    
    /* Dataframe styling */
    .dataframe {
        border: 1px solid rgba(99, 102, 241, 0.2) !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Animation */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .analyzing {
        animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'metadata' not in st.session_state:
        st.session_state.metadata = None
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'aggregated' not in st.session_state:
        st.session_state.aggregated = None
    if 'explanations' not in st.session_state:
        st.session_state.explanations = None
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = None
    if 'X' not in st.session_state:
        st.session_state.X = None
    if 'y' not in st.session_state:
        st.session_state.y = None
    if 'preprocessing_info' not in st.session_state:
        st.session_state.preprocessing_info = None
    if 'shap_values' not in st.session_state:
        st.session_state.shap_values = None


def render_sidebar():
    """Render the sidebar."""
    with st.sidebar:
        st.markdown("## 🔬 AutoFeature")
        st.markdown("*Automated Feature Engineering*")
        
        st.markdown("---")
        
        st.markdown("### 📖 How It Works")
        st.markdown("""
        1. **Upload** your tabular dataset
        2. **Select** the target variable
        3. **Analyze** feature importance
        4. **Review** recommendations
        5. **Export** results
        """)
        
        st.markdown("---")
        
        st.markdown("### 🔍 Analysis Methods")
        st.markdown("""
        - 📊 **Correlation Analysis**
        - 📈 **Mutual Information**
        - 🌲 **Random Forest**
        - ⚡ **XGBoost**
        - 🔄 **Permutation Importance**
        - 🔮 **SHAP Values**
        """)
        
        st.markdown("---")
        
        st.markdown("### ⚙️ Settings")
        
        theme = st.selectbox(
            "Chart Theme",
            options=['dark', 'light'],
            index=0
        )
        
        st.session_state.visualizer = Visualizer(theme=theme)
        
        st.markdown("---")
        
        st.markdown(
            "<p style='text-align: center; color: #6B7280; font-size: 0.8em;'>"
            "Built with Streamlit"
            "</p>",
            unsafe_allow_html=True
        )


def render_header():
    """Render the main header."""
    st.markdown("<h1 class='main-header'>🔬 AutoFeature</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='sub-header'>Automated Feature Engineering Pipeline for Data Scientists</p>",
        unsafe_allow_html=True
    )
    st.markdown("")


def run_analysis(df, target_column, preprocessing_options, methods):
    """Run the complete feature analysis pipeline."""
    
    progress = st.progress(0, text="Initializing...")
    
    # Step 1: Preprocessing
    progress.progress(10, text="Preprocessing data...")
    
    preprocessor = Preprocessor()
    X, y, preprocessing_info = preprocessor.prepare_for_analysis(
        df,
        target_column,
        handle_missing=preprocessing_options['handle_missing'],
        encode_categorical=preprocessing_options['encode_categorical'],
        remove_high_cardinality=preprocessing_options['remove_high_cardinality'],
        cardinality_threshold=preprocessing_options['cardinality_threshold']
    )
    
    st.session_state.X = X
    st.session_state.y = y
    st.session_state.preprocessing_info = preprocessing_info
    
    # Step 2: Detect problem type
    progress.progress(20, text="Detecting problem type...")
    problem_type = Preprocessor.detect_problem_type(y)
    
    # Step 3: Feature Analysis
    progress.progress(30, text="Running feature analysis...")
    
    analyzer = FeatureAnalyzer(problem_type=problem_type)
    results = analyzer.analyze_all(X, y, methods=methods)
    
    st.session_state.results = results
    
    # Store SHAP values if available
    if analyzer.shap_values is not None:
        st.session_state.shap_values = analyzer.shap_values
    
    # Step 4: Aggregate results
    progress.progress(70, text="Aggregating results...")
    aggregated = analyzer.get_aggregated_importance()
    st.session_state.aggregated = aggregated
    
    # Step 5: Generate explanations
    progress.progress(80, text="Generating explanations...")
    explanations = analyzer.get_feature_explanations(X, y, top_n=15)
    st.session_state.explanations = explanations
    
    # Step 6: Get recommendations
    progress.progress(90, text="Generating recommendations...")
    selector = FeatureSelector(problem_type=problem_type)
    recommendations = selector.get_feature_recommendations(aggregated, X, y)
    st.session_state.recommendations = recommendations
    
    progress.progress(100, text="Analysis complete!")
    
    st.session_state.analysis_complete = True
    
    return True


def main():
    """Main application entry point."""
    
    init_session_state()
    render_sidebar()
    render_header()
    
    # Initialize visualizer if not exists
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = Visualizer(theme='dark')
    
    # Main content
    st.markdown("---")
    
    # Step 1: Upload
    df, metadata = render_upload_section()
    
    if df is not None:
        st.session_state.df = df
        st.session_state.metadata = metadata
        
        st.markdown("---")
        
        # Step 2: Target selection
        target_column = render_target_selector(df, metadata)
        
        if target_column:
            st.markdown("---")
            
            # Step 3: Preprocessing options
            preprocessing_options = render_preprocessing_options()
            
            # Step 4: Analysis methods
            st.markdown("### 🔬 Select Analysis Methods")
            
            col1, col2 = st.columns(2)
            
            with col1:
                use_correlation = st.checkbox("📊 Correlation Analysis", value=True)
                use_mutual_info = st.checkbox("📈 Mutual Information", value=True)
                use_random_forest = st.checkbox("🌲 Random Forest", value=True)
            
            with col2:
                use_xgboost = st.checkbox("⚡ XGBoost", value=True)
                use_permutation = st.checkbox("🔄 Permutation Importance", value=True)
                use_shap = st.checkbox("🔮 SHAP Values", value=True)
            
            methods = []
            if use_correlation:
                methods.append('correlation')
            if use_mutual_info:
                methods.append('mutual_info')
            if use_random_forest:
                methods.append('random_forest')
            if use_xgboost:
                methods.append('xgboost')
            if use_permutation:
                methods.append('permutation')
            if use_shap:
                methods.append('shap')
            
            st.markdown("---")
            
            # Run analysis button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                analyze_button = st.button(
                    "🚀 Run Feature Analysis",
                    type="primary",
                    use_container_width=True
                )
            
            if analyze_button:
                if not methods:
                    st.error("Please select at least one analysis method.")
                else:
                    with st.spinner("Running analysis..."):
                        success = run_analysis(
                            df, target_column, preprocessing_options, methods
                        )
                    
                    if success:
                        st.success("✅ Analysis complete! Scroll down to view results.")
                        st.balloons()
            
            # Display results if analysis is complete
            if st.session_state.analysis_complete:
                st.markdown("---")
                st.markdown("# 📊 Analysis Results")
                
                # Preprocessing summary
                render_preprocessing_summary(st.session_state.preprocessing_info)
                
                # Main dashboard
                render_analysis_dashboard(
                    st.session_state.results,
                    st.session_state.aggregated,
                    st.session_state.explanations,
                    st.session_state.recommendations
                )
                
                st.markdown("---")
                
                # Visualizations
                render_visualization_panel(
                    st.session_state.visualizer,
                    st.session_state.results,
                    st.session_state.aggregated,
                    st.session_state.X,
                    st.session_state.y,
                    st.session_state.recommendations,
                    st.session_state.explanations,
                    st.session_state.shap_values
                )
                
                st.markdown("---")
                
                # Recommendation summary chart
                st.markdown("## 📋 Recommendation Summary")
                render_recommendation_summary_chart(
                    st.session_state.visualizer,
                    st.session_state.recommendations,
                    st.session_state.explanations
                )
                
                st.markdown("---")
                
                # Export section
                render_export_section(
                    st.session_state.df,
                    st.session_state.X,
                    st.session_state.y,
                    st.session_state.results,
                    st.session_state.aggregated,
                    st.session_state.recommendations,
                    st.session_state.explanations,
                    st.session_state.preprocessing_info
                )


if __name__ == "__main__":
    main()
