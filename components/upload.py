"""
File Upload Component
Handles dataset upload and initial preview
"""

import streamlit as st
import pandas as pd
from typing import Optional, Tuple, Dict, Any
from src.data_loader import DataLoader


def render_upload_section() -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
    """
    Render the file upload section.
    
    Returns:
        Tuple of (DataFrame, metadata) if file uploaded, (None, None) otherwise
    """
    st.markdown("### 📁 Upload Your Dataset")
    
    # File uploader with styling
    uploaded_file = st.file_uploader(
        "Drop your CSV, Excel, or JSON file here",
        type=['csv', 'xlsx', 'xls', 'json'],
        help="Supported formats: CSV, Excel (.xlsx, .xls), JSON"
    )
    
    if uploaded_file is None:
        _render_upload_placeholder()
        return None, None
    
    # Load the file
    try:
        with st.spinner("Loading dataset..."):
            df, metadata = DataLoader.load_from_file(uploaded_file, uploaded_file.name)
        
        st.success(f"✅ Loaded **{metadata['n_rows']:,}** rows × **{metadata['n_cols']}** columns")
        
        # Show dataset info
        _render_dataset_info(df, metadata)
        
        return df, metadata
        
    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")
        return None, None


def _render_upload_placeholder():
    """Render a styled placeholder for file upload."""
    st.markdown("""
    <div style="
        border: 2px dashed #6366F1;
        border-radius: 10px;
        padding: 40px;
        text-align: center;
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
        margin: 20px 0;
    ">
        <h4 style="color: #8B5CF6; margin-bottom: 10px;">🚀 Get Started</h4>
        <p style="color: #9CA3AF;">Upload a tabular dataset to begin feature analysis</p>
        <p style="font-size: 0.85em; color: #6B7280;">
            Supported formats: CSV, Excel (.xlsx), JSON
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sample datasets info
    with st.expander("📊 Don't have a dataset? Try these sample sources"):
        st.markdown("""
        - **Kaggle**: [titanic](https://www.kaggle.com/c/titanic), [house-prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
        - **UCI ML Repository**: [iris](https://archive.ics.uci.edu/ml/datasets/iris), [wine](https://archive.ics.uci.edu/ml/datasets/wine)
        - **Scikit-learn**: Use `sklearn.datasets` to generate sample data
        """)


def _render_dataset_info(df: pd.DataFrame, metadata: Dict[str, Any]):
    """Render dataset information cards."""
    
    # Info metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Rows", f"{metadata['n_rows']:,}")
    with col2:
        st.metric("📋 Columns", metadata['n_cols'])
    with col3:
        st.metric("🔢 Numeric", len(metadata['numeric_columns']))
    with col4:
        st.metric("📝 Categorical", len(metadata['categorical_columns']))
    
    # Data preview tabs
    tab1, tab2, tab3 = st.tabs(["👀 Preview", "📈 Statistics", "🔍 Column Info"])
    
    with tab1:
        st.dataframe(
            df.head(10),
            use_container_width=True,
            height=300
        )
    
    with tab2:
        # Only show stats for numeric columns
        if metadata['numeric_columns']:
            st.dataframe(
                df[metadata['numeric_columns']].describe().round(2),
                use_container_width=True
            )
        else:
            st.info("No numeric columns found for statistics.")
    
    with tab3:
        _render_column_info_table(df, metadata)


def _render_column_info_table(df: pd.DataFrame, metadata: Dict[str, Any]):
    """Render detailed column information."""
    
    column_info = []
    for col in df.columns:
        info = {
            'Column': col,
            'Type': str(df[col].dtype),
            'Missing': metadata['missing_counts'].get(col, 0),
            'Missing %': f"{metadata['missing_percentages'].get(col, 0):.1f}%",
            'Unique': metadata['unique_counts'].get(col, 0)
        }
        column_info.append(info)
    
    info_df = pd.DataFrame(column_info)
    
    # Style the dataframe
    st.dataframe(
        info_df,
        use_container_width=True,
        hide_index=True,
        height=min(400, len(df.columns) * 35 + 50)
    )


def render_target_selector(
    df: pd.DataFrame,
    metadata: Dict[str, Any]
) -> Optional[str]:
    """
    Render target column selector.
    
    Returns:
        Selected target column name or None
    """
    st.markdown("### 🎯 Select Target Variable")
    
    # Try to auto-detect target
    auto_detected = DataLoader.detect_target_column(df)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Default to auto-detected if available
        default_idx = df.columns.tolist().index(auto_detected) if auto_detected and auto_detected in df.columns else 0
        
        target_column = st.selectbox(
            "Choose the column to predict (target variable)",
            options=df.columns.tolist(),
            index=default_idx,
            help="Select the variable you want to predict. Features will be ranked by their importance for predicting this target."
        )
    
    with col2:
        if auto_detected:
            st.info(f"💡 Auto-detected: `{auto_detected}`")
    
    if target_column:
        # Show target info
        col_data = df[target_column]
        unique_values = col_data.nunique()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Unique Values", unique_values)
        with col2:
            missing = col_data.isnull().sum()
            st.metric("Missing", f"{missing} ({missing/len(df)*100:.1f}%)")
        with col3:
            problem_type = "Classification" if unique_values <= 20 or col_data.dtype == 'object' else "Regression"
            st.metric("Problem Type", problem_type)
        
        return target_column
    
    return None


def render_preprocessing_options() -> Dict[str, Any]:
    """
    Render preprocessing configuration options.
    
    Returns:
        Dictionary of preprocessing settings
    """
    with st.expander("⚙️ Preprocessing Options", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            missing_strategy = st.selectbox(
                "Handle Missing Values",
                options=['auto', 'median', 'mean', 'drop'],
                index=0,
                help="Strategy for handling missing values in features"
            )
            
            encode_categorical = st.checkbox(
                "Encode Categorical Variables",
                value=True,
                help="Convert categorical columns to numeric using label encoding"
            )
        
        with col2:
            remove_high_cardinality = st.checkbox(
                "Remove High-Cardinality Columns",
                value=True,
                help="Remove categorical columns with too many unique values"
            )
            
            cardinality_threshold = st.slider(
                "Cardinality Threshold",
                min_value=10,
                max_value=100,
                value=50,
                help="Maximum unique values for categorical columns"
            )
    
    return {
        'handle_missing': missing_strategy,
        'encode_categorical': encode_categorical,
        'remove_high_cardinality': remove_high_cardinality,
        'cardinality_threshold': cardinality_threshold,
        'scale_features': False
    }
