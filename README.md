# 🔬 AutoFeature - Automated Feature Engineering Pipeline

A powerful, interactive tool for data scientists to automatically analyze tabular datasets, identify the most important features, and understand *why* those features matter through rich visualizations and explainability.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## ✨ Features

### 🔍 Multi-Method Feature Analysis
- **Correlation Analysis** - Pearson & Spearman correlation with target
- **Mutual Information** - Non-linear dependency detection
- **Random Forest Importance** - Tree-based Gini importance
- **XGBoost Importance** - Gradient boosting feature weights
- **Permutation Importance** - Model-agnostic importance via shuffling
- **SHAP Values** - Game-theoretic explainability scores

### 📊 Rich Visualizations
- Interactive feature importance bar charts
- Correlation heatmaps
- Cumulative importance curves
- Method comparison charts
- Feature distribution plots
- SHAP summary and waterfall plots

### 💡 Intelligent Recommendations
- Aggregated feature rankings across all methods
- Confidence-based categorization (High/Medium/Low)
- Natural language explanations for each feature
- Optimal feature set recommendations

### 📥 Export & Reporting
- Export importance scores to CSV
- Download reduced datasets with selected features
- Generate HTML/Markdown/Text reports
- Save complete analysis state as JSON

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Installation

1. **Clone or navigate to the project directory:**
```bash
cd "c:\Users\Farhan Ramadhan\Desktop\Bvrie"
```

2. **Create a virtual environment (recommended):**
```bash
python -m venv venv
venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Run the application:**
```bash
streamlit run app.py
```

5. **Open your browser** and navigate to `http://localhost:8501`

## 📖 Usage Guide

### Step 1: Upload Your Dataset
- Drag and drop a CSV, Excel (.xlsx), or JSON file
- The application will automatically detect data types and show a preview

### Step 2: Select Target Variable
- Choose the column you want to predict
- The app will auto-detect the problem type (classification/regression)

### Step 3: Configure Analysis
- Adjust preprocessing options (missing value handling, encoding)
- Select which analysis methods to run

### Step 4: Run Analysis
- Click "Run Feature Analysis" to start
- Watch the progress bar as each method is applied

### Step 5: Review Results
- Browse the top features with explanations
- Explore interactive visualizations
- Compare rankings across different methods

### Step 6: Export
- Download recommended feature lists
- Generate detailed reports
- Export reduced datasets

## 📁 Project Structure

```
Bvrie/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Dataset loading utilities
│   ├── preprocessor.py         # Data cleaning & transformation
│   ├── feature_analyzer.py     # Feature importance calculations
│   ├── feature_selector.py     # Feature selection algorithms
│   └── visualizer.py           # Visualization generators
├── components/
│   ├── __init__.py
│   ├── upload.py               # File upload component
│   ├── analysis_dashboard.py   # Feature analysis UI
│   ├── visualization_panel.py  # Interactive charts
│   └── report_generator.py     # Export & report functionality
└── sample_data/
    └── sample_dataset.csv      # Example Titanic dataset
```

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Frontend** | Streamlit |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn, XGBoost |
| **Explainability** | SHAP |
| **Visualization** | Plotly, Matplotlib, Seaborn |

## 🎨 Screenshots

The application features a modern dark theme with:
- Gradient accents and smooth animations
- Interactive Plotly charts
- Clean metric cards and data tables
- Responsive layout for all screen sizes

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📄 License

This project is licensed under the MIT License.

---

<p align="center">
  Built for Data Scientists
</p>
