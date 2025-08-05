import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os
import numpy as np
from pathlib import Path

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

from data_manager import data_manager
from data_viewer import display_data_section, create_data_sidebar

# Page configuration
st.set_page_config(
    page_title="Data Explorer - FinOps",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .dataset-card {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    .dataset-card:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
    .dataset-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 1rem 0;
    }
    .metric-highlight {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.25rem;
    }
    .chart-container {
        background-color: white;
        border: 1px solid #e9ecef;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def get_dataset_info():
    """Get information about all available datasets"""
    data_dir = Path("data")
    datasets = []
    
    if data_dir.exists():
        for file_path in data_dir.glob("*.csv"):
            try:
                df = pd.read_csv(file_path)
                file_size = file_path.stat().st_size / (1024 * 1024)  # MB
                last_modified = datetime.fromtimestamp(file_path.stat().st_mtime)
                
                # Determine dataset type based on filename
                dataset_type = file_path.stem.replace('_', ' ').title()
                
                # Get column types
                column_types = df.dtypes.to_dict()
                numeric_columns = [col for col, dtype in column_types.items() if pd.api.types.is_numeric_dtype(dtype)]
                date_columns = [col for col, dtype in column_types.items() if pd.api.types.is_datetime64_any_dtype(dtype) or 'date' in col.lower()]
                
                datasets.append({
                    'filename': file_path.name,
                    'dataset_type': dataset_type,
                    'rows': len(df),
                    'columns': len(df.columns),
                    'size_mb': file_size,
                    'last_modified': last_modified,
                    'numeric_columns': numeric_columns,
                    'date_columns': date_columns,
                    'all_columns': list(df.columns),
                    'sample_data': df.head(5)
                })
            except Exception as e:
                st.error(f"Error reading {file_path.name}: {str(e)}")
    
    return datasets

def create_visualization(df, dataset_type):
    """Create appropriate visualizations based on dataset type and content"""
    charts = []
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    date_cols = [col for col in df.columns if 'date' in col.lower() or pd.api.types.is_datetime64_any_dtype(df[col])]
    
    # Convert date columns
    for col in date_cols:
        try:
            df[col] = pd.to_datetime(df[col])
        except:
            pass
    
    # 1. Time series plot (if date column exists)
    if date_cols and numeric_cols:
        date_col = date_cols[0]
        numeric_col = numeric_cols[0]
        
        try:
            fig_time = px.line(df, x=date_col, y=numeric_col, title=f"{numeric_col} Over Time")
            fig_time.update_layout(height=400)
            charts.append(("Time Series", fig_time))
        except:
            pass
    
    # 2. Distribution plot for numeric columns
    if numeric_cols:
        for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
            try:
                fig_dist = px.histogram(df, x=col, title=f"Distribution of {col}")
                fig_dist.update_layout(height=400)
                charts.append((f"Distribution - {col}", fig_dist))
            except:
                pass
    
    # 3. Correlation heatmap (if multiple numeric columns)
    if len(numeric_cols) > 1:
        try:
            corr_matrix = df[numeric_cols].corr()
            fig_corr = px.imshow(
                corr_matrix,
                title="Correlation Matrix",
                color_continuous_scale='RdBu',
                aspect='auto'
            )
            fig_corr.update_layout(height=400)
            charts.append(("Correlation Matrix", fig_corr))
        except:
            pass
    
    # 4. Bar chart for categorical columns (if any)
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        for col in categorical_cols[:2]:  # Limit to first 2 categorical columns
            try:
                value_counts = df[col].value_counts().head(10)  # Top 10 values
                fig_bar = px.bar(
                    x=value_counts.index,
                    y=value_counts.values,
                    title=f"Top 10 Values in {col}",
                    labels={'x': col, 'y': 'Count'}
                )
                fig_bar.update_layout(height=400)
                charts.append((f"Bar Chart - {col}", fig_bar))
            except:
                pass
    
    # 5. Scatter plot (if multiple numeric columns)
    if len(numeric_cols) >= 2:
        try:
            fig_scatter = px.scatter(
                df, 
                x=numeric_cols[0], 
                y=numeric_cols[1], 
                title=f"{numeric_cols[0]} vs {numeric_cols[1]}"
            )
            fig_scatter.update_layout(height=400)
            charts.append(("Scatter Plot", fig_scatter))
        except:
            pass
    
    # 6. Box plot for numeric columns
    if numeric_cols:
        try:
            fig_box = px.box(df, y=numeric_cols[0], title=f"Box Plot - {numeric_cols[0]}")
            fig_box.update_layout(height=400)
            charts.append(("Box Plot", fig_box))
        except:
            pass
    
    return charts

def main():
    st.title("🔍 Data Explorer")
    st.markdown("Explore, visualize, and download all FinOps datasets")
    
    # Data Explorer Documentation
    with st.expander("📚 Data Explorer Guide & Features", expanded=False):
        st.markdown("""
        ### 🔍 Data Explorer Overview
        
        The Data Explorer provides a comprehensive interface for exploring, analyzing, and downloading all FinOps datasets.
        
        #### 📊 Available Datasets:
        - **Cost Data**: Daily AWS cost data with trends and seasonality (30 days)
        - **Service Breakdown**: Cost distribution by AWS service (EC2, S3, RDS, etc.)
        - **Anomaly Data**: Cost data with detected anomalies and severity levels
        - **Multivariate Data**: Multi-dimensional cost data with usage metrics and API calls
        - **Forecast Scenarios**: Different forecasting scenarios (Conservative, Baseline, Aggressive)
        - **Optimization Recommendations**: Cost optimization suggestions with savings estimates
        - **Budget Data**: Budget tracking and utilization data by environment
        - **Forecast Data**: Historical vs forecasted costs with confidence intervals
        - **Cost Allocation**: Department and project cost allocation with variance analysis
        - **Tag Analysis**: Cost analysis by AWS tags (Environment, Team, Application, CostCenter)
        - **Historical Cost Data**: Long-term historical cost trends (365 days)
        
        #### 🔧 Key Features:
        - **📋 Data Preview**: View first 10 rows of any dataset
        - **📊 Automatic Visualizations**: Smart chart generation based on data types
        - **📈 Statistical Analysis**: Comprehensive statistics and missing data analysis
        - **📥 Data Download**: Export in CSV, JSON, or Excel formats
        - **🔍 Data Quality Assessment**: Column types, null values, and data ranges
        - **📊 Dataset Comparison**: Overview table of all available datasets
        
        #### 📈 Visualization Types:
        - **Time Series**: For temporal data showing trends over time
        - **Distribution Histograms**: For numeric data showing data spread
        - **Correlation Heatmaps**: For multiple numeric columns showing relationships
        - **Bar Charts**: For categorical data showing frequency distributions
        - **Scatter Plots**: For numeric pairs showing relationships
        - **Box Plots**: For numeric data showing quartiles and outliers
        
        #### 💡 Analysis Insights:
        - **Data Structure**: Column types, null counts, and data ranges
        - **Statistical Summary**: Mean, median, standard deviation, quartiles
        - **Missing Data**: Identification and percentage of missing values
        - **Data Quality**: Assessment of data completeness and consistency
        - **Trend Analysis**: For time series data, identification of patterns
        
        #### 🎯 Best Practices:
        - Start with the data overview to understand structure
        - Use visualizations to identify patterns and outliers
        - Check missing data analysis for data quality issues
        - Export data for further analysis in external tools
        - Compare datasets to understand relationships between different data types
        
        #### 🔍 Data Quality Indicators:
        - **Complete Data**: No missing values in critical columns
        - **Consistent Data**: Data types match expected formats
        - **Valid Ranges**: Values within expected business ranges
        - **Temporal Consistency**: Date ranges align with business expectations
        - **Statistical Validity**: Distributions and correlations make business sense
        
        #### 📊 Interpreting Visualizations:
        - **Time Series**: Look for trends, seasonality, and anomalies
        - **Distributions**: Identify skewness, outliers, and central tendency
        - **Correlations**: Understand relationships between variables
        - **Bar Charts**: Identify most common categories and patterns
        - **Scatter Plots**: Find relationships and clusters in data
        - **Box Plots**: Detect outliers and understand data spread
        """)
    
    # Get all dataset information
    datasets = get_dataset_info()
    
    if not datasets:
        st.error("No datasets found in the data folder. Please generate data first.")
        return
    
    # Sidebar for navigation and data management
    with st.sidebar:
        st.header("🗂️ Dataset Navigation")
        
        # Dataset selector
        selected_dataset = st.selectbox(
            "Select Dataset",
            [d['dataset_type'] for d in datasets],
            index=0
        )
        
        st.markdown("---")
        
        # Quick stats
        st.subheader("📊 Quick Stats")
        total_datasets = len(datasets)
        total_rows = sum(d['rows'] for d in datasets)
        total_size = sum(d['size_mb'] for d in datasets)
        
        st.metric("Total Datasets", total_datasets)
        st.metric("Total Rows", f"{total_rows:,}")
        st.metric("Total Size", f"{total_size:.2f} MB")
        
        # Add data management sidebar
        create_data_sidebar(data_manager)
    
    # Get selected dataset
    selected_data_info = next(d for d in datasets if d['dataset_type'] == selected_dataset)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"""
        <div class="dataset-header">
            <h2>📊 {selected_dataset}</h2>
            <p>File: {selected_data_info['filename']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Dataset metrics
        st.markdown("**📈 Dataset Metrics**")
        st.metric("Rows", f"{selected_data_info['rows']:,}")
        st.metric("Columns", selected_data_info['columns'])
        st.metric("Size", f"{selected_data_info['size_mb']:.2f} MB")
        st.metric("Last Modified", selected_data_info['last_modified'].strftime('%Y-%m-%d'))
    
    # Load the full dataset
    try:
        # Map filename to data type for data manager
        filename_to_type = {
            'cost_data.csv': 'cost_data',
            'service_breakdown.csv': 'service_breakdown',
            'anomaly_data.csv': 'anomaly_data',
            'multivariate_cost_data.csv': 'multivariate_data',
            'forecast_scenarios.csv': 'forecast_scenarios',
            'optimization_recommendations.csv': 'optimization_recommendations',
            'budget_data.csv': 'budget_data',
            'forecast_data.csv': 'forecast_data',
            'cost_allocation.csv': 'cost_allocation',
            'tag_analysis.csv': 'tag_analysis',
            'historical_cost_data.csv': 'historical_cost_data'
        }
        
        data_type = filename_to_type.get(selected_data_info['filename'])
        if data_type:
            df = data_manager.get_data(data_type)
        else:
            # Fallback: read directly from CSV
            df = pd.read_csv(f"data/{selected_data_info['filename']}")
        
        # Data overview
        st.markdown("---")
        st.subheader("📋 Data Overview")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**📊 Column Information**")
            for col in df.columns:
                dtype = str(df[col].dtype)
                null_count = df[col].isnull().sum()
                st.markdown(f"• **{col}**: {dtype} ({null_count} nulls)")
        
        with col2:
            st.markdown("**🔢 Numeric Columns**")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                for col in numeric_cols:
                    mean_val = df[col].mean()
                    st.markdown(f"• **{col}**: {mean_val:.2f} (mean)")
            else:
                st.markdown("• No numeric columns")
        
        with col3:
            st.markdown("**📅 Date Columns**")
            date_cols = [col for col in df.columns if 'date' in col.lower() or pd.api.types.is_datetime64_any_dtype(df[col])]
            if date_cols:
                for col in date_cols:
                    try:
                        df[col] = pd.to_datetime(df[col])
                        date_range = f"{df[col].min().strftime('%Y-%m-%d')} to {df[col].max().strftime('%Y-%m-%d')}"
                        st.markdown(f"• **{col}**: {date_range}")
                    except:
                        st.markdown(f"• **{col}**: Date column")
            else:
                st.markdown("• No date columns")
        
        # Data preview
        st.markdown("---")
        st.subheader("👀 Data Preview")
        
        # Show first few rows
        st.dataframe(df.head(10), use_container_width=True)
        
        if len(df) > 10:
            st.info(f"Showing first 10 rows of {len(df)} total rows")
        
        # Visualizations
        st.markdown("---")
        st.subheader("📊 Data Visualizations")
        
        charts = create_visualization(df, selected_dataset)
        
        if charts:
            # Create tabs for different chart types
            chart_names = [name for name, _ in charts]
            chart_tabs = st.tabs(chart_names)
            
            for i, (name, fig) in enumerate(charts):
                with chart_tabs[i]:
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add chart insights
                    if "Time Series" in name:
                        st.markdown("**💡 Time Series Insights**")
                        st.markdown("• Shows trends and patterns over time")
                        st.markdown("• Look for seasonality, trends, and anomalies")
                    elif "Distribution" in name:
                        st.markdown("**💡 Distribution Insights**")
                        st.markdown("• Shows the spread and shape of the data")
                        st.markdown("• Look for skewness, outliers, and central tendency")
                    elif "Correlation" in name:
                        st.markdown("**💡 Correlation Insights**")
                        st.markdown("• Shows relationships between variables")
                        st.markdown("• Red = negative correlation, Blue = positive correlation")
                    elif "Bar Chart" in name:
                        st.markdown("**💡 Categorical Insights**")
                        st.markdown("• Shows frequency of categorical values")
                        st.markdown("• Useful for identifying most common categories")
        else:
            st.info("No suitable visualizations could be generated for this dataset.")
        
        # Data download section
        st.markdown("---")
        st.subheader("📥 Download Data")
        display_data_section(df, selected_dataset, f"Complete {selected_dataset} dataset", show_preview=False)
        
        # Data statistics
        st.markdown("---")
        st.subheader("📈 Detailed Statistics")
        
        if numeric_cols:
            st.markdown("**🔢 Numeric Column Statistics**")
            stats_df = df[numeric_cols].describe()
            st.dataframe(stats_df, use_container_width=True)
        
        # Missing data analysis
        st.markdown("**🔍 Missing Data Analysis**")
        missing_data = df.isnull().sum()
        missing_df = pd.DataFrame({
            'Column': missing_data.index,
            'Missing Count': missing_data.values,
            'Missing Percentage': (missing_data.values / len(df)) * 100
        })
        st.dataframe(missing_df, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
    
    # Dataset comparison
    st.markdown("---")
    st.subheader("📊 All Datasets Overview")
    
    # Create comparison table
    comparison_data = []
    for dataset in datasets:
        comparison_data.append({
            'Dataset': dataset['dataset_type'],
            'Rows': dataset['rows'],
            'Columns': dataset['columns'],
            'Size (MB)': f"{dataset['size_mb']:.2f}",
            'Last Modified': dataset['last_modified'].strftime('%Y-%m-%d'),
            'Numeric Cols': len(dataset['numeric_columns']),
            'Date Cols': len(dataset['date_columns'])
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)

if __name__ == "__main__":
    main() 