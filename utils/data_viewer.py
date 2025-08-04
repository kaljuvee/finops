import streamlit as st
import pandas as pd
from datetime import datetime
import io

def display_data_section(data, title, description="", show_download=True, show_preview=True, max_preview_rows=10):
    """
    Display a data section with download capabilities
    
    Args:
        data: pandas DataFrame to display
        title: Section title
        description: Optional description
        show_download: Whether to show download buttons
        show_preview: Whether to show data preview
        max_preview_rows: Maximum rows to show in preview
    """
    st.subheader(title)
    
    if description:
        st.markdown(description)
    
    # Data info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", len(data))
    with col2:
        st.metric("Columns", len(data.columns))
    with col3:
        st.metric("Size", f"{data.memory_usage(deep=True).sum() / 1024:.1f} KB")
    
    # Download section
    if show_download:
        st.markdown("### 📥 Download Data")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # CSV download
            csv_data = data.to_csv(index=False)
            st.download_button(
                label="📄 Download CSV",
                data=csv_data,
                file_name=f"{title.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # JSON download
            json_data = data.to_json(orient='records', indent=2)
            st.download_button(
                label="📋 Download JSON",
                data=json_data,
                file_name=f"{title.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
        
        with col3:
            # Excel download
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                data.to_excel(writer, sheet_name='Data', index=False)
            excel_data = buffer.getvalue()
            st.download_button(
                label="📊 Download Excel",
                data=excel_data,
                file_name=f"{title.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    # Data preview
    if show_preview:
        st.markdown("### 👀 Data Preview")
        
        # Show first few rows
        preview_data = data.head(max_preview_rows)
        st.dataframe(preview_data, use_container_width=True)
        
        if len(data) > max_preview_rows:
            st.info(f"Showing first {max_preview_rows} rows of {len(data)} total rows")
    
    st.markdown("---")

def display_data_info(data_info):
    """
    Display information about all available data files
    
    Args:
        data_info: List of dictionaries with data file information
    """
    st.subheader("📊 Data Files Overview")
    
    # Create a DataFrame for better display
    info_df = pd.DataFrame(data_info)
    
    if 'error' in info_df.columns:
        # Handle files with errors
        error_files = info_df[info_df['error'].notna()]
        if not error_files.empty:
            st.error("Some data files have errors:")
            for _, row in error_files.iterrows():
                st.write(f"- {row['filename']}: {row['error']}")
    
    # Display successful files
    success_files = info_df[info_df['error'].isna()] if 'error' in info_df.columns else info_df
    
    if not success_files.empty:
        # Format the data for display
        display_data = []
        for _, row in success_files.iterrows():
            display_data.append({
                'File': row['filename'],
                'Type': row['type'].replace('_', ' ').title(),
                'Rows': row.get('rows', 'N/A'),
                'Columns': len(row.get('columns', [])) if row.get('columns') else 'N/A',
                'Size (MB)': f"{row.get('size_mb', 0):.2f}" if row.get('size_mb') else 'N/A',
                'Last Modified': row.get('last_modified', 'N/A').strftime('%Y-%m-%d %H:%M') if hasattr(row.get('last_modified', ''), 'strftime') else 'N/A'
            })
        
        display_df = pd.DataFrame(display_data)
        st.dataframe(display_df, use_container_width=True)
    
    # Summary metrics
    if not success_files.empty:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Files", len(success_files))
        with col2:
            total_rows = sum(row.get('rows', 0) for _, row in success_files.iterrows())
            st.metric("Total Rows", f"{total_rows:,}")
        with col3:
            total_size = sum(row.get('size_mb', 0) for _, row in success_files.iterrows())
            st.metric("Total Size", f"{total_size:.2f} MB")

def create_data_sidebar(data_manager):
    """
    Create a sidebar for data management options
    
    Args:
        data_manager: DataManager instance
    """
    with st.sidebar:
        st.header("🗂️ Data Management")
        
        # Regenerate data options
        st.subheader("🔄 Regenerate Data")
        
        data_types = list(data_manager.data_files.keys())
        selected_data_type = st.selectbox(
            "Select data type to regenerate",
            ["All Data"] + [dt.replace('_', ' ').title() for dt in data_types]
        )
        
        if st.button("🔄 Regenerate Selected Data"):
            with st.spinner("Regenerating data..."):
                if selected_data_type == "All Data":
                    data_manager.regenerate_data()
                    st.success("All data regenerated successfully!")
                else:
                    # Convert back to data_type format
                    data_type = selected_data_type.lower().replace(' ', '_')
                    data_manager.regenerate_data(data_type)
                    st.success(f"{selected_data_type} data regenerated successfully!")
        
        # Data info
        st.subheader("📊 Data Information")
        if st.button("📈 Show Data Info"):
            data_info = data_manager.get_data_info()
            display_data_info(data_info)
        
        st.markdown("---")

def display_chart_with_data_download(chart, data, chart_title, data_title=None):
    """
    Display a chart with associated data download
    
    Args:
        chart: Plotly figure object
        data: pandas DataFrame used for the chart
        chart_title: Title for the chart
        data_title: Title for the data section (defaults to chart_title)
    """
    if data_title is None:
        data_title = chart_title
    
    # Display chart
    st.subheader(chart_title)
    st.plotly_chart(chart, use_container_width=True)
    
    # Data download section
    with st.expander(f"📥 Download {data_title} Data"):
        display_data_section(data, data_title, show_preview=False) 