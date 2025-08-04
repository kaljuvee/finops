import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os
import numpy as np

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from cost_monitor import CostMonitor
from data_manager import data_manager
from data_viewer import display_data_section, create_data_sidebar

# Page configuration
st.set_page_config(
    page_title="FinOps Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .alert-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🏦 FinOps Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("**Financial Operations for AWS Cost Management**")
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/200x100/1f77b4/ffffff?text=FinOps", width=200)
        st.markdown("### Navigation")
        st.markdown("Use the pages in the sidebar to explore different FinOps features:")
        st.markdown("- 📊 **Cost Monitoring**: Real-time cost visualization")
        st.markdown("- 🏷️ **Cost Allocation**: Tag management and allocation")
        st.markdown("- 🚨 **Anomaly Detection**: Cost anomaly alerts")
        st.markdown("- 💡 **Optimization**: Cost saving recommendations")
        st.markdown("- 📈 **Budget Management**: Budget tracking and forecasting")
        
        st.markdown("---")
        st.markdown("### Quick Stats")
        
        # Add data management sidebar
        create_data_sidebar(data_manager)
        
    # Initialize cost monitor
    cost_monitor = CostMonitor()
    
    # Get data from CSV files
    cost_data = data_manager.get_cost_data(30)
    service_breakdown = data_manager.get_service_breakdown()
    
    # Main dashboard content
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate metrics from generated data
    total_cost = cost_data['cost'].sum()
    daily_avg = cost_data['cost'].mean()
    
    with col1:
        st.metric(
            label="💰 Monthly Spend",
            value=f"${total_cost:,.2f}",
            delta="3.6%"
        )
    
    with col2:
        st.metric(
            label="📈 Daily Average",
            value=f"${daily_avg:,.2f}",
            delta="3.3%"
        )
    
    with col3:
        st.metric(
            label="🎯 Budget Utilization",
            value="71.9%",
            delta="4.6%"
        )
    
    with col4:
        st.metric(
            label="💡 Potential Savings",
            value="$1,366.19",
            delta="12.4%"
        )
    
    st.markdown("---")
    
    # Charts section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Monthly Spend Trend")
        
        # Use generated cost data
        fig = px.line(cost_data, x='date', y='cost', title="Daily AWS Spend")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Data download section
        with st.expander("📥 Download Cost Data"):
            display_data_section(cost_data, "Cost Data", "Daily AWS cost data with trends and seasonality")
    
    with col2:
        st.subheader("🔧 Service Breakdown")
        
        # Use generated service breakdown
        services = service_breakdown['Service'].tolist()
        costs = service_breakdown['cost'].tolist()
        
        fig = px.pie(values=costs, names=services, title="Cost by AWS Service")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Data download section
        with st.expander("📥 Download Service Breakdown Data"):
            display_data_section(service_breakdown, "Service Breakdown", "Cost breakdown by AWS service")
    
    # Alerts section
    st.markdown("---")
    st.subheader("🚨 Recent Alerts")
    
    alerts = [
        {"type": "warning", "message": "EC2 spend increased by 25% in the last 24 hours", "time": "2 hours ago"},
        {"type": "info", "message": "New optimization recommendation available for S3 storage", "time": "4 hours ago"},
        {"type": "success", "message": "Budget target met for Lambda functions", "time": "1 day ago"}
    ]
    
    for alert in alerts:
        icon = "⚠️" if alert["type"] == "warning" else "ℹ️" if alert["type"] == "info" else "✅"
        st.markdown(f"""
        <div class="alert-box">
            {icon} <strong>{alert['message']}</strong><br>
            <small>{alert['time']}</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Recent activity
    st.markdown("---")
    st.subheader("📋 Recent Activity")
    
    activity_data = pd.DataFrame({
        'Time': ['10:30 AM', '09:15 AM', '08:45 AM', '07:20 AM'],
        'Action': ['Cost anomaly detected', 'Budget alert triggered', 'Optimization applied', 'Tag policy updated'],
        'Service': ['EC2', 'S3', 'Lambda', 'All Services'],
        'Impact': ['$125 spike', '$50 over budget', '$30 saved', 'Compliance improved']
    })
    
    st.dataframe(activity_data, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("**FinOps MVP** - Built with Streamlit | Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M"))

if __name__ == "__main__":
    main()

