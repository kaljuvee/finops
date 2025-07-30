import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os
import numpy as np

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

from data_generator import generate_cost_data, generate_service_breakdown

# Page configuration
st.set_page_config(
    page_title="Cost Monitoring - FinOps",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }
    .cost-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 1rem 0;
    }
    .service-card {
        background-color: #ffffff;
        border: 1px solid #e9ecef;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.title("📊 Cost Monitoring & Visualization")
    st.markdown("Real-time AWS cost monitoring and analysis dashboard")
    
    # Generate sample data
    cost_data = generate_cost_data(30)
    service_breakdown = generate_service_breakdown()
    
    # Sidebar filters
    with st.sidebar:
        st.header("🔧 Filters & Settings")
        
        # Date range selector
        date_range = st.selectbox(
            "Select Time Range",
            ["Last 7 days", "Last 30 days", "Last 90 days", "Custom range"]
        )
        
        if date_range == "Custom range":
            start_date = st.date_input("Start Date", datetime.now() - timedelta(days=30))
            end_date = st.date_input("End Date", datetime.now())
        else:
            days_map = {"Last 7 days": 7, "Last 30 days": 30, "Last 90 days": 90}
            days = days_map[date_range]
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
        
        # Service filter
        available_services = [
            "All Services", "Amazon EC2", "Amazon S3", "AWS Lambda", 
            "Amazon RDS", "Amazon CloudWatch", "Amazon VPC"
        ]
        selected_services = st.multiselect(
            "Filter by Service",
            available_services,
            default=["All Services"]
        )
        
        # Account filter (mock)
        accounts = st.multiselect(
            "Filter by Account",
            ["Production", "Development", "Staging"],
            default=["Production", "Development", "Staging"]
        )
        
        # Region filter
        regions = st.multiselect(
            "Filter by Region",
            ["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1"],
            default=["us-east-1", "us-west-2"]
        )
        
        st.markdown("---")
        
        # Export options
        st.subheader("📤 Export Options")
        if st.button("Export to CSV"):
            st.success("Cost data exported to CSV!")
        if st.button("Export to PDF"):
            st.success("Report exported to PDF!")
    
    # Main content area
    col1, col2, col3, col4 = st.columns(4)
    
    # Generate sample data
    cost_trend_data = cost_data  # Use the data already generated
    service_breakdown = generate_service_breakdown()
    
    # Calculate metrics
    total_cost = cost_trend_data['cost'].sum()
    avg_daily_cost = cost_trend_data['cost'].mean()
    max_daily_cost = cost_trend_data['cost'].max()
    cost_change = ((cost_trend_data['cost'].tail(7).mean() - 
                   cost_trend_data['cost'].head(7).mean()) / 
                  cost_trend_data['cost'].head(7).mean()) * 100
    
    # Display key metrics
    with col1:
        st.metric(
            label="💰 Total Cost (30 days)",
            value=f"${total_cost:,.2f}",
            delta=f"{cost_change:.1f}%"
        )
    
    with col2:
        st.metric(
            label="📈 Average Daily Cost",
            value=f"${avg_daily_cost:,.2f}",
            delta="Stable"
        )
    
    with col3:
        st.metric(
            label="📊 Peak Daily Cost",
            value=f"${max_daily_cost:,.2f}",
            delta="Within limits"
        )
    
    with col4:
        st.metric(
            label="🎯 Budget Utilization",
            value="78.5%",
            delta="-2.1%"
        )
    
    st.markdown("---")
    
    # Cost trend visualization
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📈 Cost Trend Analysis")
           # Daily cost trend chart
        fig_trend = px.line(
            cost_trend_data,
            x='date',
            y='cost',
            title="Daily Cost Trend (Last 30 Days)",
            labels={'cost': 'Daily Cost ($)', 'date': 'Date'}
        )
        fig_trend.update_layout(
            height=400,
            showlegend=False,
            xaxis_title="Date",
            yaxis_title="Cost ($)"
        )
        
        fig.update_traces(
            line=dict(color='#007bff', width=3),
            fill='tonexty',
            fillcolor='rgba(0, 123, 255, 0.1)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add trend insights
        st.markdown("**📊 Trend Insights:**")
        if cost_change > 0:
            st.info(f"📈 Costs have increased by {cost_change:.1f}% over the last week")
        else:
            st.success(f"📉 Costs have decreased by {abs(cost_change):.1f}% over the last week")
    
    with col2:
        st.subheader("🔧 Service Breakdown")
        
        # Service breakdown pie chart
        fig_pie = px.pie(
            service_breakdown.head(6), 
            values='cost', 
            names='Service',
            title="Top 6 Services by Cost"
        )
        
        fig_pie.update_layout(height=400)
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Service cost table
        st.markdown("**💰 Service Costs:**")
        for _, row in service_breakdown.head(5).iterrows():
            st.markdown(f"""
            <div class="service-card">
                <strong>{row['Service']}</strong><br>
                <span style="font-size: 1.2em; color: #007bff;">${row['cost']:,.2f}</span>
                <span style="float: right;">{row['Percentage']:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Detailed analysis section
    st.subheader("🔍 Detailed Cost Analysis")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Service Details", "📅 Daily Breakdown", "🏷️ Tag Analysis", "📈 Forecasting"])
    
    with tab1:
        st.markdown("**Service-wise Cost Analysis**")
        
        # Enhanced service breakdown table
        service_breakdown_enhanced = service_breakdown.copy()
        service_breakdown_enhanced['Daily Average'] = service_breakdown_enhanced['cost'] / 30
        service_breakdown_enhanced['Monthly Trend'] = ['↗️' if i % 2 == 0 else '↘️' for i in range(len(service_breakdown_enhanced))]
        
        st.dataframe(
            service_breakdown_enhanced[['Service', 'cost', 'Daily Average', 'percentage', 'Monthly Trend']],
            use_container_width=True,
            column_config={
                'cost': st.column_config.NumberColumn('Cost ($)', format='$%.2f'),
                'Daily Average': st.column_config.NumberColumn('Daily Avg ($)', format='$%.2f'),
                'percentage': st.column_config.NumberColumn('Percentage (%)', format='%.1f%%')
            }
        )
        
        # Service comparison chart
        fig_bar = px.bar(
            service_breakdown.head(8),
            x='Service',
            y='cost',
            title="Service Cost Comparison",
            color='cost',
            color_continuous_scale='Blues'
        )
        fig_bar.update_layout(height=400)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with tab2:
        st.markdown("**Daily Cost Breakdown**")
        
        # Daily cost analysis
        daily_stats = cost_trend_data.copy()
        daily_stats['Day of Week'] = daily_stats['Date'].dt.day_name()
        daily_stats['Week'] = daily_stats['Date'].dt.isocalendar().week
        
        # Weekly pattern analysis
        weekly_pattern = daily_stats.groupby('Day of Week')['cost'].mean().reset_index()
        weekly_pattern = weekly_pattern.reindex([6, 0, 1, 2, 3, 4, 5])  # Start with Monday
        
        fig_weekly = px.bar(
            weekly_pattern,
            x='Day of Week',
            y='cost',
            title="Average Cost by Day of Week",
            color='cost',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig_weekly, use_container_width=True)
        
        # Daily cost table
        st.markdown("**Recent Daily Costs:**")
        recent_costs = cost_trend_data.tail(10).copy()
        recent_costs['Change'] = recent_costs['cost'].pct_change() * 100
        recent_costs['Status'] = recent_costs['Change'].apply(
            lambda x: '📈' if x > 5 else '📉' if x < -5 else '➡️'
        )
        
        st.dataframe(
            recent_costs[['Date', 'cost', 'Change', 'Status']],
            use_container_width=True,
            column_config={
                'cost': st.column_config.NumberColumn('Cost ($)', format='$%.2f'),
                'Change': st.column_config.NumberColumn('Change (%)', format='%.1f%%')
            }
        )
    
    with tab3:
        st.markdown("**Cost Allocation by Tags**")
        
        # Mock tag-based cost allocation
        tag_data = pd.DataFrame({
            'Environment': ['Production', 'Development', 'Staging', 'Testing'],
            'cost': [8500, 2200, 1100, 800],
            'Resources': [45, 18, 12, 8]
        })
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_env = px.pie(
                tag_data,
                values='cost',
                names='Environment',
                title="Cost by Environment"
            )
            st.plotly_chart(fig_env, use_container_width=True)
        
        with col2:
            fig_resources = px.bar(
                tag_data,
                x='Environment',
                y='Resources',
                title="Resource Count by Environment",
                color='cost',
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig_resources, use_container_width=True)
        
        # Tag allocation table
        tag_data['Cost per Resource'] = tag_data['cost'] / tag_data['Resources']
        st.dataframe(
            tag_data,
            use_container_width=True,
            column_config={
                'cost': st.column_config.NumberColumn('Total Cost ($)', format='$%.2f'),
                'Cost per Resource': st.column_config.NumberColumn('Cost/Resource ($)', format='$%.2f')
            }
        )
    
    with tab4:
        st.markdown("**Cost Forecasting**")
        
        # Simple forecast calculation
        recent_trend = cost_trend_data.tail(7)['cost'].mean()
        forecast_days = 30
        
        # Generate forecast data
        forecast_dates = pd.date_range(
            start=cost_trend_data['Date'].max() + timedelta(days=1),
            periods=forecast_days,
            freq='D'
        )
        
        # Simple linear forecast with some variation
        base_forecast = recent_trend
        forecast_costs = []
        for i in range(forecast_days):
            # Add slight trend and random variation
            trend_factor = 1 + (i * 0.001)  # 0.1% daily increase
            variation = np.random.uniform(0.9, 1.1)
            forecast_cost = base_forecast * trend_factor * variation
            forecast_costs.append(forecast_cost)
        
        forecast_data = pd.DataFrame({
            'Date': forecast_dates,
            'cost': forecast_costs,
            'Type': 'Forecast'
        })
        
        # Combine historical and forecast
        historical_data = cost_trend_data.copy()
        historical_data['Type'] = 'Historical'
        
        combined_data = pd.concat([
            historical_data[['Date', 'cost', 'Type']],
            forecast_data
        ])
        
        # Forecast chart
        fig_forecast = px.line(
            combined_data,
            x='Date',
            y='cost',
            color='Type',
            title="30-Day Cost Forecast",
            color_discrete_map={'Historical': '#007bff', 'Forecast': '#ff6b6b'}
        )
        
        fig_forecast.update_layout(height=400)
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        # Forecast summary
        forecast_total = sum(forecast_costs)
        current_monthly = cost_trend_data['cost'].sum()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📈 Forecasted Monthly Cost", f"${forecast_total:,.2f}")
        with col2:
            st.metric("📊 Current Monthly Cost", f"${current_monthly:,.2f}")
        with col3:
            change_pct = ((forecast_total - current_monthly) / current_monthly) * 100
            st.metric("📈 Projected Change", f"{change_pct:+.1f}%")
    
    # Cost optimization insights
    st.markdown("---")
    st.subheader("💡 Cost Optimization Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-container">
            <h4>🎯 Right-sizing Opportunities</h4>
            <p>Identified <strong>12 EC2 instances</strong> with low utilization</p>
            <p><strong>Potential Savings:</strong> $450/month</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-container">
            <h4>💾 Storage Optimization</h4>
            <p>Found <strong>8 unused EBS volumes</strong></p>
            <p><strong>Potential Savings:</strong> $95/month</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-container">
            <h4>🔄 Reserved Instances</h4>
            <p>Opportunity for <strong>RDS reservations</strong></p>
            <p><strong>Potential Savings:</strong> $720/month</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

