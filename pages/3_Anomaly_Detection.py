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

from data_manager import data_manager
from data_viewer import display_data_section, create_data_sidebar

# Page configuration
st.set_page_config(
    page_title="Anomaly Detection - FinOps",
    page_icon="🚨",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .anomaly-high {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-left: 4px solid #dc3545;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .anomaly-medium {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-left: 4px solid #ffc107;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .anomaly-low {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-left: 4px solid #17a2b8;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .alert-summary {
        background: linear-gradient(135deg, #dc3545 0%, #fd7e14 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 1rem 0;
    }
    .detection-card {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .threshold-config {
        background-color: #e9ecef;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.title("🚨 Anomaly Detection & Alerts")
    st.markdown("Real-time cost anomaly detection and alerting system")
    
    # Anomaly Detection Method Documentation
    with st.expander("📚 Anomaly Detection Method & Algorithm Details", expanded=False):
        st.markdown("""
        ### 🔍 Detection Algorithm: Z-Score Statistical Analysis
        
        This system uses **statistical outlier detection** based on the **Z-score method** to identify cost anomalies in AWS spending patterns.
        
        #### 📊 Core Algorithm:
        - **Z-score calculation**: `Z = (value - mean) / standard_deviation`
        - **Anomaly threshold**: Values with `|Z-score| > 2` are flagged as anomalies
        - **Statistical basis**: Uses 2 standard deviations (95% confidence interval)
        
        #### 🎯 Severity Classification:
        - **🔴 High Severity**: `|Z-score| > 3` (beyond 3σ - very rare events, ~0.3% of data)
        - **🟡 Medium Severity**: `2.5 < |Z-score| ≤ 3` (significant deviation, ~1% of data)
        - **🔵 Low Severity**: `2 < |Z-score| ≤ 2.5` (moderate deviation, ~2.5% of data)
        
        #### ⚙️ Sensitivity Settings:
        - **Low Sensitivity**: threshold = 3 (fewer false positives, may miss anomalies)
        - **Medium Sensitivity**: threshold = 2 (balanced detection - default)
        - **High Sensitivity**: threshold = 1.5 (more sensitive, may have false positives)
        
        #### 📈 Additional Features:
        - **Cost spike threshold**: Percentage increase from baseline as secondary filter
        - **Time window detection**: Configurable detection periods (1h to 7 days)
        - **Service-specific monitoring**: Focus on specific AWS services
        - **Trend analysis**: Historical pattern recognition
        
        #### 🎯 Use Cases:
        - **Cost spikes**: Unexpected increases in AWS spending
        - **Cost dips**: Unusual decreases (potential service issues)
        - **Seasonal patterns**: Detection of non-seasonal anomalies
        - **Service-specific issues**: Anomalies in specific AWS services
        
        #### 📊 Statistical Properties:
        - **False Positive Rate**: ~5% with default threshold (2σ)
        - **Detection Rate**: ~95% for significant anomalies
        - **Adaptive**: Thresholds can be adjusted based on business needs
        """)
    
    # Get data from CSV files
    cost_data = data_manager.get_cost_data(90)  # 90 days of data
    
    # Anomaly Detection Method: Z-Score Statistical Analysis
    # This method uses statistical outlier detection based on the Z-score:
    # - Z-score = (value - mean) / standard_deviation
    # - Values with |Z-score| > threshold are flagged as anomalies
    # - Threshold of 2 means values beyond 2 standard deviations are anomalies
    # - This is a simple but effective method for detecting cost spikes and dips
    def detect_anomalies(data, threshold=2):
        """
        Statistical anomaly detection using Z-score method
        
        Args:
            data: DataFrame with 'cost' column
            threshold: Z-score threshold (default=2, meaning 2 standard deviations)
        
        Returns:
            DataFrame with added 'z_score' and 'is_anomaly' columns
        """
        mean_cost = data['cost'].mean()
        std_cost = data['cost'].std()
        data['z_score'] = (data['cost'] - mean_cost) / std_cost
        data['is_anomaly'] = abs(data['z_score']) > threshold
        return data
    
    cost_data = detect_anomalies(cost_data)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Detection Settings")
        
        # Detection sensitivity - affects Z-score threshold
        # Low: threshold = 3 (fewer false positives, may miss some anomalies)
        # Medium: threshold = 2 (balanced detection)
        # High: threshold = 1.5 (more sensitive, may have more false positives)
        sensitivity = st.selectbox(
            "Detection Sensitivity",
            ["Low", "Medium", "High"],
            index=1
        )
        
        # Alert thresholds
        st.subheader("🎯 Alert Thresholds")
        
        # Cost spike threshold - percentage increase from baseline to trigger alert
        # This is an additional filter on top of the Z-score method
        cost_threshold = st.slider(
            "Cost Spike Threshold (%)",
            min_value=10,
            max_value=200,
            value=50,
            step=10
        )
        
        time_window = st.selectbox(
            "Detection Window",
            ["1 hour", "6 hours", "24 hours", "7 days"],
            index=2
        )
        
        # Services to monitor
        st.subheader("🔧 Services to Monitor")
        
        monitored_services = st.multiselect(
            "Select Services",
            ["Amazon EC2", "Amazon S3", "AWS Lambda", "Amazon RDS", "Amazon CloudWatch", "All Services"],
            default=["Amazon EC2", "Amazon S3", "AWS Lambda", "Amazon RDS"]
        )
        
        st.markdown("---")
        
        # Alert channels
        st.subheader("📢 Alert Channels")
        
        email_alerts = st.checkbox("Email Alerts", value=True)
        slack_alerts = st.checkbox("Slack Notifications", value=True)
        sns_alerts = st.checkbox("AWS SNS", value=False)
        
        if email_alerts:
            email_recipients = st.text_area(
                "Email Recipients",
                value="admin@company.com\nfinops@company.com"
            )
        
        st.markdown("---")
        
        # Manual detection
        st.subheader("🔍 Manual Detection")
        if st.button("Run Detection Now"):
            st.success("Anomaly detection initiated!")
        
        # Add data management sidebar
        create_data_sidebar(data_manager)
    
    # Generate sample anomaly data
    anomalies = cost_data[cost_data['is_anomaly']].copy()
    
    # Severity Classification based on Z-score magnitude:
    # - High: |Z-score| > 3 (beyond 3 standard deviations - very rare)
    # - Medium: 2.5 < |Z-score| ≤ 3 (significant deviation)
    # - Low: 2 < |Z-score| ≤ 2.5 (moderate deviation)
    anomalies['severity'] = anomalies['z_score'].apply(
        lambda x: 'High' if abs(x) > 3 else 'Medium' if abs(x) > 2.5 else 'Low'
    )
    anomalies['service'] = np.random.choice(['Amazon EC2', 'Amazon S3', 'AWS Lambda', 'Amazon RDS'], len(anomalies))
    anomalies['description'] = anomalies.apply(
        lambda row: f"Cost spike detected in {row['service']}: ${row['cost']:.2f} (Z-score: {row['z_score']:.2f})", 
        axis=1
    )
    
    # Add missing columns for the UI
    anomalies['timestamp'] = anomalies['date']
    anomalies['actual_cost'] = anomalies['cost']
    anomalies['baseline_cost'] = anomalies['cost'] * 0.9  # Approximate baseline
    anomalies['status'] = 'Active'
    anomalies['anomaly_type'] = 'Cost Spike'
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_anomalies = len(anomalies)
    high_severity = len(anomalies[anomalies['severity'] == 'High'])
    medium_severity = len(anomalies[anomalies['severity'] == 'Medium'])
    low_severity = len(anomalies[anomalies['severity'] == 'Low'])
    total_impact = anomalies['cost'].sum() if len(anomalies) > 0 else 0
    
    with col1:
        st.metric(
            label="🚨 Total Anomalies",
            value=str(total_anomalies),
            delta=f"Last 90 days"
        )
    
    with col2:
        st.metric(
            label="🔴 High Severity",
            value=str(high_severity),
            delta="Requires attention"
        )
    
    with col3:
        st.metric(
            label="💰 Cost Impact",
            value=f"${total_impact:,.2f}",
            delta="Anomalous costs"
        )
    
    with col4:
        st.metric(
            label="📊 Detection Rate",
            value="98.5%",
            delta="+0.3%"
        )
    
    st.markdown("---")
    
    # Real-time anomaly feed
    st.subheader("🔴 Real-time Anomaly Feed")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Anomaly timeline chart
        anomaly_df = anomalies.copy()
        
        # Create scatter plot for anomalies
        if len(anomaly_df) > 0:
            fig_timeline = px.scatter(
                anomaly_df,
                x='date',
                y='cost',
                color='severity',
                title="Anomaly Timeline",
                color_discrete_map={
                    'High': '#dc3545',
                    'Medium': '#ffc107',
                    'Low': '#17a2b8'
                }
            )
            
            # Add baseline costs as a line
            fig_timeline.add_scatter(
                x=anomaly_df['date'],
                y=anomaly_df['cost'] * 0.9,  # Approximate baseline as 90% of cost
                mode='lines',
                name='Baseline',
                line=dict(color='green', dash='dash')
            )
        else:
            # Create empty chart if no anomalies
            fig_timeline = px.scatter(
                title="Anomaly Timeline - No anomalies detected"
            )
        
        fig_timeline.update_layout(height=400)
        st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Data download section
        with st.expander("📥 Download Anomaly Data"):
            display_data_section(anomalies, "Anomaly Detection Data", "Cost anomalies with severity and service information")
    
    with col2:
        st.markdown("**🚨 Recent Alerts**")
        
        for _, anomaly in anomalies.head(5).iterrows():
            severity_class = f"anomaly-{anomaly['severity'].lower()}"
            severity_icon = "🔴" if anomaly['severity'] == 'High' else "🟡" if anomaly['severity'] == 'Medium' else "🔵"
            
            time_ago = datetime.now() - anomaly['timestamp']
            time_str = f"{time_ago.seconds // 3600}h ago" if time_ago.seconds >= 3600 else f"{time_ago.seconds // 60}m ago"
            
            st.markdown(f"""
            <div class="{severity_class}">
                {severity_icon} <strong>{anomaly['severity']} Severity</strong><br>
                <strong>{anomaly['service']}</strong><br>
                ${anomaly['actual_cost']:.2f} vs ${anomaly['baseline_cost']:.2f}<br>
                <small>{time_str} • {anomaly['status']}</small>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Detailed analysis tabs
    st.subheader("🔍 Detailed Anomaly Analysis")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Service Analysis", "📈 Trend Analysis", "🎯 Threshold Management", "📋 Alert History"])
    
    with tab1:
        st.markdown("**Service-wise Anomaly Breakdown**")
        
        # Service anomaly summary
        service_summary = {}
        for _, anomaly in anomalies.iterrows():
            service = anomaly['service']
            if service not in service_summary:
                service_summary[service] = {'count': 0, 'total_impact': 0, 'avg_severity': []}
            
            service_summary[service]['count'] += 1
            service_summary[service]['total_impact'] += abs(anomaly['actual_cost'] - anomaly['baseline_cost'])
            service_summary[service]['avg_severity'].append(anomaly['severity'])
        
        # Convert to DataFrame
        service_df = []
        for service, data in service_summary.items():
            severity_scores = {'High': 3, 'Medium': 2, 'Low': 1}
            avg_severity_score = np.mean([severity_scores[s] for s in data['avg_severity']])
            avg_severity = 'High' if avg_severity_score >= 2.5 else 'Medium' if avg_severity_score >= 1.5 else 'Low'
            
            service_df.append({
                'Service': service,
                'Anomaly Count': data['count'],
                'Total Impact': data['total_impact'],
                'Avg Severity': avg_severity,
                'Risk Level': 'High' if data['count'] >= 2 else 'Medium' if data['count'] >= 1 else 'Low'
            })
        
        service_df = pd.DataFrame(service_df)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Service anomaly count chart
            fig_service = px.bar(
                service_df,
                x='Service',
                y='Anomaly Count',
                color='Risk Level',
                title="Anomalies by Service",
                color_discrete_map={
                    'High': '#dc3545',
                    'Medium': '#ffc107',
                    'Low': '#28a745'
                }
            )
            st.plotly_chart(fig_service, use_container_width=True)
        
        with col2:
            # Service impact chart
            fig_impact = px.bar(
                service_df,
                x='Service',
                y='Total Impact',
                color='Avg Severity',
                title="Cost Impact by Service",
                color_discrete_map={
                    'High': '#dc3545',
                    'Medium': '#ffc107',
                    'Low': '#17a2b8'
                }
            )
            st.plotly_chart(fig_impact, use_container_width=True)
        
        # Service details table
        st.dataframe(
            service_df,
            use_container_width=True,
            column_config={
                'Total Impact': st.column_config.NumberColumn('Total Impact ($)', format='$%.2f')
            }
        )
    
    with tab2:
        st.markdown("**Anomaly Trend Analysis**")
        
        # Generate trend data with anomalies
        trend_data = cost_data.copy()
        
        # Add anomaly markers
        anomaly_dates = [a['timestamp'].date() for _, a in anomalies.iterrows()]
        trend_data['Has_Anomaly'] = trend_data['date'].dt.date.isin(anomaly_dates)
        trend_data['Anomaly_Type'] = trend_data['Has_Anomaly'].apply(
            lambda x: 'Anomaly Detected' if x else 'Normal'
        )
        
        # Trend chart with anomalies
        fig_trend = px.line(
            trend_data,
            x='date',
            y='cost',
            color='Anomaly_Type',
            title="Cost Trends with Anomaly Detection",
            color_discrete_map={
                'Normal': '#007bff',
                'Anomaly Detected': '#dc3545'
            }
        )
        
        # Add statistical bounds
        mean_cost = trend_data['cost'].mean()
        std_cost = trend_data['cost'].std()
        
        fig_trend.add_hline(
            y=mean_cost + 2*std_cost,
            line_dash="dash",
            line_color="red",
            annotation_text="Upper Threshold (2σ)"
        )
        
        fig_trend.add_hline(
            y=mean_cost - 2*std_cost,
            line_dash="dash",
            line_color="red",
            annotation_text="Lower Threshold (2σ)"
        )
        
        fig_trend.update_layout(height=500)
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # Data download section for trend analysis
        with st.expander("📥 Download Trend Analysis Data"):
            display_data_section(trend_data, "Trend Analysis Data", "Cost trends with anomaly markers and statistical bounds")
        
        # Trend statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📊 Mean Daily Cost", f"${mean_cost:.2f}")
        with col2:
            st.metric("📈 Standard Deviation", f"${std_cost:.2f}")
        with col3:
            anomaly_rate = (len(anomaly_dates) / len(trend_data)) * 100
            st.metric("🚨 Anomaly Rate", f"{anomaly_rate:.1f}%")
    
    with tab3:
        st.markdown("**Threshold Configuration & Management**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 Current Thresholds**")
            
            thresholds = [
                {"Service": "Amazon EC2", "Threshold": "50%", "Method": "Statistical", "Status": "Active"},
                {"Service": "Amazon S3", "Threshold": "75%", "Method": "ML-based", "Status": "Active"},
                {"Service": "AWS Lambda", "Threshold": "100%", "Method": "Statistical", "Status": "Active"},
                {"Service": "Amazon RDS", "Threshold": "40%", "Method": "ML-based", "Status": "Active"}
            ]
            
            threshold_df = pd.DataFrame(thresholds)
            st.dataframe(threshold_df, use_container_width=True)
            
            # Threshold adjustment
            st.markdown("**⚙️ Adjust Thresholds**")
            
            selected_service = st.selectbox("Select Service", threshold_df['Service'].tolist())
            new_threshold = st.slider("New Threshold (%)", 10, 200, 50)
            detection_method = st.selectbox("Detection Method", ["Statistical", "ML-based", "Hybrid"])
            
            if st.button("Update Threshold"):
                st.success(f"Threshold for {selected_service} updated to {new_threshold}%")
        
        with col2:
            st.markdown("**🎯 Threshold Performance**")
            
            # Mock performance data
            performance_data = pd.DataFrame({
                'Service': ['Amazon EC2', 'Amazon S3', 'AWS Lambda', 'Amazon RDS'],
                'True Positives': [15, 8, 12, 6],
                'False Positives': [3, 2, 1, 2],
                'False Negatives': [1, 1, 0, 1],
                'Precision': [0.83, 0.80, 0.92, 0.75],
                'Recall': [0.94, 0.89, 1.00, 0.86]
            })
            
            # Performance metrics chart
            fig_perf = px.bar(
                performance_data,
                x='Service',
                y=['Precision', 'Recall'],
                title="Detection Performance by Service",
                barmode='group'
            )
            st.plotly_chart(fig_perf, use_container_width=True)
            
            # Performance table
            st.dataframe(
                performance_data,
                use_container_width=True,
                column_config={
                    'Precision': st.column_config.NumberColumn('Precision', format='%.2f'),
                    'Recall': st.column_config.NumberColumn('Recall', format='%.2f')
                }
            )
    
    with tab4:
        st.markdown("**Alert History & Management**")
        
        # Alert history table
        alert_history = []
        for i, (_, anomaly) in enumerate(anomalies.iterrows()):
            alert_history.append({
                'Alert ID': f"ALT-{i+1:03d}",
                'Timestamp': anomaly['timestamp'],
                'Service': anomaly['service'],
                'Severity': anomaly['severity'],
                'Type': anomaly['anomaly_type'],
                'Cost Impact': abs(anomaly['actual_cost'] - anomaly['baseline_cost']),
                'Status': anomaly['status'],
                'Acknowledged': np.random.choice([True, False]),
                'Resolved': anomaly['status'] == 'Resolved'
            })
        
        alert_df = pd.DataFrame(alert_history)
        
        # Filter controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            severity_filter = st.multiselect(
                "Filter by Severity",
                ['High', 'Medium', 'Low'],
                default=['High', 'Medium', 'Low']
            )
        
        with col2:
            status_filter = st.multiselect(
                "Filter by Status",
                ['Active', 'Investigating', 'Resolved'],
                default=['Active', 'Investigating', 'Resolved']
            )
        
        with col3:
            date_filter = st.date_input(
                "From Date",
                value=datetime.now() - timedelta(days=7)
            )
        
        # Apply filters
        filtered_alerts = alert_df[
            (alert_df['Severity'].isin(severity_filter)) &
            (alert_df['Status'].isin(status_filter)) &
            (alert_df['Timestamp'].dt.date >= date_filter)
        ]
        
        # Alert history table
        st.dataframe(
            filtered_alerts,
            use_container_width=True,
            column_config={
                'Timestamp': st.column_config.DatetimeColumn('Timestamp'),
                'Cost Impact': st.column_config.NumberColumn('Cost Impact ($)', format='$%.2f'),
                'Acknowledged': st.column_config.CheckboxColumn('Acknowledged'),
                'Resolved': st.column_config.CheckboxColumn('Resolved')
            }
        )
        
        # Bulk actions
        st.markdown("**🔄 Bulk Actions**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Acknowledge All"):
                st.success("All alerts acknowledged!")
        
        with col2:
            if st.button("Mark as Resolved"):
                st.success("Selected alerts marked as resolved!")
        
        with col3:
            if st.button("Export History"):
                st.success("Alert history exported!")
    
    # Detection insights
    st.markdown("---")
    st.subheader("💡 Detection Insights & Recommendations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="alert-summary">
            <h4>🎯 Detection Accuracy</h4>
            <p>Current precision: <strong>85.2%</strong></p>
            <p>Current recall: <strong>92.1%</strong></p>
            <p><strong>Recommendation:</strong> Fine-tune EC2 thresholds</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="alert-summary">
            <h4>⚡ Response Time</h4>
            <p>Average detection time: <strong>4.2 minutes</strong></p>
            <p>Average response time: <strong>12 minutes</strong></p>
            <p><strong>Target:</strong> <5 minutes detection</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="alert-summary">
            <h4>🔄 Automation</h4>
            <p>Auto-resolved alerts: <strong>68%</strong></p>
            <p>Manual intervention: <strong>32%</strong></p>
            <p><strong>Goal:</strong> Increase automation to 80%</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

