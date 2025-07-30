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

from budget_manager import BudgetManager
from data_generator import generate_budget_data, generate_forecast_data

# Page configuration
st.set_page_config(
    page_title="Budget Management - FinOps",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .budget-on-track {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-left: 4px solid #28a745;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .budget-at-risk {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-left: 4px solid #ffc107;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .budget-exceeded {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-left: 4px solid #dc3545;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .budget-summary {
        background: linear-gradient(135deg, #007bff 0%, #6610f2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 1rem 0;
    }
    .forecast-card {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .budget-form {
        background-color: #e9ecef;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.title("📈 Budget Management & Forecasting")
    st.markdown("Comprehensive budget tracking, forecasting, and variance analysis")
    
    # Initialize budget manager
    budget_manager = BudgetManager()
    
    # Sidebar for budget management
    with st.sidebar:
        st.header("💼 Budget Controls")
        
        # Budget period selector
        budget_period = st.selectbox(
            "Budget Period",
            ["Monthly", "Quarterly", "Annual"],
            index=0
        )
        
        # Budget filters
        st.subheader("🔍 Filters")
        
        budget_status = st.multiselect(
            "Budget Status",
            ["On Track", "At Risk", "Exceeded"],
            default=["On Track", "At Risk", "Exceeded"]
        )
        
        budget_type = st.multiselect(
            "Budget Type",
            ["Department", "Project", "Service", "Environment"],
            default=["Department", "Project"]
        )
        
        st.markdown("---")
        
        # Quick budget creation
        st.subheader("➕ Quick Budget Creation")
        
        with st.form("quick_budget"):
            budget_name = st.text_input("Budget Name")
            budget_amount = st.number_input("Budget Amount ($)", min_value=0.0, value=5000.0)
            budget_category = st.selectbox("Category", ["Department", "Project", "Service", "Environment"])
            
            submitted = st.form_submit_button("Create Budget")
            if submitted and budget_name and budget_amount > 0:
                st.success(f"Budget '{budget_name}' created successfully!")
        
        st.markdown("---")
        
        # Alert settings
        st.subheader("🚨 Alert Settings")
        
        warning_threshold = st.slider("Warning Threshold (%)", 50, 90, 75)
        critical_threshold = st.slider("Critical Threshold (%)", 80, 100, 90)
        
        email_alerts = st.checkbox("Email Alerts", value=True)
        slack_alerts = st.checkbox("Slack Notifications", value=True)
        
        st.markdown("---")
        
        # Export options
        st.subheader("📤 Export & Reports")
        
        if st.button("📊 Generate Budget Report"):
            st.success("Budget report generated!")
        
        if st.button("📈 Export Forecast"):
            st.success("Forecast data exported!")
        
        if st.button("📧 Email Summary"):
            st.success("Budget summary emailed!")
    
    # Generate sample data
    budget_data = generate_budget_data()
    forecast_data = generate_forecast_data(30)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="💰 Total Budget",
            value=f"${budget_data['total_budget']:,.2f}",
            delta="Monthly"
        )
    
    with col2:
        st.metric(
            label="💸 Total Spent",
            value=f"${budget_data['total_spent']:,.2f}",
            delta=f"{budget_data['overall_utilization']:.1f}% utilized"
        )
    
    with col3:
        st.metric(
            label="💵 Remaining",
            value=f"${budget_data['total_remaining']:,.2f}",
            delta=f"{100 - budget_data['overall_utilization']:.1f}% left"
        )
    
    with col4:
        # Calculate forecast vs budget
        forecast_total = forecast_data[forecast_data['Type'] == 'Forecast']['Cost'].sum()
        forecast_vs_budget = ((forecast_total - budget_data['total_budget']) / budget_data['total_budget']) * 100
        
        st.metric(
            label="📊 Forecast vs Budget",
            value=f"{forecast_vs_budget:+.1f}%",
            delta="Next 30 days"
        )
    
    st.markdown("---")
    
    # Budget overview
    st.subheader("📊 Budget Overview")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Budget utilization chart
        budget_df = pd.DataFrame(budget_data['budgets'])
        
        fig_utilization = px.bar(
            budget_df,
            x='name',
            y=['spent_amount', 'remaining_amount'],
            title="Budget Utilization by Department",
            labels={'value': 'Amount ($)', 'name': 'Budget'},
            color_discrete_map={
                'spent_amount': '#007bff',
                'remaining_amount': '#6c757d'
            }
        )
        
        fig_utilization.update_layout(height=400)
        st.plotly_chart(fig_utilization, use_container_width=True)
    
    with col2:
        st.markdown("**🎯 Budget Status Summary**")
        
        for budget in budget_data['budgets']:
            if budget['status'] == 'On Track':
                css_class = "budget-on-track"
                icon = "🟢"
            elif budget['status'] == 'At Risk':
                css_class = "budget-at-risk"
                icon = "🟡"
            else:
                css_class = "budget-exceeded"
                icon = "🔴"
            
            st.markdown(f"""
            <div class="{css_class}">
                {icon} <strong>{budget['name']}</strong><br>
                ${budget['spent_amount']:,.2f} / ${budget['budget_amount']:,.2f}<br>
                <strong>{budget['utilization']:.1f}% utilized</strong><br>
                <small>Status: {budget['status']}</small>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Detailed analysis tabs
    st.subheader("🔍 Detailed Budget Analysis")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Budget Details", "📈 Forecasting", "⚠️ Variance Analysis", "🚨 Alerts & Notifications", "📋 Budget Planning"])
    
    with tab1:
        st.markdown("**Detailed Budget Breakdown**")
        
        # Enhanced budget table
        budget_df_enhanced = budget_df.copy()
        budget_df_enhanced['variance'] = budget_df_enhanced['forecast'] - budget_df_enhanced['budget_amount']
        budget_df_enhanced['variance_pct'] = (budget_df_enhanced['variance'] / budget_df_enhanced['budget_amount']) * 100
        budget_df_enhanced['days_remaining'] = 30 - datetime.now().day
        budget_df_enhanced['daily_burn_rate'] = budget_df_enhanced['spent_amount'] / datetime.now().day
        budget_df_enhanced['projected_end_date'] = budget_df_enhanced.apply(
            lambda row: datetime.now() + timedelta(days=row['remaining_amount'] / row['daily_burn_rate']) 
            if row['daily_burn_rate'] > 0 else None, axis=1
        )
        
        st.dataframe(
            budget_df_enhanced[[
                'name', 'budget_amount', 'spent_amount', 'remaining_amount', 
                'utilization', 'forecast', 'variance', 'variance_pct', 
                'daily_burn_rate', 'status'
            ]],
            use_container_width=True,
            column_config={
                'name': 'Budget Name',
                'budget_amount': st.column_config.NumberColumn('Budget ($)', format='$%.2f'),
                'spent_amount': st.column_config.NumberColumn('Spent ($)', format='$%.2f'),
                'remaining_amount': st.column_config.NumberColumn('Remaining ($)', format='$%.2f'),
                'utilization': st.column_config.NumberColumn('Utilization (%)', format='%.1f%%'),
                'forecast': st.column_config.NumberColumn('Forecast ($)', format='$%.2f'),
                'variance': st.column_config.NumberColumn('Variance ($)', format='$%.2f'),
                'variance_pct': st.column_config.NumberColumn('Variance (%)', format='%.1f%%'),
                'daily_burn_rate': st.column_config.NumberColumn('Daily Burn ($)', format='$%.2f')
            }
        )
        
        # Budget trends
        st.markdown("**📈 Budget Trends**")
        
        # Generate trend data for each budget
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
        trend_data = []
        
        for budget in budget_data['budgets']:
            daily_spend = budget['spent_amount'] / len(dates)
            cumulative = 0
            
            for i, date in enumerate(dates):
                cumulative += daily_spend * np.random.uniform(0.8, 1.2)  # Add variation
                trend_data.append({
                    'Date': date,
                    'Budget': budget['name'],
                    'Cumulative Spend': cumulative,
                    'Budget Limit': budget['budget_amount']
                })
        
        trend_df = pd.DataFrame(trend_data)
        
        fig_trends = px.line(
            trend_df,
            x='Date',
            y='Cumulative Spend',
            color='Budget',
            title="Budget Spend Trends (30 Days)"
        )
        
        # Add budget limits as horizontal lines
        for budget in budget_data['budgets']:
            fig_trends.add_hline(
                y=budget['budget_amount'],
                line_dash="dash",
                annotation_text=f"{budget['name']} Limit"
            )
        
        st.plotly_chart(fig_trends, use_container_width=True)
    
    with tab2:
        st.markdown("**Cost Forecasting & Projections**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Forecast chart
            fig_forecast = px.line(
                forecast_data,
                x='Date',
                y='Cost',
                color='Type',
                title="30-Day Cost Forecast",
                color_discrete_map={
                    'Historical': '#007bff',
                    'Forecast': '#28a745'
                }
            )
            
            fig_forecast.update_layout(height=400)
            st.plotly_chart(fig_forecast, use_container_width=True)
        
        with col2:
            # Forecast accuracy metrics
            st.markdown("**🎯 Forecast Accuracy**")
            
            accuracy_metrics = {
                'Last Month Accuracy': '94.2%',
                'Average Error': '$127',
                'Confidence Interval': '±8.5%',
                'Model Type': 'Polynomial Regression'
            }
            
            for metric, value in accuracy_metrics.items():
                st.metric(metric, value)
            
            # Forecast scenarios
            st.markdown("**📊 Forecast Scenarios**")
            
            scenarios = pd.DataFrame({
                'Scenario': ['Conservative', 'Most Likely', 'Optimistic'],
                'Forecast': [12500, 11800, 11200],
                'Probability': ['25%', '50%', '25%']
            })
            
            st.dataframe(scenarios, use_container_width=True)
        
        # Monthly forecast breakdown
        st.markdown("**📅 Monthly Forecast Breakdown**")
        
        # Generate monthly forecast data
        monthly_forecast = []
        base_cost = 11000
        
        for i in range(12):
            month_name = (datetime.now() + timedelta(days=30*i)).strftime('%B %Y')
            trend_factor = 1 + (i * 0.02)  # 2% monthly growth
            seasonal_factor = 1 + 0.1 * np.sin(i * np.pi / 6)  # Seasonal variation
            forecast_cost = base_cost * trend_factor * seasonal_factor
            
            monthly_forecast.append({
                'Month': month_name,
                'Forecast': forecast_cost,
                'Budget': 12000,
                'Variance': forecast_cost - 12000,
                'Confidence': np.random.uniform(85, 95)
            })
        
        monthly_df = pd.DataFrame(monthly_forecast)
        
        fig_monthly = px.bar(
            monthly_df,
            x='Month',
            y=['Forecast', 'Budget'],
            title="12-Month Budget vs Forecast",
            barmode='group'
        )
        
        st.plotly_chart(fig_monthly, use_container_width=True)
        
        # Monthly forecast table
        st.dataframe(
            monthly_df,
            use_container_width=True,
            column_config={
                'Forecast': st.column_config.NumberColumn('Forecast ($)', format='$%.2f'),
                'Budget': st.column_config.NumberColumn('Budget ($)', format='$%.2f'),
                'Variance': st.column_config.NumberColumn('Variance ($)', format='$%.2f'),
                'Confidence': st.column_config.NumberColumn('Confidence (%)', format='%.1f%%')
            }
        )
    
    with tab3:
        st.markdown("**Budget Variance Analysis**")
        
        # Variance summary
        col1, col2, col3 = st.columns(3)
        
        total_variance = sum(b['forecast'] - b['budget_amount'] for b in budget_data['budgets'])
        avg_variance_pct = np.mean([(b['forecast'] - b['budget_amount']) / b['budget_amount'] * 100 for b in budget_data['budgets']])
        
        with col1:
            st.metric("💰 Total Variance", f"${total_variance:,.2f}")
        with col2:
            st.metric("📊 Average Variance", f"{avg_variance_pct:+.1f}%")
        with col3:
            over_budget_count = sum(1 for b in budget_data['budgets'] if b['utilization'] > 100)
            st.metric("🚨 Over Budget Count", str(over_budget_count))
        
        # Variance analysis chart
        variance_data = []
        for budget in budget_data['budgets']:
            variance_data.append({
                'Budget': budget['name'],
                'Actual': budget['spent_amount'],
                'Budget': budget['budget_amount'],
                'Forecast': budget['forecast'],
                'Variance': budget['forecast'] - budget['budget_amount'],
                'Variance %': ((budget['forecast'] - budget['budget_amount']) / budget['budget_amount']) * 100
            })
        
        variance_df = pd.DataFrame(variance_data)
        
        fig_variance = px.bar(
            variance_df,
            x='Budget',
            y='Variance %',
            title="Budget Variance by Department",
            color='Variance %',
            color_continuous_scale='RdYlGn_r'
        )
        
        fig_variance.add_hline(y=0, line_dash="dash", line_color="black")
        st.plotly_chart(fig_variance, use_container_width=True)
        
        # Variance details table
        st.dataframe(
            variance_df,
            use_container_width=True,
            column_config={
                'Actual': st.column_config.NumberColumn('Actual ($)', format='$%.2f'),
                'Budget': st.column_config.NumberColumn('Budget ($)', format='$%.2f'),
                'Forecast': st.column_config.NumberColumn('Forecast ($)', format='$%.2f'),
                'Variance': st.column_config.NumberColumn('Variance ($)', format='$%.2f'),
                'Variance %': st.column_config.NumberColumn('Variance (%)', format='%.1f%%')
            }
        )
        
        # Root cause analysis
        st.markdown("**🔍 Variance Root Cause Analysis**")
        
        root_causes = [
            {"Budget": "Production Environment", "Variance": "+$450", "Root Cause": "Increased EC2 usage due to traffic spike", "Action": "Implement auto-scaling"},
            {"Budget": "Development Environment", "Variance": "+$100", "Root Cause": "Extended testing period", "Action": "Optimize test schedules"},
            {"Budget": "Data Analytics", "Variance": "-$200", "Root Cause": "Delayed project start", "Action": "Reallocate budget"}
        ]
        
        for cause in root_causes:
            st.markdown(f"""
            <div class="forecast-card">
                <strong>{cause['Budget']}</strong> ({cause['Variance']})<br>
                <strong>Root Cause:</strong> {cause['Root Cause']}<br>
                <strong>Recommended Action:</strong> {cause['Action']}
            </div>
            """, unsafe_allow_html=True)
    
    with tab4:
        st.markdown("**Budget Alerts & Notifications**")
        
        # Alert summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🚨 Active Alerts", "3")
        with col2:
            st.metric("⚠️ Warnings", "2")
        with col3:
            st.metric("📧 Notifications Sent", "15")
        
        # Recent alerts
        st.markdown("**🔔 Recent Alerts**")
        
        alerts = [
            {
                "timestamp": datetime.now() - timedelta(hours=2),
                "budget": "Production Environment",
                "type": "Critical",
                "message": "Budget exceeded 90% threshold",
                "status": "Active"
            },
            {
                "timestamp": datetime.now() - timedelta(hours=6),
                "budget": "Development Environment",
                "type": "Warning",
                "message": "Budget reached 75% threshold",
                "status": "Acknowledged"
            },
            {
                "timestamp": datetime.now() - timedelta(days=1),
                "budget": "Data Analytics",
                "type": "Info",
                "message": "Budget forecast updated",
                "status": "Resolved"
            }
        ]
        
        for alert in alerts:
            alert_class = "budget-exceeded" if alert["type"] == "Critical" else "budget-at-risk" if alert["type"] == "Warning" else "budget-on-track"
            icon = "🔴" if alert["type"] == "Critical" else "🟡" if alert["type"] == "Warning" else "🔵"
            
            time_ago = datetime.now() - alert["timestamp"]
            time_str = f"{time_ago.seconds // 3600}h ago" if time_ago.seconds >= 3600 else f"{time_ago.seconds // 60}m ago"
            
            st.markdown(f"""
            <div class="{alert_class}">
                {icon} <strong>{alert['type']} Alert</strong><br>
                <strong>{alert['budget']}</strong><br>
                {alert['message']}<br>
                <small>{time_str} • {alert['status']}</small>
            </div>
            """, unsafe_allow_html=True)
        
        # Alert configuration
        st.markdown("**⚙️ Alert Configuration**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 Threshold Settings**")
            
            thresholds = pd.DataFrame({
                'Budget': ['Production Environment', 'Development Environment', 'Data Analytics'],
                'Warning (%)': [75, 80, 70],
                'Critical (%)': [90, 95, 85],
                'Forecast Alert': [True, True, False]
            })
            
            st.dataframe(thresholds, use_container_width=True)
        
        with col2:
            st.markdown("**📧 Notification Settings**")
            
            notification_settings = {
                'Email Recipients': 'admin@company.com, finance@company.com',
                'Slack Channel': '#finops-alerts',
                'SMS Alerts': 'Enabled for Critical only',
                'Frequency': 'Immediate + Daily Summary'
            }
            
            for setting, value in notification_settings.items():
                st.markdown(f"**{setting}:** {value}")
    
    with tab5:
        st.markdown("**Budget Planning & Management**")
        
        # Budget planning form
        st.markdown("**📋 Create New Budget**")
        
        with st.form("budget_planning"):
            col1, col2 = st.columns(2)
            
            with col1:
                new_budget_name = st.text_input("Budget Name")
                new_budget_amount = st.number_input("Budget Amount ($)", min_value=0.0, value=10000.0)
                new_budget_period = st.selectbox("Period", ["Monthly", "Quarterly", "Annual"])
                new_budget_category = st.selectbox("Category", ["Department", "Project", "Service", "Environment"])
            
            with col2:
                new_budget_owner = st.text_input("Budget Owner")
                new_budget_description = st.text_area("Description")
                new_budget_tags = st.text_input("Tags (comma-separated)")
                auto_alerts = st.checkbox("Enable Auto Alerts", value=True)
            
            # Advanced settings
            st.markdown("**⚙️ Advanced Settings**")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                warning_threshold_new = st.slider("Warning Threshold (%)", 50, 90, 75, key="new_warning")
            with col2:
                critical_threshold_new = st.slider("Critical Threshold (%)", 80, 100, 90, key="new_critical")
            with col3:
                rollover_enabled = st.checkbox("Enable Budget Rollover")
            
            submitted = st.form_submit_button("Create Budget", type="primary")
            
            if submitted and new_budget_name and new_budget_amount > 0:
                st.success(f"Budget '{new_budget_name}' created successfully!")
                st.balloons()
        
        # Budget templates
        st.markdown("**📄 Budget Templates**")
        
        templates = [
            {"name": "Department Budget", "amount": "$15,000", "period": "Monthly", "description": "Standard department budget template"},
            {"name": "Project Budget", "amount": "$50,000", "period": "Quarterly", "description": "Project-based budget template"},
            {"name": "Environment Budget", "amount": "$8,000", "period": "Monthly", "description": "Environment-specific budget template"}
        ]
        
        col1, col2, col3 = st.columns(3)
        
        for i, template in enumerate(templates):
            with [col1, col2, col3][i]:
                st.markdown(f"""
                <div class="forecast-card">
                    <h4>{template['name']}</h4>
                    <p><strong>Amount:</strong> {template['amount']}</p>
                    <p><strong>Period:</strong> {template['period']}</p>
                    <p>{template['description']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button(f"Use Template", key=f"template_{i}"):
                    st.info(f"Template '{template['name']}' loaded!")
        
        # Budget approval workflow
        st.markdown("**✅ Budget Approval Workflow**")
        
        pending_budgets = [
            {"name": "Q2 Marketing Budget", "amount": "$25,000", "requestor": "Marketing Team", "status": "Pending Approval"},
            {"name": "Infrastructure Upgrade", "amount": "$40,000", "requestor": "DevOps Team", "status": "Under Review"},
            {"name": "Data Science Tools", "amount": "$15,000", "requestor": "Data Team", "status": "Approved"}
        ]
        
        for budget in pending_budgets:
            status_class = "budget-on-track" if budget["status"] == "Approved" else "budget-at-risk"
            
            st.markdown(f"""
            <div class="{status_class}">
                <strong>{budget['name']}</strong> - {budget['amount']}<br>
                <strong>Requestor:</strong> {budget['requestor']}<br>
                <strong>Status:</strong> {budget['status']}
            </div>
            """, unsafe_allow_html=True)
    
    # Budget insights
    st.markdown("---")
    st.subheader("💡 Budget Insights & Recommendations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="budget-summary">
            <h4>📊 Spending Pattern</h4>
            <p>Peak spending occurs in <strong>week 3</strong> of each month</p>
            <p><strong>Recommendation:</strong> Implement spending controls</p>
            <p><strong>Potential Impact:</strong> 15% cost reduction</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="budget-summary">
            <h4>🎯 Budget Accuracy</h4>
            <p>Forecast accuracy: <strong>94.2%</strong></p>
            <p>Average variance: <strong>±5.8%</strong></p>
            <p><strong>Goal:</strong> Achieve 96% accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="budget-summary">
            <h4>🔄 Optimization</h4>
            <p>Identified <strong>3 budget reallocation</strong> opportunities</p>
            <p><strong>Potential Savings:</strong> $2,400/month</p>
            <p><strong>Action:</strong> Review and reallocate unused budgets</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

