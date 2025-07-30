import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

from data_generator import generate_tag_allocation_data

# Page configuration
st.set_page_config(
    page_title="Cost Allocation - FinOps",
    page_icon="🏷️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .tag-card {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .allocation-summary {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 1rem 0;
    }
    .tag-policy {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .compliance-good {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .compliance-warning {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .compliance-danger {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.title("🏷️ Cost Allocation & Tag Management")
    st.markdown("Manage cost allocation through tagging policies and track spending by business units")
    
    # Sidebar for tag management
    with st.sidebar:
        st.header("🔧 Tag Management")
        
        # Tag policy configuration
        st.subheader("📋 Tag Policies")
        
        required_tags = st.multiselect(
            "Required Tags",
            ["Environment", "Team", "Project", "CostCenter", "Owner"],
            default=["Environment", "Team", "Project"]
        )
        
        st.subheader("➕ Create New Tag")
        new_tag_key = st.text_input("Tag Key")
        new_tag_value = st.text_input("Tag Value")
        
        if st.button("Add Tag"):
            if new_tag_key and new_tag_value:
                st.success(f"Tag {new_tag_key}:{new_tag_value} added successfully!")
            else:
                st.error("Please provide both tag key and value")
        
        st.markdown("---")
        
        # Bulk tag operations
        st.subheader("🔄 Bulk Operations")
        
        bulk_action = st.selectbox(
            "Select Action",
            ["Apply tags to untagged resources", "Update existing tags", "Remove obsolete tags"]
        )
        
        if st.button("Execute Bulk Action"):
            st.info(f"Executing: {bulk_action}")
            st.success("Bulk operation completed successfully!")
        
        st.markdown("---")
        
        # Export options
        st.subheader("📤 Export")
        if st.button("Export Allocation Report"):
            st.success("Allocation report exported!")
    
    # Main content
    # Generate sample allocation data
    allocation_data = generate_tag_allocation_data()
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_allocated = allocation_data['Cost'].sum()
    environment_costs = allocation_data[allocation_data['Tag_Type'] == 'Environment']['Cost'].sum()
    team_costs = allocation_data[allocation_data['Tag_Type'] == 'Team']['Cost'].sum()
    project_costs = allocation_data[allocation_data['Tag_Type'] == 'Project']['Cost'].sum()
    
    with col1:
        st.metric(
            label="💰 Total Allocated",
            value=f"${total_allocated:,.2f}",
            delta="100% tagged"
        )
    
    with col2:
        st.metric(
            label="🌍 Environment Allocation",
            value=f"${environment_costs:,.2f}",
            delta="4 environments"
        )
    
    with col3:
        st.metric(
            label="👥 Team Allocation",
            value=f"${team_costs:,.2f}",
            delta="5 teams"
        )
    
    with col4:
        st.metric(
            label="📊 Project Allocation",
            value=f"${project_costs:,.2f}",
            delta="4 projects"
        )
    
    st.markdown("---")
    
    # Tag compliance overview
    st.subheader("📊 Tag Compliance Overview")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Mock compliance data
        compliance_data = pd.DataFrame({
            'Service': ['EC2', 'S3', 'Lambda', 'RDS', 'EBS', 'CloudWatch'],
            'Total Resources': [150, 89, 45, 12, 78, 25],
            'Tagged Resources': [142, 85, 45, 12, 65, 20],
            'Compliance %': [94.7, 95.5, 100.0, 100.0, 83.3, 80.0]
        })
        
        # Compliance chart
        fig_compliance = px.bar(
            compliance_data,
            x='Service',
            y='Compliance %',
            title="Tag Compliance by Service",
            color='Compliance %',
            color_continuous_scale='RdYlGn',
            range_color=[0, 100]
        )
        
        fig_compliance.update_layout(height=400)
        fig_compliance.add_hline(y=90, line_dash="dash", line_color="red", 
                               annotation_text="Target: 90%")
        
        st.plotly_chart(fig_compliance, use_container_width=True)
    
    with col2:
        st.markdown("**🎯 Compliance Status**")
        
        for _, row in compliance_data.iterrows():
            compliance = row['Compliance %']
            if compliance >= 95:
                css_class = "compliance-good"
                icon = "✅"
            elif compliance >= 85:
                css_class = "compliance-warning"
                icon = "⚠️"
            else:
                css_class = "compliance-danger"
                icon = "❌"
            
            st.markdown(f"""
            <div class="{css_class}">
                {icon} <strong>{row['Service']}</strong><br>
                {row['Tagged Resources']}/{row['Total Resources']} resources<br>
                <strong>{compliance:.1f}% compliant</strong>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Cost allocation analysis
    st.subheader("💰 Cost Allocation Analysis")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🌍 Environment", "👥 Team", "📊 Project", "🏢 Cost Center"])
    
    with tab1:
        st.markdown("**Environment-based Cost Allocation**")
        
        env_data = allocation_data[allocation_data['Tag_Type'] == 'Environment'].copy()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Environment pie chart
            fig_env = px.pie(
                env_data,
                values='Cost',
                names='Tag_Value',
                title="Cost Distribution by Environment"
            )
            st.plotly_chart(fig_env, use_container_width=True)
        
        with col2:
            # Environment cost breakdown
            env_data['Monthly Budget'] = [8000, 2500, 1500, 1000]
            env_data['Budget Utilization'] = (env_data['Cost'] / env_data['Monthly Budget']) * 100
            env_data['Status'] = env_data['Budget Utilization'].apply(
                lambda x: '🟢 On Track' if x < 80 else '🟡 At Risk' if x < 100 else '🔴 Over Budget'
            )
            
            st.dataframe(
                env_data[['Tag_Value', 'Cost', 'Monthly Budget', 'Budget Utilization', 'Status']],
                use_container_width=True,
                column_config={
                    'Tag_Value': 'Environment',
                    'Cost': st.column_config.NumberColumn('Current Cost ($)', format='$%.2f'),
                    'Monthly Budget': st.column_config.NumberColumn('Budget ($)', format='$%.2f'),
                    'Budget Utilization': st.column_config.NumberColumn('Utilization (%)', format='%.1f%%')
                }
            )
        
        # Environment trends
        st.markdown("**📈 Environment Cost Trends**")
        
        # Mock trend data
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
        trend_data = []
        
        for env in env_data['Tag_Value']:
            base_cost = env_data[env_data['Tag_Value'] == env]['Cost'].iloc[0] / 30
            for date in dates:
                variation = np.random.uniform(0.8, 1.2)
                trend_data.append({
                    'Date': date,
                    'Environment': env,
                    'Daily Cost': base_cost * variation
                })
        
        trend_df = pd.DataFrame(trend_data)
        
        fig_trend = px.line(
            trend_df,
            x='Date',
            y='Daily Cost',
            color='Environment',
            title="Daily Cost Trends by Environment"
        )
        
        st.plotly_chart(fig_trend, use_container_width=True)
    
    with tab2:
        st.markdown("**Team-based Cost Allocation**")
        
        team_data = allocation_data[allocation_data['Tag_Type'] == 'Team'].copy()
        
        # Team allocation sunburst chart
        fig_sunburst = px.sunburst(
            team_data,
            path=['Tag_Type', 'Tag_Value'],
            values='Cost',
            title="Team Cost Allocation"
        )
        
        st.plotly_chart(fig_sunburst, use_container_width=True)
        
        # Team performance metrics
        team_data['Team Lead'] = ['Alice Johnson', 'Bob Smith', 'Carol Davis', 'David Wilson', 'Eva Brown']
        team_data['Resources'] = [25, 18, 22, 15, 12]
        team_data['Cost per Resource'] = team_data['Cost'] / team_data['Resources']
        team_data['Efficiency Score'] = np.random.uniform(7.5, 9.5, len(team_data))
        
        st.markdown("**👥 Team Performance Dashboard**")
        
        st.dataframe(
            team_data[['Tag_Value', 'Team Lead', 'Cost', 'Resources', 'Cost per Resource', 'Efficiency Score']],
            use_container_width=True,
            column_config={
                'Tag_Value': 'Team',
                'Cost': st.column_config.NumberColumn('Total Cost ($)', format='$%.2f'),
                'Cost per Resource': st.column_config.NumberColumn('Cost/Resource ($)', format='$%.2f'),
                'Efficiency Score': st.column_config.NumberColumn('Efficiency', format='%.1f/10')
            }
        )
    
    with tab3:
        st.markdown("**Project-based Cost Allocation**")
        
        project_data = allocation_data[allocation_data['Tag_Type'] == 'Project'].copy()
        
        # Project timeline and costs
        project_data['Start Date'] = ['2024-01-15', '2024-02-01', '2024-03-10', '2024-01-01']
        project_data['End Date'] = ['2024-06-30', '2024-08-15', '2024-12-31', '2024-12-31']
        project_data['Status'] = ['Active', 'Active', 'Planning', 'Ongoing']
        project_data['Progress'] = [65, 40, 15, 80]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Project cost comparison
            fig_project = px.bar(
                project_data,
                x='Tag_Value',
                y='Cost',
                color='Status',
                title="Project Costs by Status"
            )
            st.plotly_chart(fig_project, use_container_width=True)
        
        with col2:
            # Project progress vs cost
            fig_progress = px.scatter(
                project_data,
                x='Progress',
                y='Cost',
                size='Cost',
                color='Tag_Value',
                title="Project Progress vs Cost",
                labels={'Progress': 'Progress (%)', 'Cost': 'Cost ($)'}
            )
            st.plotly_chart(fig_progress, use_container_width=True)
        
        # Project details table
        st.dataframe(
            project_data[['Tag_Value', 'Cost', 'Start Date', 'End Date', 'Status', 'Progress']],
            use_container_width=True,
            column_config={
                'Tag_Value': 'Project',
                'Cost': st.column_config.NumberColumn('Cost ($)', format='$%.2f'),
                'Progress': st.column_config.ProgressColumn('Progress', min_value=0, max_value=100)
            }
        )
    
    with tab4:
        st.markdown("**Cost Center Allocation**")
        
        # Mock cost center data
        cost_center_data = pd.DataFrame({
            'Cost Center': ['Engineering', 'Marketing', 'Sales', 'Operations', 'R&D'],
            'Budget': [15000, 8000, 5000, 12000, 10000],
            'Actual': [14200, 7800, 4900, 11500, 9800],
            'Variance': [-800, -200, -100, -500, -200],
            'Department Head': ['John Doe', 'Jane Smith', 'Mike Johnson', 'Sarah Wilson', 'Tom Brown']
        })
        
        cost_center_data['Variance %'] = (cost_center_data['Variance'] / cost_center_data['Budget']) * 100
        cost_center_data['Status'] = cost_center_data['Variance %'].apply(
            lambda x: '🟢 Under Budget' if x < -5 else '🟡 On Track' if x < 5 else '🔴 Over Budget'
        )
        
        # Cost center performance
        fig_cc = px.bar(
            cost_center_data,
            x='Cost Center',
            y=['Budget', 'Actual'],
            title="Budget vs Actual by Cost Center",
            barmode='group'
        )
        
        st.plotly_chart(fig_cc, use_container_width=True)
        
        # Cost center table
        st.dataframe(
            cost_center_data,
            use_container_width=True,
            column_config={
                'Budget': st.column_config.NumberColumn('Budget ($)', format='$%.2f'),
                'Actual': st.column_config.NumberColumn('Actual ($)', format='$%.2f'),
                'Variance': st.column_config.NumberColumn('Variance ($)', format='$%.2f'),
                'Variance %': st.column_config.NumberColumn('Variance (%)', format='%.1f%%')
            }
        )
    
    # Tag management section
    st.markdown("---")
    st.subheader("🔧 Tag Management & Policies")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📋 Current Tag Policies**")
        
        policies = [
            {"Tag": "Environment", "Required": True, "Values": ["Production", "Development", "Staging", "Testing"]},
            {"Tag": "Team", "Required": True, "Values": ["Backend", "Frontend", "DevOps", "Data", "QA"]},
            {"Tag": "Project", "Required": True, "Values": ["ProjectA", "ProjectB", "ProjectC", "Infrastructure"]},
            {"Tag": "CostCenter", "Required": False, "Values": ["Engineering", "Marketing", "Sales", "Operations"]},
            {"Tag": "Owner", "Required": True, "Values": ["Dynamic - Email addresses"]}
        ]
        
        for policy in policies:
            required_text = "✅ Required" if policy["Required"] else "⚪ Optional"
            st.markdown(f"""
            <div class="tag-policy">
                <strong>{policy['Tag']}</strong> - {required_text}<br>
                <small>Allowed values: {', '.join(policy['Values'][:3])}{'...' if len(policy['Values']) > 3 else ''}</small>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("**🚨 Tag Violations**")
        
        violations = [
            {"Resource": "i-1234567890abcdef0", "Service": "EC2", "Missing Tags": ["Team", "Project"]},
            {"Resource": "vol-0987654321fedcba0", "Service": "EBS", "Missing Tags": ["Environment"]},
            {"Resource": "bucket-example-logs", "Service": "S3", "Missing Tags": ["Owner", "CostCenter"]},
            {"Resource": "lambda-data-processor", "Service": "Lambda", "Missing Tags": ["Team"]}
        ]
        
        for violation in violations:
            st.markdown(f"""
            <div class="compliance-warning">
                <strong>{violation['Resource']}</strong> ({violation['Service']})<br>
                Missing: {', '.join(violation['Missing Tags'])}
            </div>
            """, unsafe_allow_html=True)
        
        if st.button("🔧 Auto-fix Violations"):
            st.success("Auto-fix initiated for tag violations!")
    
    # Allocation insights
    st.markdown("---")
    st.subheader("💡 Allocation Insights & Recommendations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="allocation-summary">
            <h4>🎯 Optimization Opportunity</h4>
            <p>Development environment is using <strong>18%</strong> of total costs</p>
            <p><strong>Recommendation:</strong> Implement scheduled scaling</p>
            <p><strong>Potential Savings:</strong> $440/month</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="allocation-summary">
            <h4>📊 Chargeback Accuracy</h4>
            <p>Current tagging compliance: <strong>91.2%</strong></p>
            <p><strong>Unallocated costs:</strong> $780/month</p>
            <p><strong>Action:</strong> Improve tagging for accurate chargeback</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="allocation-summary">
            <h4>🔄 Resource Optimization</h4>
            <p>Backend team has highest cost per resource</p>
            <p><strong>Cost/Resource:</strong> $178</p>
            <p><strong>Suggestion:</strong> Review resource utilization</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

