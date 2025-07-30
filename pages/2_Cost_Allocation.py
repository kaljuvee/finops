import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

from data_generator import generate_cost_allocation_data, generate_tag_analysis_data

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
    allocation_data = generate_cost_allocation_data()
    tag_data = generate_tag_analysis_data()
    
    # Convert to DataFrame for easier processing
    allocation_df = pd.DataFrame(allocation_data)
    tag_df = pd.DataFrame(tag_data)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_allocated = allocation_df['cost'].sum()
    dept_count = allocation_df['department'].nunique()
    project_count = allocation_df['project'].nunique()
    avg_project_cost = allocation_df.groupby('project')['cost'].sum().mean()
    
    with col1:
        st.metric(
            label="💰 Total Allocated",
            value=f"${total_allocated:,.2f}",
            delta="100% tagged"
        )
    
    with col2:
        st.metric(
            label="🏢 Departments",
            value=f"{dept_count}",
            delta="Active departments"
        )
    
    with col3:
        st.metric(
            label="📊 Projects",
            value=f"{project_count}",
            delta="Active projects"
        )
    
    with col4:
        st.metric(
            label="📈 Avg Project Cost",
            value=f"${avg_project_cost:,.2f}",
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
    
    tab1, tab2, tab3, tab4 = st.tabs(["🏢 Department", "📊 Project", "🏷️ Tag Analysis", "💼 Cost Center"])
    
    with tab1:
        st.markdown("**Department-based Cost Allocation**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Department pie chart
            dept_costs = allocation_df.groupby('department')['cost'].sum().reset_index()
            fig_dept = px.pie(
                dept_costs,
                values='cost',
                names='department',
                title="Cost Distribution by Department"
            )
            st.plotly_chart(fig_dept, use_container_width=True)
        
        with col2:
            # Department cost breakdown
            dept_summary = allocation_df.groupby('department').agg({
                'cost': 'sum',
                'budget': 'sum',
                'project': 'count'
            }).reset_index()
            dept_summary['utilization'] = (dept_summary['cost'] / dept_summary['budget']) * 100
            dept_summary['status'] = dept_summary['utilization'].apply(
                lambda x: '🟢 On Track' if x < 80 else '🟡 At Risk' if x < 100 else '🔴 Over Budget'
            )
            
            st.dataframe(
                dept_summary,
                use_container_width=True,
                column_config={
                    'department': 'Department',
                    'cost': st.column_config.NumberColumn('Current Cost ($)', format='$%.2f'),
                    'budget': st.column_config.NumberColumn('Budget ($)', format='$%.2f'),
                    'project': 'Projects',
                    'utilization': st.column_config.NumberColumn('Utilization (%)', format='%.1f%%'),
                    'status': 'Status'
                }
            )
    
    with tab2:
        st.markdown("**Project-based Cost Allocation**")
        
        project_summary = allocation_df.groupby('project').agg({
            'cost': 'sum',
            'budget': 'sum',
            'department': 'first'
        }).reset_index()
        project_summary['variance'] = ((project_summary['cost'] - project_summary['budget']) / project_summary['budget']) * 100
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Project cost comparison
            fig_project = px.bar(
                project_summary,
                x='project',
                y='cost',
                color='department',
                title="Project Costs by Department"
            )
            st.plotly_chart(fig_project, use_container_width=True)
        
        with col2:
            # Project variance analysis
            fig_variance = px.scatter(
                project_summary,
                x='budget',
                y='cost',
                size='cost',
                color='department',
                title="Budget vs Actual Cost by Project",
                labels={'budget': 'Budget ($)', 'cost': 'Actual Cost ($)'}
            )
            # Add diagonal line for perfect budget adherence
            fig_variance.add_shape(
                type="line",
                x0=project_summary['budget'].min(),
                y0=project_summary['budget'].min(),
                x1=project_summary['budget'].max(),
                y1=project_summary['budget'].max(),
                line=dict(dash="dash", color="red")
            )
            st.plotly_chart(fig_variance, use_container_width=True)
        
        # Project details table
        st.dataframe(
            project_summary,
            use_container_width=True,
            column_config={
                'project': 'Project',
                'cost': st.column_config.NumberColumn('Cost ($)', format='$%.2f'),
                'budget': st.column_config.NumberColumn('Budget ($)', format='$%.2f'),
                'variance': st.column_config.NumberColumn('Variance (%)', format='%.1f%%'),
                'department': 'Department'
            }
        )
    
    with tab3:
        st.markdown("**Tag-based Analysis**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Tag distribution
            tag_summary = tag_df.groupby('tag_key').agg({
                'cost': 'sum',
                'resource_count': 'sum'
            }).reset_index()
            
            fig_tags = px.bar(
                tag_summary,
                x='tag_key',
                y='cost',
                title="Cost Distribution by Tag Category"
            )
            st.plotly_chart(fig_tags, use_container_width=True)
        
        with col2:
            # Resource efficiency by tag
            fig_efficiency = px.scatter(
                tag_df,
                x='resource_count',
                y='avg_cost_per_resource',
                color='tag_key',
                size='cost',
                title="Resource Efficiency by Tag",
                labels={'resource_count': 'Resource Count', 'avg_cost_per_resource': 'Avg Cost per Resource ($)'}
            )
            st.plotly_chart(fig_efficiency, use_container_width=True)
        
        # Tag details table
        st.dataframe(
            tag_df,
            use_container_width=True,
            column_config={
                'tag_key': 'Tag Category',
                'tag_value': 'Tag Value',
                'cost': st.column_config.NumberColumn('Cost ($)', format='$%.2f'),
                'resource_count': 'Resources',
                'avg_cost_per_resource': st.column_config.NumberColumn('Avg Cost/Resource ($)', format='$%.2f'),
                'percentage_of_total': st.column_config.NumberColumn('% of Total', format='%.1f%%')
            }
        )
    
    with tab4:
        st.markdown("**Cost Center Allocation**")
        
        # Mock cost center data based on departments
        cost_center_summary = allocation_df.groupby('department').agg({
            'cost': 'sum',
            'budget': 'sum'
        }).reset_index()
        cost_center_summary['variance'] = cost_center_summary['cost'] - cost_center_summary['budget']
        cost_center_summary['variance_pct'] = (cost_center_summary['variance'] / cost_center_summary['budget']) * 100
        cost_center_summary['status'] = cost_center_summary['variance_pct'].apply(
            lambda x: '🟢 Under Budget' if x < -5 else '🟡 On Track' if x < 5 else '🔴 Over Budget'
        )
        
        # Cost center performance
        fig_cc = px.bar(
            cost_center_summary,
            x='department',
            y=['budget', 'cost'],
            title="Budget vs Actual by Cost Center",
            barmode='group'
        )
        
        st.plotly_chart(fig_cc, use_container_width=True)
        
        # Cost center table
        st.dataframe(
            cost_center_summary,
            use_container_width=True,
            column_config={
                'department': 'Cost Center',
                'budget': st.column_config.NumberColumn('Budget ($)', format='$%.2f'),
                'cost': st.column_config.NumberColumn('Actual ($)', format='$%.2f'),
                'variance': st.column_config.NumberColumn('Variance ($)', format='$%.2f'),
                'variance_pct': st.column_config.NumberColumn('Variance (%)', format='%.1f%%'),
                'status': 'Status'
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

