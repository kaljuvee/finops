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

from optimizer import CostOptimizer
from data_manager import data_manager
from data_viewer import display_data_section, create_data_sidebar

# Page configuration
st.set_page_config(
    page_title="Cost Optimization - FinOps",
    page_icon="💡",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .recommendation-high {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-left: 4px solid #28a745;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .recommendation-medium {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-left: 4px solid #ffc107;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .recommendation-low {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-left: 4px solid #dc3545;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .savings-summary {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 1rem 0;
    }
    .implementation-card {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .progress-bar {
        background-color: #e9ecef;
        border-radius: 0.25rem;
        height: 1rem;
        overflow: hidden;
    }
    .progress-fill {
        background-color: #007bff;
        height: 100%;
        transition: width 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.title("💡 Cost Optimization Recommendations")
    st.markdown("AI-powered cost optimization recommendations and implementation tracking")
    
    # Initialize optimizer
    optimizer = CostOptimizer()
    
    # Sidebar for filters and settings
    with st.sidebar:
        st.header("⚙️ Optimization Settings")
        
        # Recommendation filters
        st.subheader("🔍 Filters")
        
        priority_filter = st.multiselect(
            "Priority Level",
            ["High", "Medium", "Low"],
            default=["High", "Medium", "Low"]
        )
        
        category_filter = st.multiselect(
            "Category",
            ["Compute Optimization", "Storage Optimization", "Commitment Discounts", 
             "Resource Cleanup", "Automation"],
            default=["Compute Optimization", "Storage Optimization", "Commitment Discounts"]
        )
        
        effort_filter = st.multiselect(
            "Implementation Effort",
            ["Low", "Medium", "High"],
            default=["Low", "Medium", "High"]
        )
        
        # Savings threshold
        min_savings = st.slider(
            "Minimum Savings ($)",
            min_value=0,
            max_value=1000,
            value=50,
            step=25
        )
        
        st.markdown("---")
        
        # Quick actions
        st.subheader("⚡ Quick Actions")
        
        if st.button("🔄 Refresh Recommendations"):
            st.success("Recommendations refreshed!")
        
        if st.button("📊 Generate Report"):
            st.success("Optimization report generated!")
        
        if st.button("📧 Email Summary"):
            st.success("Summary emailed to stakeholders!")
        
        st.markdown("---")
        
        # Implementation tracking
        st.subheader("📈 Implementation Progress")
        
        total_recommendations = 15
        implemented = 8
        in_progress = 4
        pending = 3
        
        progress_pct = (implemented / total_recommendations) * 100
        
        st.metric("Overall Progress", f"{progress_pct:.1f}%")
        st.progress(progress_pct / 100)
        
        st.markdown(f"""
        - ✅ Implemented: {implemented}
        - 🔄 In Progress: {in_progress}
        - ⏳ Pending: {pending}
        """)
        
        # Add data management sidebar
        create_data_sidebar(data_manager)
    
    # Get data from CSV files
    recommendations = data_manager.get_optimization_recommendations().to_dict('records')
    service_data = data_manager.get_service_breakdown()
    
    # Calculate summary metrics
    total_savings = sum(rec['potential_savings'] for rec in recommendations)
    high_priority = len([r for r in recommendations if r['priority'] == 'High'])
    quick_wins = len([r for r in recommendations if r['effort'] == 'Low' and r['potential_savings'] > 100])
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="💰 Total Potential Savings",
            value=f"${total_savings:,.2f}",
            delta="Monthly"
        )
    
    with col2:
        st.metric(
            label="🎯 High Priority Items",
            value=str(high_priority),
            delta="Immediate action"
        )
    
    with col3:
        st.metric(
            label="⚡ Quick Wins",
            value=str(quick_wins),
            delta="Low effort, high impact"
        )
    
    with col4:
        annual_savings = total_savings * 12
        st.metric(
            label="📈 Annual Savings",
            value=f"${annual_savings:,.2f}",
            delta="Projected"
        )
    
    st.markdown("---")
    
    # Recommendations overview
    st.subheader("📋 Optimization Recommendations")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Recommendations by category chart
        category_savings = {}
        for rec in recommendations:
            category = rec.get('category', 'Other')
            category_savings[category] = category_savings.get(category, 0) + rec['potential_savings']
        
        category_df = pd.DataFrame([
            {'Category': k, 'Savings': v} for k, v in category_savings.items()
        ])
        
        fig_category = px.bar(
            category_df,
            x='Category',
            y='Savings',
            title="Potential Savings by Category",
            color='Savings',
            color_continuous_scale='Greens'
        )
        
        fig_category.update_layout(height=400)
        st.plotly_chart(fig_category, use_container_width=True)
    
    with col2:
        # Priority distribution pie chart
        priority_counts = {}
        for rec in recommendations:
            priority = rec['priority']
            priority_counts[priority] = priority_counts.get(priority, 0) + 1
        
        priority_df = pd.DataFrame([
            {'Priority': k, 'Count': v} for k, v in priority_counts.items()
        ])
        
        fig_priority = px.pie(
            priority_df,
            values='Count',
            names='Priority',
            title="Recommendations by Priority",
            color_discrete_map={
                'High': '#dc3545',
                'Medium': '#ffc107',
                'Low': '#28a745'
            }
        )
        
        fig_priority.update_layout(height=400)
        st.plotly_chart(fig_priority, use_container_width=True)
    
    st.markdown("---")
    
    # Detailed recommendations tabs
    st.subheader("🔍 Detailed Recommendations")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 High Priority", "⚡ Quick Wins", "📊 All Recommendations", "📈 Implementation Plan"])
    
    with tab1:
        st.markdown("**High Priority Recommendations (Immediate Action Required)**")
        
        high_priority_recs = [r for r in recommendations if r['priority'] == 'High']
        
        for rec in high_priority_recs:
            priority_class = f"recommendation-{rec['priority'].lower()}"
            
            st.markdown(f"""
            <div class="{priority_class}">
                <h4>🎯 {rec['title']}</h4>
                <p><strong>Service:</strong> {rec['service']} | <strong>Type:</strong> {rec['type']}</p>
                <p>{rec['description']}</p>
                <p><strong>💰 Potential Savings:</strong> ${rec['potential_savings']:.2f}/month</p>
                <p><strong>⚡ Effort:</strong> {rec['effort']} | <strong>🛡️ Risk:</strong> {rec['risk']}</p>
                <p><strong>🔧 Implementation:</strong> {rec['implementation']}</p>
                <p><strong>📅 Timeline:</strong> {rec.get('timeline', 'TBD')} | <strong>Status:</strong> {rec['status']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Action buttons
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button(f"✅ Approve {rec['id']}", key=f"approve_{rec['id']}"):
                    st.success(f"Recommendation {rec['id']} approved!")
            with col2:
                if st.button(f"🔄 Start Implementation {rec['id']}", key=f"implement_{rec['id']}"):
                    st.info(f"Implementation started for {rec['id']}")
            with col3:
                if st.button(f"📋 View Details {rec['id']}", key=f"details_{rec['id']}"):
                    st.info(f"Showing details for {rec['id']}")
    
    with tab2:
        st.markdown("**Quick Wins (Low Effort, High Impact)**")
        
        quick_win_recs = [r for r in recommendations if r['effort'] == 'Low' and r['potential_savings'] > 100]
        
        if not quick_win_recs:
            st.info("No quick wins available at the moment.")
        else:
            # Quick wins summary
            total_quick_savings = sum(r['potential_savings'] for r in quick_win_recs)
            avg_timeline = "1-2 weeks"  # Mock average
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("💰 Total Quick Win Savings", f"${total_quick_savings:.2f}")
            with col2:
                st.metric("⚡ Number of Quick Wins", str(len(quick_win_recs)))
            with col3:
                st.metric("📅 Average Timeline", avg_timeline)
            
            # Quick wins table
            quick_wins_df = pd.DataFrame([
                {
                    'ID': rec['id'],
                    'Title': rec['title'],
                    'Service': rec['service'],
                    'Savings': rec['potential_savings'],
                    'Risk': rec['risk'],
                    'Timeline': rec.get('timeline', 'TBD'),
                    'Status': rec['status']
                }
                for rec in quick_win_recs
            ])
            
            st.dataframe(
                quick_wins_df,
                use_container_width=True,
                column_config={
                    'Savings': st.column_config.NumberColumn('Savings ($)', format='$%.2f')
                }
            )
            
            # Bulk implementation
            if st.button("🚀 Implement All Quick Wins"):
                st.success("All quick wins scheduled for implementation!")
    
    with tab3:
        st.markdown("**All Optimization Recommendations**")
        
        # Filter recommendations
        filtered_recs = [
            r for r in recommendations
            if (r['priority'] in priority_filter and
                r.get('category', 'Other') in category_filter and
                r['effort'] in effort_filter and
                r['potential_savings'] >= min_savings)
        ]
        
        if not filtered_recs:
            st.warning("No recommendations match the current filters.")
        else:
            # Create comprehensive table
            recs_df = pd.DataFrame([
                {
                    'ID': rec['id'],
                    'Title': rec['title'],
                    'Service': rec['service'],
                    'Category': rec.get('category', 'Other'),
                    'Priority': rec['priority'],
                    'Savings': rec['potential_savings'],
                    'Effort': rec['effort'],
                    'Risk': rec['risk'],
                    'Status': rec['status'],
                    'Timeline': rec.get('timeline', 'TBD')
                }
                for rec in filtered_recs
            ])
            
            # Sort by savings (descending)
            recs_df = recs_df.sort_values('Savings', ascending=False)
            
            st.dataframe(
                recs_df,
                use_container_width=True,
                column_config={
                    'Savings': st.column_config.NumberColumn('Savings ($)', format='$%.2f'),
                    'Priority': st.column_config.SelectboxColumn(
                        'Priority',
                        options=['High', 'Medium', 'Low']
                    )
                }
            )
            
            # Detailed view for selected recommendation
            st.markdown("**📋 Recommendation Details**")
            
            selected_id = st.selectbox(
                "Select recommendation for details:",
                options=[rec['id'] for rec in filtered_recs],
                format_func=lambda x: f"{x} - {next(r['title'] for r in filtered_recs if r['id'] == x)}"
            )
            
            if selected_id:
                selected_rec = next(r for r in filtered_recs if r['id'] == selected_id)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    **📊 Recommendation Details**
                    - **ID:** {selected_rec['id']}
                    - **Title:** {selected_rec['title']}
                    - **Service:** {selected_rec['service']}
                    - **Type:** {selected_rec['type']}
                    - **Category:** {selected_rec.get('category', 'Other')}
                    - **Priority:** {selected_rec['priority']}
                    - **Status:** {selected_rec['status']}
                    """)
                
                with col2:
                    st.markdown(f"""
                    **💰 Financial Impact**
                    - **Monthly Savings:** ${selected_rec['potential_savings']:.2f}
                    - **Annual Savings:** ${selected_rec['potential_savings'] * 12:.2f}
                    - **Implementation Effort:** {selected_rec['effort']}
                    - **Risk Level:** {selected_rec['risk']}
                    - **Timeline:** {selected_rec.get('timeline', 'TBD')}
                    """)
                
                st.markdown(f"""
                **📝 Description**
                {selected_rec['description']}
                
                **🔧 Implementation Steps**
                {selected_rec['implementation']}
                """)
                
                # Prerequisites
                if 'prerequisites' in selected_rec:
                    st.markdown("**📋 Prerequisites**")
                    for prereq in selected_rec['prerequisites']:
                        st.markdown(f"- {prereq}")
    
    with tab4:
        st.markdown("**Implementation Plan & Roadmap**")
        
        # Implementation phases
        phases = {
            'Phase 1 (0-30 days)': [r for r in recommendations if r['effort'] == 'Low'],
            'Phase 2 (1-3 months)': [r for r in recommendations if r['effort'] == 'Medium'],
            'Phase 3 (3-6 months)': [r for r in recommendations if r['effort'] == 'High']
        }
        
        for phase_name, phase_recs in phases.items():
            if phase_recs:
                st.markdown(f"**{phase_name}**")
                
                phase_savings = sum(r['potential_savings'] for r in phase_recs)
                phase_count = len(phase_recs)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("💰 Phase Savings", f"${phase_savings:.2f}")
                with col2:
                    st.metric("📊 Recommendations", str(phase_count))
                with col3:
                    st.metric("⏱️ Duration", phase_name.split('(')[1].split(')')[0])
                
                # Phase recommendations table
                phase_df = pd.DataFrame([
                    {
                        'ID': rec['id'],
                        'Title': rec['title'][:50] + '...' if len(rec['title']) > 50 else rec['title'],
                        'Service': rec['service'],
                        'Savings': rec['potential_savings'],
                        'Risk': rec['risk'],
                        'Status': rec['status']
                    }
                    for rec in phase_recs
                ])
                
                st.dataframe(
                    phase_df,
                    use_container_width=True,
                    column_config={
                        'Savings': st.column_config.NumberColumn('Savings ($)', format='$%.2f')
                    }
                )
                
                st.markdown("---")
        
        # Implementation timeline chart
        st.markdown("**📅 Implementation Timeline**")
        
        # Create Gantt-like chart
        timeline_data = []
        start_date = datetime.now()
        
        for i, (phase_name, phase_recs) in enumerate(phases.items()):
            if phase_recs:
                phase_start = start_date + timedelta(days=i*30)
                phase_end = phase_start + timedelta(days=30)
                
                timeline_data.append({
                    'Phase': phase_name.split(' (')[0],
                    'Start': phase_start,
                    'End': phase_end,
                    'Savings': sum(r['potential_savings'] for r in phase_recs),
                    'Count': len(phase_recs)
                })
        
        if timeline_data:
            timeline_df = pd.DataFrame(timeline_data)
            
            fig_timeline = px.timeline(
                timeline_df,
                x_start='Start',
                x_end='End',
                y='Phase',
                color='Savings',
                title="Implementation Timeline",
                color_continuous_scale='Greens'
            )
            
            fig_timeline.update_layout(height=300)
            st.plotly_chart(fig_timeline, use_container_width=True)
    
    # ROI Analysis
    st.markdown("---")
    st.subheader("📈 ROI Analysis & Impact Projection")
    
    col1, col2, col3 = st.columns(3)
    
    # Calculate ROI metrics
    implementation_cost = 5000  # Mock implementation cost
    monthly_savings = total_savings
    annual_savings = monthly_savings * 12
    roi_months = implementation_cost / monthly_savings if monthly_savings > 0 else 0
    
    with col1:
        st.markdown("""
        <div class="savings-summary">
            <h4>💰 Financial Impact</h4>
            <p><strong>Monthly Savings:</strong> ${:,.2f}</p>
            <p><strong>Annual Savings:</strong> ${:,.2f}</p>
            <p><strong>3-Year Savings:</strong> ${:,.2f}</p>
        </div>
        """.format(monthly_savings, annual_savings, annual_savings * 3), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="savings-summary">
            <h4>⏱️ ROI Timeline</h4>
            <p><strong>Implementation Cost:</strong> ${:,.2f}</p>
            <p><strong>Payback Period:</strong> {:.1f} months</p>
            <p><strong>3-Year ROI:</strong> {:,.0f}%</p>
        </div>
        """.format(implementation_cost, roi_months, ((annual_savings * 3 - implementation_cost) / implementation_cost) * 100), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="savings-summary">
            <h4>📊 Implementation Stats</h4>
            <p><strong>Total Recommendations:</strong> {}</p>
            <p><strong>High Priority:</strong> {}</p>
            <p><strong>Quick Wins:</strong> {}</p>
        </div>
        """.format(len(recommendations), high_priority, quick_wins), unsafe_allow_html=True)
    
    # Savings projection chart
    st.markdown("**📈 Cumulative Savings Projection**")
    
    months = list(range(1, 37))  # 3 years
    cumulative_savings = [monthly_savings * month - implementation_cost for month in months]
    
    projection_df = pd.DataFrame({
        'Month': months,
        'Cumulative Savings': cumulative_savings,
        'Break Even': [0] * len(months)
    })
    
    fig_projection = px.line(
        projection_df,
        x='Month',
        y=['Cumulative Savings', 'Break Even'],
        title="3-Year Cumulative Savings Projection",
        labels={'value': 'Cumulative Savings ($)', 'Month': 'Month'}
    )
    
    fig_projection.update_layout(height=400)
    st.plotly_chart(fig_projection, use_container_width=True)

if __name__ == "__main__":
    main()

