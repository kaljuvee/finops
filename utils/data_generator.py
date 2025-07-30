import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import Dict, List

def generate_sample_data() -> Dict:
    """
    Generate sample data for the FinOps dashboard
    
    Returns:
        Dictionary containing various metrics and sample data
    """
    # Basic metrics
    total_spend = random.uniform(8000, 15000)
    spend_change = random.uniform(-15, 25)
    daily_avg = total_spend / 30
    daily_change = random.uniform(-10, 15)
    budget_utilization = random.uniform(65, 95)
    budget_change = random.uniform(-5, 10)
    potential_savings = random.uniform(500, 2000)
    
    return {
        'total_spend': total_spend,
        'spend_change': spend_change,
        'daily_avg': daily_avg,
        'daily_change': daily_change,
        'budget_utilization': budget_utilization,
        'budget_change': budget_change,
        'potential_savings': potential_savings
    }

def generate_cost_trend_data(days: int = 30) -> pd.DataFrame:
    """
    Generate cost trend data for visualization
    
    Args:
        days: Number of days to generate data for
        
    Returns:
        DataFrame with date and cost columns
    """
    dates = pd.date_range(start=datetime.now() - timedelta(days=days),
                         end=datetime.now(), freq='D')
    
    # Generate realistic cost data with trends and variations
    base_cost = 400
    costs = []
    
    for i, date in enumerate(dates):
        # Add weekly pattern (higher on weekdays)
        weekly_factor = 1.2 if date.weekday() < 5 else 0.8
        
        # Add monthly trend
        monthly_trend = i * 3
        
        # Add random variation
        random_variation = random.uniform(-50, 100)
        
        # Calculate final cost
        cost = base_cost * weekly_factor + monthly_trend + random_variation
        costs.append(max(cost, 50))  # Ensure positive costs
    
    return pd.DataFrame({
        'Date': dates,
        'Cost': costs
    })

def generate_service_breakdown() -> pd.DataFrame:
    """
    Generate AWS service cost breakdown data
    
    Returns:
        DataFrame with service names and costs
    """
    services = [
        'Amazon EC2', 'Amazon S3', 'AWS Lambda', 'Amazon RDS',
        'Amazon CloudWatch', 'Amazon VPC', 'Amazon EBS',
        'Amazon CloudFront', 'AWS IAM', 'Amazon Route 53'
    ]
    
    # Generate realistic cost distribution
    weights = [0.35, 0.20, 0.10, 0.15, 0.05, 0.03, 0.07, 0.02, 0.01, 0.02]
    total_budget = 10000
    
    data = []
    for service, weight in zip(services, weights):
        cost = total_budget * weight * random.uniform(0.7, 1.3)
        data.append({
            'Service': service,
            'Cost': cost,
            'Percentage': (cost / total_budget) * 100
        })
    
    return pd.DataFrame(data).sort_values('Cost', ascending=False)

def generate_anomaly_data() -> List[Dict]:
    """
    Generate sample anomaly detection data
    
    Returns:
        List of anomaly dictionaries
    """
    anomalies = [
        {
            'timestamp': datetime.now() - timedelta(hours=2),
            'service': 'Amazon EC2',
            'anomaly_type': 'Cost Spike',
            'severity': 'High',
            'description': 'EC2 costs increased by 45% compared to baseline',
            'baseline_cost': 850.00,
            'actual_cost': 1232.50,
            'deviation': 45.0,
            'status': 'Active'
        },
        {
            'timestamp': datetime.now() - timedelta(hours=6),
            'service': 'Amazon S3',
            'anomaly_type': 'Usage Anomaly',
            'severity': 'Medium',
            'description': 'Unusual data transfer patterns detected',
            'baseline_cost': 120.00,
            'actual_cost': 185.00,
            'deviation': 54.2,
            'status': 'Investigating'
        },
        {
            'timestamp': datetime.now() - timedelta(days=1),
            'service': 'AWS Lambda',
            'anomaly_type': 'Execution Spike',
            'severity': 'Low',
            'description': 'Lambda invocations 30% above normal',
            'baseline_cost': 45.00,
            'actual_cost': 58.50,
            'deviation': 30.0,
            'status': 'Resolved'
        }
    ]
    
    return anomalies

def generate_optimization_recommendations() -> List[Dict]:
    """
    Generate sample optimization recommendations
    
    Returns:
        List of recommendation dictionaries
    """
    recommendations = [
        {
            'id': 'OPT-001',
            'type': 'Right-sizing',
            'service': 'Amazon EC2',
            'title': 'Right-size over-provisioned EC2 instances',
            'description': 'Identified 12 EC2 instances with consistently low CPU utilization (<20%)',
            'potential_savings': 450.00,
            'effort': 'Low',
            'risk': 'Low',
            'implementation': 'Resize instances from m5.large to m5.medium',
            'status': 'Pending',
            'priority': 'High'
        },
        {
            'id': 'OPT-002',
            'type': 'Storage Optimization',
            'service': 'Amazon S3',
            'title': 'Implement S3 Intelligent Tiering',
            'description': 'Move infrequently accessed data to cheaper storage classes',
            'potential_savings': 280.00,
            'effort': 'Medium',
            'risk': 'Low',
            'implementation': 'Enable S3 Intelligent Tiering on identified buckets',
            'status': 'In Progress',
            'priority': 'Medium'
        },
        {
            'id': 'OPT-003',
            'type': 'Reserved Instances',
            'service': 'Amazon RDS',
            'title': 'Purchase RDS Reserved Instances',
            'description': 'Convert on-demand RDS instances to 1-year reserved instances',
            'potential_savings': 720.00,
            'effort': 'Low',
            'risk': 'Low',
            'implementation': 'Purchase reserved instances for production databases',
            'status': 'Recommended',
            'priority': 'High'
        },
        {
            'id': 'OPT-004',
            'type': 'Resource Cleanup',
            'service': 'Amazon EBS',
            'title': 'Delete unused EBS volumes',
            'description': 'Found 8 unattached EBS volumes consuming storage costs',
            'potential_savings': 95.00,
            'effort': 'Low',
            'risk': 'Medium',
            'implementation': 'Verify and delete unattached volumes after backup',
            'status': 'Pending',
            'priority': 'Medium'
        }
    ]
    
    return recommendations

def generate_budget_data() -> Dict:
    """
    Generate sample budget tracking data
    
    Returns:
        Dictionary containing budget information
    """
    budgets = [
        {
            'name': 'Production Environment',
            'budget_amount': 8000.00,
            'spent_amount': 6450.00,
            'remaining_amount': 1550.00,
            'utilization': 80.6,
            'forecast': 7200.00,
            'status': 'On Track',
            'period': 'Monthly'
        },
        {
            'name': 'Development Environment',
            'budget_amount': 2000.00,
            'spent_amount': 1850.00,
            'remaining_amount': 150.00,
            'utilization': 92.5,
            'forecast': 2100.00,
            'status': 'At Risk',
            'period': 'Monthly'
        },
        {
            'name': 'Data Analytics',
            'budget_amount': 3000.00,
            'spent_amount': 2100.00,
            'remaining_amount': 900.00,
            'utilization': 70.0,
            'forecast': 2800.00,
            'status': 'On Track',
            'period': 'Monthly'
        }
    ]
    
    return {
        'budgets': budgets,
        'total_budget': sum(b['budget_amount'] for b in budgets),
        'total_spent': sum(b['spent_amount'] for b in budgets),
        'total_remaining': sum(b['remaining_amount'] for b in budgets),
        'overall_utilization': (sum(b['spent_amount'] for b in budgets) / 
                              sum(b['budget_amount'] for b in budgets)) * 100
    }

def generate_tag_allocation_data() -> pd.DataFrame:
    """
    Generate sample cost allocation data by tags
    
    Returns:
        DataFrame with tag-based cost allocation
    """
    data = []
    
    # Environment tags
    environments = ['Production', 'Development', 'Staging', 'Testing']
    env_costs = [5500, 2200, 1100, 800]
    
    for env, cost in zip(environments, env_costs):
        data.append({
            'Tag_Type': 'Environment',
            'Tag_Value': env,
            'Cost': cost,
            'Percentage': (cost / sum(env_costs)) * 100
        })
    
    # Team tags
    teams = ['Backend', 'Frontend', 'DevOps', 'Data Science', 'QA']
    team_costs = [3200, 2100, 2800, 1500, 900]
    
    for team, cost in zip(teams, team_costs):
        data.append({
            'Tag_Type': 'Team',
            'Tag_Value': team,
            'Cost': cost,
            'Percentage': (cost / sum(team_costs)) * 100
        })
    
    # Project tags
    projects = ['Project Alpha', 'Project Beta', 'Project Gamma', 'Infrastructure']
    project_costs = [2800, 2200, 1800, 3200]
    
    for project, cost in zip(projects, project_costs):
        data.append({
            'Tag_Type': 'Project',
            'Tag_Value': project,
            'Cost': cost,
            'Percentage': (cost / sum(project_costs)) * 100
        })
    
    return pd.DataFrame(data)

def generate_forecast_data(days: int = 30) -> pd.DataFrame:
    """
    Generate cost forecast data
    
    Args:
        days: Number of days to forecast
        
    Returns:
        DataFrame with historical and forecasted costs
    """
    # Historical data (last 30 days)
    historical_dates = pd.date_range(start=datetime.now() - timedelta(days=30),
                                   end=datetime.now(), freq='D')
    
    # Forecast data (next 30 days)
    forecast_dates = pd.date_range(start=datetime.now() + timedelta(days=1),
                                 end=datetime.now() + timedelta(days=days), freq='D')
    
    # Generate historical costs
    historical_costs = []
    base_cost = 400
    for i, date in enumerate(historical_dates):
        cost = base_cost + i * 2 + random.uniform(-50, 50)
        historical_costs.append(cost)
    
    # Generate forecast costs with trend
    forecast_costs = []
    last_cost = historical_costs[-1]
    for i, date in enumerate(forecast_dates):
        # Add trend and uncertainty
        trend = 3  # Slight upward trend
        uncertainty = random.uniform(-30, 30)
        cost = last_cost + (i * trend) + uncertainty
        forecast_costs.append(cost)
    
    # Combine data
    all_dates = list(historical_dates) + list(forecast_dates)
    all_costs = historical_costs + forecast_costs
    data_types = ['Historical'] * len(historical_dates) + ['Forecast'] * len(forecast_dates)
    
    return pd.DataFrame({
        'Date': all_dates,
        'Cost': all_costs,
        'Type': data_types
    })

