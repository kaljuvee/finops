import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_cost_data(days=30):
    """Generate sample AWS cost data"""
    dates = pd.date_range(start=datetime.now() - timedelta(days=days), end=datetime.now(), freq='D')
    
    # Base cost with trend and seasonality
    base_cost = 250
    trend = np.linspace(0, 50, len(dates))
    seasonal = 30 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)  # Weekly seasonality
    noise = np.random.normal(0, 20, len(dates))
    
    costs = base_cost + trend + seasonal + noise
    costs = np.maximum(costs, 50)  # Ensure positive costs
    
    return pd.DataFrame({
        'date': dates,
        'cost': costs
    })

def generate_historical_cost_data(days=365):
    """Generate historical cost data for forecasting"""
    dates = pd.date_range(start=datetime.now() - timedelta(days=days), end=datetime.now(), freq='D')
    
    # More complex pattern for forecasting
    base_cost = 300
    
    # Long-term trend (yearly growth)
    yearly_trend = 0.15 * np.arange(len(dates)) / 365
    
    # Seasonal patterns
    yearly_seasonal = 50 * np.sin(2 * np.pi * np.arange(len(dates)) / 365)  # Yearly
    weekly_seasonal = 20 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)   # Weekly
    monthly_seasonal = 15 * np.sin(2 * np.pi * np.arange(len(dates)) / 30) # Monthly
    
    # Business events (random spikes)
    events = np.zeros(len(dates))
    event_days = np.random.choice(len(dates), size=int(len(dates) * 0.05), replace=False)
    events[event_days] = np.random.normal(100, 30, len(event_days))
    
    # Noise
    noise = np.random.normal(0, 25, len(dates))
    
    costs = base_cost * (1 + yearly_trend) + yearly_seasonal + weekly_seasonal + monthly_seasonal + events + noise
    costs = np.maximum(costs, 50)  # Ensure positive costs
    
    return pd.DataFrame({
        'date': dates,
        'cost': costs
    })

def generate_anomaly_data(days=30):
    """Generate cost data with anomalies"""
    dates = pd.date_range(start=datetime.now() - timedelta(days=days), end=datetime.now(), freq='D')
    
    # Base cost pattern
    base_cost = 300
    seasonal = 20 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)
    noise = np.random.normal(0, 15, len(dates))
    
    costs = base_cost + seasonal + noise
    
    # Inject anomalies (5-10% of data points)
    num_anomalies = int(len(dates) * np.random.uniform(0.05, 0.1))
    anomaly_indices = np.random.choice(len(dates), size=num_anomalies, replace=False)
    
    for idx in anomaly_indices:
        # Random anomaly type
        anomaly_type = np.random.choice(['spike', 'dip', 'sustained'])
        
        if anomaly_type == 'spike':
            costs[idx] *= np.random.uniform(2.5, 4.0)
        elif anomaly_type == 'dip':
            costs[idx] *= np.random.uniform(0.1, 0.4)
        else:  # sustained anomaly
            duration = min(np.random.randint(3, 8), len(dates) - idx)
            multiplier = np.random.uniform(1.8, 2.5)
            costs[idx:idx+duration] *= multiplier
    
    costs = np.maximum(costs, 10)  # Ensure positive costs
    
    return pd.DataFrame({
        'date': dates,
        'cost': costs,
        'is_anomaly': np.isin(np.arange(len(dates)), anomaly_indices)
    })

def generate_multivariate_cost_data(days=30):
    """Generate multivariate cost data for advanced anomaly detection"""
    dates = pd.date_range(start=datetime.now() - timedelta(days=days), end=datetime.now(), freq='D')
    
    # Generate correlated features
    n_points = len(dates)
    
    # Base patterns
    base_cost = 300
    base_usage = 100
    base_requests = 10000
    
    # Seasonal patterns
    seasonal_cost = 30 * np.sin(2 * np.pi * np.arange(n_points) / 7)
    seasonal_usage = 15 * np.sin(2 * np.pi * np.arange(n_points) / 7)
    seasonal_requests = 2000 * np.sin(2 * np.pi * np.arange(n_points) / 7)
    
    # Correlated noise
    noise_matrix = np.random.multivariate_normal(
        mean=[0, 0, 0],
        cov=[[400, 200, 1000], [200, 100, 500], [1000, 500, 5000000]],
        size=n_points
    )
    
    # Generate features
    cost = base_cost + seasonal_cost + noise_matrix[:, 0]
    usage_hours = base_usage + seasonal_usage + noise_matrix[:, 1]
    request_count = base_requests + seasonal_requests + noise_matrix[:, 2]
    
    # Ensure positive values
    cost = np.maximum(cost, 50)
    usage_hours = np.maximum(usage_hours, 10)
    request_count = np.maximum(request_count, 1000)
    
    # Add some anomalies
    num_anomalies = int(n_points * 0.08)
    anomaly_indices = np.random.choice(n_points, size=num_anomalies, replace=False)
    
    for idx in anomaly_indices:
        # Correlated anomaly - if cost spikes, usage and requests also spike
        multiplier = np.random.uniform(2.0, 3.5)
        cost[idx] *= multiplier
        usage_hours[idx] *= np.random.uniform(1.5, 2.0)
        request_count[idx] *= np.random.uniform(1.8, 2.5)
    
    return pd.DataFrame({
        'date': dates,
        'cost': cost,
        'usage_hours': usage_hours,
        'request_count': request_count,
        'data_transfer_gb': np.random.uniform(50, 500, n_points),
        'api_calls': np.random.randint(5000, 50000, n_points)
    })

def generate_forecast_scenarios(days=90):
    """Generate forecast scenario data"""
    dates = pd.date_range(start=datetime.now(), periods=days, freq='D')
    
    scenarios = {}
    base_cost = 350
    
    # Conservative scenario (2% growth)
    conservative_trend = np.linspace(0, base_cost * 0.02, days)
    conservative_seasonal = 20 * np.sin(2 * np.pi * np.arange(days) / 7)
    conservative_noise = np.random.normal(0, 10, days)
    scenarios['Conservative'] = base_cost + conservative_trend + conservative_seasonal + conservative_noise
    
    # Baseline scenario (5% growth)
    baseline_trend = np.linspace(0, base_cost * 0.05, days)
    baseline_seasonal = 25 * np.sin(2 * np.pi * np.arange(days) / 7)
    baseline_noise = np.random.normal(0, 15, days)
    scenarios['Baseline'] = base_cost + baseline_trend + baseline_seasonal + baseline_noise
    
    # Aggressive scenario (10% growth)
    aggressive_trend = np.linspace(0, base_cost * 0.10, days)
    aggressive_seasonal = 30 * np.sin(2 * np.pi * np.arange(days) / 7)
    aggressive_noise = np.random.normal(0, 20, days)
    scenarios['Aggressive'] = base_cost + aggressive_trend + aggressive_seasonal + aggressive_noise
    
    # Ensure positive values
    for scenario in scenarios:
        scenarios[scenario] = np.maximum(scenarios[scenario], 50)
    
    return pd.DataFrame({
        'date': dates,
        **scenarios
    })

def generate_service_breakdown():
    """Generate service-wise cost breakdown"""
    services = ['Amazon EC2', 'Amazon S3', 'Amazon RDS', 'AWS Lambda', 'Amazon CloudWatch', 'Amazon VPC']
    
    # Generate realistic cost distribution
    total_cost = 15000
    weights = [0.45, 0.22, 0.15, 0.08, 0.06, 0.04]  # EC2 dominates
    
    breakdown = []
    for service, weight in zip(services, weights):
        cost = total_cost * weight * np.random.uniform(0.8, 1.2)  # Add some variation
        breakdown.append({
            'service': service,
            'cost': cost,
            'percentage': (cost / total_cost) * 100
        })
    
    return breakdown

def generate_optimization_recommendations():
    """Generate cost optimization recommendations"""
    recommendations = [
        {
            'id': 'OPT-001',
            'title': 'Right-size EC2 instances in Production',
            'description': 'Analysis shows 15 EC2 instances are over-provisioned with CPU utilization below 20%. Right-sizing to smaller instance types can reduce costs significantly.',
            'service': 'Amazon EC2',
            'type': 'Right-sizing',
            'category': 'Compute Optimization',
            'priority': 'High',
            'potential_savings': 450.75,
            'effort': 'Low',
            'risk': 'Low',
            'implementation': 'Use AWS Compute Optimizer recommendations to resize instances during maintenance window.',
            'status': 'New',
            'timeline': '1-2 weeks'
        },
        {
            'id': 'OPT-002', 
            'title': 'Purchase Reserved Instances for stable workloads',
            'description': 'Identified 8 EC2 instances running 24/7 for over 6 months. Reserved Instance purchase can provide up to 60% savings.',
            'service': 'Amazon EC2',
            'type': 'Reserved Instances',
            'category': 'Commitment Discounts',
            'priority': 'High',
            'potential_savings': 890.25,
            'effort': 'Low',
            'risk': 'Low',
            'implementation': 'Purchase 1-year Standard Reserved Instances for identified instance types.',
            'status': 'In Progress',
            'timeline': '1 week'
        },
        {
            'id': 'OPT-003',
            'title': 'Implement S3 Intelligent Tiering',
            'description': 'Large amount of S3 data (2.5TB) has not been accessed in 90+ days. Intelligent Tiering can automatically move to cheaper storage classes.',
            'service': 'Amazon S3',
            'type': 'Storage Optimization',
            'category': 'Storage Optimization',
            'priority': 'Medium',
            'potential_savings': 125.50,
            'effort': 'Low',
            'risk': 'Low',
            'implementation': 'Enable S3 Intelligent Tiering on identified buckets and prefixes.',
            'status': 'New',
            'timeline': '3-5 days'
        },
        {
            'id': 'OPT-004',
            'title': 'Clean up unused EBS volumes',
            'description': '12 EBS volumes (total 480GB) are unattached and incurring storage costs. These appear to be leftover from terminated instances.',
            'service': 'Amazon EBS',
            'type': 'Resource Cleanup',
            'category': 'Resource Cleanup',
            'priority': 'Medium',
            'potential_savings': 48.60,
            'effort': 'Low',
            'risk': 'Medium',
            'implementation': 'Verify volumes are not needed, create snapshots if required, then delete unused volumes.',
            'status': 'New',
            'timeline': '1 week'
        },
        {
            'id': 'OPT-005',
            'title': 'Optimize Lambda memory allocation',
            'description': 'Lambda functions are over-provisioned with memory. Analysis shows 40% memory reduction possible without performance impact.',
            'service': 'AWS Lambda',
            'type': 'Right-sizing',
            'category': 'Compute Optimization',
            'priority': 'Low',
            'potential_savings': 67.80,
            'effort': 'Medium',
            'risk': 'Medium',
            'implementation': 'Use AWS Lambda Power Tuning tool to optimize memory settings for each function.',
            'status': 'New',
            'timeline': '2-3 weeks'
        },
        {
            'id': 'OPT-006',
            'title': 'Implement auto-scaling for development environments',
            'description': 'Development instances run 24/7 but are only used during business hours. Auto-scaling can reduce costs by 60%.',
            'service': 'Amazon EC2',
            'type': 'Automation',
            'category': 'Automation',
            'priority': 'Medium',
            'potential_savings': 320.40,
            'effort': 'High',
            'risk': 'Low',
            'implementation': 'Set up Auto Scaling Groups with scheduled scaling policies for dev environments.',
            'status': 'New',
            'timeline': '3-4 weeks'
        }
    ]
    
    return recommendations

def generate_budget_data():
    """Generate budget tracking data"""
    budgets = [
        {
            'name': 'Production Environment',
            'budget_amount': 8000.00,
            'spent_amount': 6450.00,
            'remaining_amount': 1550.00,
            'utilization': 80.6,
            'forecast': 7800.00,
            'status': 'On Track'
        },
        {
            'name': 'Development Environment', 
            'budget_amount': 2000.00,
            'spent_amount': 1850.00,
            'remaining_amount': 150.00,
            'utilization': 92.5,
            'forecast': 2100.00,
            'status': 'At Risk'
        },
        {
            'name': 'Data Analytics',
            'budget_amount': 3000.00,
            'spent_amount': 2100.00,
            'remaining_amount': 900.00,
            'utilization': 70.0,
            'forecast': 2800.00,
            'status': 'On Track'
        }
    ]
    
    total_budget = sum(b['budget_amount'] for b in budgets)
    total_spent = sum(b['spent_amount'] for b in budgets)
    total_remaining = sum(b['remaining_amount'] for b in budgets)
    overall_utilization = (total_spent / total_budget) * 100
    
    return {
        'budgets': budgets,
        'total_budget': total_budget,
        'total_spent': total_spent,
        'total_remaining': total_remaining,
        'overall_utilization': overall_utilization
    }

def generate_forecast_data(days=30):
    """Generate forecast vs actual data"""
    dates = pd.date_range(start=datetime.now() - timedelta(days=days//2), 
                         end=datetime.now() + timedelta(days=days//2), freq='D')
    
    data = []
    base_cost = 350
    
    for i, date in enumerate(dates):
        # Historical data (first half)
        if i < len(dates) // 2:
            actual_cost = base_cost + 30 * np.sin(2 * np.pi * i / 7) + np.random.normal(0, 20)
            data.append({
                'Date': date,
                'Cost': max(actual_cost, 50),
                'Type': 'Historical'
            })
        else:
            # Forecast data (second half)
            forecast_cost = base_cost * 1.05 + 25 * np.sin(2 * np.pi * i / 7) + np.random.normal(0, 15)
            data.append({
                'Date': date,
                'Cost': max(forecast_cost, 50),
                'Type': 'Forecast'
            })
    
    return pd.DataFrame(data)

def generate_cost_allocation_data():
    """Generate cost allocation data by departments/projects"""
    departments = ['Engineering', 'Marketing', 'Sales', 'Operations', 'Finance']
    projects = ['Project Alpha', 'Project Beta', 'Project Gamma', 'Infrastructure', 'Analytics']
    
    allocation_data = []
    
    for dept in departments:
        for project in projects:
            if np.random.random() > 0.3:  # Not all combinations exist
                cost = np.random.uniform(500, 5000)
                allocation_data.append({
                    'department': dept,
                    'project': project,
                    'cost': cost,
                    'percentage': 0,  # Will be calculated
                    'budget': cost * np.random.uniform(1.1, 1.5),
                    'variance': 0  # Will be calculated
                })
    
    # Calculate percentages and variances
    total_cost = sum(item['cost'] for item in allocation_data)
    for item in allocation_data:
        item['percentage'] = (item['cost'] / total_cost) * 100
        item['variance'] = ((item['cost'] - item['budget']) / item['budget']) * 100
    
    return allocation_data

def generate_tag_analysis_data():
    """Generate tag-based cost analysis data"""
    tags = {
        'Environment': ['Production', 'Development', 'Staging', 'Test'],
        'Team': ['Backend', 'Frontend', 'DevOps', 'Data', 'Mobile'],
        'Application': ['WebApp', 'API', 'Database', 'Analytics', 'Monitoring'],
        'CostCenter': ['Engineering', 'Marketing', 'Sales', 'Operations']
    }
    
    tag_data = []
    
    for tag_key, tag_values in tags.items():
        for tag_value in tag_values:
            cost = np.random.uniform(1000, 8000)
            resource_count = np.random.randint(5, 50)
            
            tag_data.append({
                'tag_key': tag_key,
                'tag_value': tag_value,
                'cost': cost,
                'resource_count': resource_count,
                'avg_cost_per_resource': cost / resource_count,
                'percentage_of_total': 0  # Will be calculated
            })
    
    # Calculate percentages
    total_cost = sum(item['cost'] for item in tag_data)
    for item in tag_data:
        item['percentage_of_total'] = (item['cost'] / total_cost) * 100
    
    return tag_data

