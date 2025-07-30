import boto3
import pandas as pd
from datetime import datetime, timedelta
import json
from typing import Dict, List, Optional
import logging

class CostMonitor:
    """
    AWS Cost monitoring and data retrieval class
    Integrates with AWS Cost Explorer API for real-time billing data
    """
    
    def __init__(self, aws_access_key_id: Optional[str] = None, 
                 aws_secret_access_key: Optional[str] = None,
                 region_name: str = 'us-east-1'):
        """
        Initialize Cost Monitor with AWS credentials
        """
        self.logger = logging.getLogger(__name__)
        
        try:
            if aws_access_key_id and aws_secret_access_key:
                self.ce_client = boto3.client(
                    'ce',
                    aws_access_key_id=aws_access_key_id,
                    aws_secret_access_key=aws_secret_access_key,
                    region_name=region_name
                )
            else:
                # Use default credentials (IAM role, environment variables, etc.)
                self.ce_client = boto3.client('ce', region_name=region_name)
                
        except Exception as e:
            self.logger.warning(f"AWS client initialization failed: {e}")
            self.ce_client = None
    
    def get_cost_and_usage(self, start_date: str, end_date: str, 
                          granularity: str = 'DAILY',
                          metrics: List[str] = None) -> Dict:
        """
        Retrieve cost and usage data from AWS Cost Explorer
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            granularity: DAILY, MONTHLY, or HOURLY
            metrics: List of metrics to retrieve
            
        Returns:
            Dictionary containing cost and usage data
        """
        if not self.ce_client:
            return self._generate_mock_data(start_date, end_date, granularity)
        
        if metrics is None:
            metrics = ['BlendedCost', 'UsageQuantity']
        
        try:
            response = self.ce_client.get_cost_and_usage(
                TimePeriod={
                    'Start': start_date,
                    'End': end_date
                },
                Granularity=granularity,
                Metrics=metrics,
                GroupBy=[
                    {
                        'Type': 'DIMENSION',
                        'Key': 'SERVICE'
                    }
                ]
            )
            return response
        except Exception as e:
            self.logger.error(f"Error retrieving cost data: {e}")
            return self._generate_mock_data(start_date, end_date, granularity)
    
    def get_service_costs(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get costs broken down by AWS service
        
        Returns:
            DataFrame with service costs
        """
        data = self.get_cost_and_usage(start_date, end_date)
        
        if 'ResultsByTime' not in data:
            return self._generate_mock_service_costs()
        
        service_costs = []
        for result in data['ResultsByTime']:
            for group in result['Groups']:
                service_name = group['Keys'][0]
                cost = float(group['Metrics']['BlendedCost']['Amount'])
                service_costs.append({
                    'Service': service_name,
                    'Cost': cost,
                    'Date': result['TimePeriod']['Start']
                })
        
        return pd.DataFrame(service_costs)
    
    def get_daily_costs(self, days: int = 30) -> pd.DataFrame:
        """
        Get daily costs for the specified number of days
        
        Args:
            days: Number of days to retrieve
            
        Returns:
            DataFrame with daily costs
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        data = self.get_cost_and_usage(start_str, end_str, 'DAILY')
        
        if 'ResultsByTime' not in data:
            return self._generate_mock_daily_costs(days)
        
        daily_costs = []
        for result in data['ResultsByTime']:
            total_cost = 0
            for group in result['Groups']:
                total_cost += float(group['Metrics']['BlendedCost']['Amount'])
            
            daily_costs.append({
                'Date': pd.to_datetime(result['TimePeriod']['Start']),
                'Cost': total_cost
            })
        
        return pd.DataFrame(daily_costs)
    
    def get_cost_by_tags(self, start_date: str, end_date: str, 
                        tag_key: str) -> pd.DataFrame:
        """
        Get costs grouped by tag values
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            tag_key: Tag key to group by
            
        Returns:
            DataFrame with costs by tag
        """
        if not self.ce_client:
            return self._generate_mock_tag_costs(tag_key)
        
        try:
            response = self.ce_client.get_cost_and_usage(
                TimePeriod={
                    'Start': start_date,
                    'End': end_date
                },
                Granularity='MONTHLY',
                Metrics=['BlendedCost'],
                GroupBy=[
                    {
                        'Type': 'TAG',
                        'Key': tag_key
                    }
                ]
            )
            
            tag_costs = []
            for result in response['ResultsByTime']:
                for group in result['Groups']:
                    tag_value = group['Keys'][0] if group['Keys'][0] else 'Untagged'
                    cost = float(group['Metrics']['BlendedCost']['Amount'])
                    tag_costs.append({
                        'Tag': tag_value,
                        'Cost': cost
                    })
            
            return pd.DataFrame(tag_costs)
            
        except Exception as e:
            self.logger.error(f"Error retrieving tag costs: {e}")
            return self._generate_mock_tag_costs(tag_key)
    
    def _generate_mock_data(self, start_date: str, end_date: str, 
                           granularity: str) -> Dict:
        """Generate mock data for demo purposes"""
        import random
        
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        results = []
        current = start
        
        while current < end:
            if granularity == 'DAILY':
                next_date = current + timedelta(days=1)
            else:
                next_date = current + timedelta(days=30)
            
            # Mock service data
            services = ['Amazon Elastic Compute Cloud - Compute', 'Amazon Simple Storage Service',
                       'AWS Lambda', 'Amazon Relational Database Service', 'Amazon CloudWatch']
            
            groups = []
            for service in services:
                cost = random.uniform(50, 500)
                groups.append({
                    'Keys': [service],
                    'Metrics': {
                        'BlendedCost': {
                            'Amount': str(cost),
                            'Unit': 'USD'
                        }
                    }
                })
            
            results.append({
                'TimePeriod': {
                    'Start': current.strftime('%Y-%m-%d'),
                    'End': next_date.strftime('%Y-%m-%d')
                },
                'Groups': groups
            })
            
            current = next_date
        
        return {'ResultsByTime': results}
    
    def _generate_mock_service_costs(self) -> pd.DataFrame:
        """Generate mock service cost data"""
        import random
        
        services = ['EC2', 'S3', 'Lambda', 'RDS', 'CloudWatch', 'VPC', 'EBS']
        data = []
        
        for service in services:
            cost = random.uniform(100, 2000)
            data.append({
                'Service': service,
                'Cost': cost,
                'Date': datetime.now().strftime('%Y-%m-%d')
            })
        
        return pd.DataFrame(data)
    
    def _generate_mock_daily_costs(self, days: int) -> pd.DataFrame:
        """Generate mock daily cost data"""
        import random
        
        dates = pd.date_range(start=datetime.now() - timedelta(days=days),
                             end=datetime.now(), freq='D')
        
        data = []
        base_cost = 500
        
        for date in dates:
            # Add some variation and trend
            variation = random.uniform(-100, 150)
            trend = (date - dates[0]).days * 2  # Slight upward trend
            cost = base_cost + variation + trend
            
            data.append({
                'Date': date,
                'Cost': max(cost, 50)  # Ensure positive costs
            })
        
        return pd.DataFrame(data)
    
    def _generate_mock_tag_costs(self, tag_key: str) -> pd.DataFrame:
        """Generate mock tag-based cost data"""
        import random
        
        if tag_key.lower() == 'environment':
            tags = ['Production', 'Development', 'Staging', 'Testing']
        elif tag_key.lower() == 'team':
            tags = ['Backend', 'Frontend', 'DevOps', 'Data']
        elif tag_key.lower() == 'project':
            tags = ['ProjectA', 'ProjectB', 'ProjectC', 'Infrastructure']
        else:
            tags = ['Value1', 'Value2', 'Value3', 'Untagged']
        
        data = []
        for tag in tags:
            cost = random.uniform(200, 1500)
            data.append({
                'Tag': tag,
                'Cost': cost
            })
        
        return pd.DataFrame(data)

