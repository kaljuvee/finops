import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import logging

class CostOptimizer:
    """
    AWS Cost optimization recommendation engine
    Analyzes usage patterns and provides actionable cost-saving recommendations
    """
    
    def __init__(self):
        """Initialize the cost optimizer"""
        self.logger = logging.getLogger(__name__)
        
        # Optimization thresholds and parameters
        self.ec2_cpu_threshold = 20.0  # CPU utilization threshold for right-sizing
        self.storage_utilization_threshold = 80.0  # Storage utilization threshold
        self.idle_threshold_days = 7  # Days to consider a resource idle
        self.reserved_instance_threshold = 0.7  # Utilization threshold for RI recommendations
        
        # Cost savings estimates (percentages)
        self.savings_estimates = {
            'right_sizing': 0.3,  # 30% savings from right-sizing
            'reserved_instances': 0.4,  # 40% savings from RIs
            'storage_optimization': 0.25,  # 25% savings from storage optimization
            'idle_resource_cleanup': 1.0,  # 100% savings from removing idle resources
            'spot_instances': 0.6,  # 60% savings from spot instances
            'scheduled_scaling': 0.2  # 20% savings from scheduled scaling
        }
    
    def generate_all_recommendations(self, cost_data: pd.DataFrame,
                                   usage_data: pd.DataFrame = None) -> List[Dict]:
        """
        Generate all types of optimization recommendations
        
        Args:
            cost_data: DataFrame containing cost data by service
            usage_data: DataFrame containing usage metrics (optional)
            
        Returns:
            List of optimization recommendations
        """
        recommendations = []
        
        # EC2 optimization recommendations
        ec2_recommendations = self._generate_ec2_recommendations(cost_data, usage_data)
        recommendations.extend(ec2_recommendations)
        
        # Storage optimization recommendations
        storage_recommendations = self._generate_storage_recommendations(cost_data)
        recommendations.extend(storage_recommendations)
        
        # Reserved Instance recommendations
        ri_recommendations = self._generate_reserved_instance_recommendations(cost_data)
        recommendations.extend(ri_recommendations)
        
        # Idle resource cleanup recommendations
        cleanup_recommendations = self._generate_cleanup_recommendations(cost_data)
        recommendations.extend(cleanup_recommendations)
        
        # Spot instance recommendations
        spot_recommendations = self._generate_spot_instance_recommendations(cost_data)
        recommendations.extend(spot_recommendations)
        
        # Scheduled scaling recommendations
        scaling_recommendations = self._generate_scaling_recommendations(cost_data)
        recommendations.extend(scaling_recommendations)
        
        # Sort by potential savings (descending)
        recommendations.sort(key=lambda x: x['potential_savings'], reverse=True)
        
        # Add recommendation IDs
        for i, rec in enumerate(recommendations, 1):
            rec['id'] = f"OPT-{i:03d}"
        
        return recommendations
    
    def _generate_ec2_recommendations(self, cost_data: pd.DataFrame,
                                    usage_data: pd.DataFrame = None) -> List[Dict]:
        """Generate EC2 right-sizing recommendations"""
        recommendations = []
        
        # Filter EC2 costs
        ec2_data = cost_data[cost_data['Service'].str.contains('EC2|Elastic Compute', case=False, na=False)]
        
        if ec2_data.empty:
            return recommendations
        
        total_ec2_cost = ec2_data['Cost'].sum()
        
        # Simulate right-sizing opportunities
        # In a real implementation, this would analyze actual CPU/memory utilization
        over_provisioned_percentage = 0.4  # Assume 40% of instances are over-provisioned
        potential_savings = total_ec2_cost * over_provisioned_percentage * self.savings_estimates['right_sizing']
        
        if potential_savings > 50:  # Only recommend if savings > $50
            recommendations.append({
                'type': 'Right-sizing',
                'service': 'Amazon EC2',
                'title': 'Right-size over-provisioned EC2 instances',
                'description': f'Analysis shows {over_provisioned_percentage*100:.0f}% of EC2 instances have low utilization (<{self.ec2_cpu_threshold}% CPU)',
                'potential_savings': potential_savings,
                'effort': 'Medium',
                'risk': 'Low',
                'implementation': 'Resize instances to smaller instance types based on utilization patterns',
                'status': 'Recommended',
                'priority': 'High' if potential_savings > 500 else 'Medium',
                'category': 'Compute Optimization',
                'impact': 'Cost Reduction',
                'timeline': '1-2 weeks',
                'prerequisites': ['Performance monitoring', 'Testing in non-prod environment']
            })
        
        return recommendations
    
    def _generate_storage_recommendations(self, cost_data: pd.DataFrame) -> List[Dict]:
        """Generate storage optimization recommendations"""
        recommendations = []
        
        # S3 optimization
        s3_data = cost_data[cost_data['Service'].str.contains('S3|Simple Storage', case=False, na=False)]
        if not s3_data.empty:
            s3_cost = s3_data['Cost'].sum()
            s3_savings = s3_cost * self.savings_estimates['storage_optimization']
            
            if s3_savings > 30:
                recommendations.append({
                    'type': 'Storage Optimization',
                    'service': 'Amazon S3',
                    'title': 'Implement S3 Intelligent Tiering',
                    'description': 'Automatically move infrequently accessed data to cheaper storage classes',
                    'potential_savings': s3_savings,
                    'effort': 'Low',
                    'risk': 'Low',
                    'implementation': 'Enable S3 Intelligent Tiering on buckets with mixed access patterns',
                    'status': 'Recommended',
                    'priority': 'Medium',
                    'category': 'Storage Optimization',
                    'impact': 'Cost Reduction',
                    'timeline': '1 week',
                    'prerequisites': ['Access pattern analysis']
                })
        
        # EBS optimization
        ebs_data = cost_data[cost_data['Service'].str.contains('EBS|Elastic Block', case=False, na=False)]
        if not ebs_data.empty:
            ebs_cost = ebs_data['Cost'].sum()
            # Assume 10% of EBS volumes are unattached
            unattached_savings = ebs_cost * 0.1
            
            if unattached_savings > 20:
                recommendations.append({
                    'type': 'Resource Cleanup',
                    'service': 'Amazon EBS',
                    'title': 'Delete unused EBS volumes',
                    'description': 'Identified unattached EBS volumes consuming storage costs',
                    'potential_savings': unattached_savings,
                    'effort': 'Low',
                    'risk': 'Medium',
                    'implementation': 'Verify and delete unattached volumes after creating snapshots',
                    'status': 'Pending Review',
                    'priority': 'Medium',
                    'category': 'Resource Cleanup',
                    'impact': 'Cost Reduction',
                    'timeline': '3-5 days',
                    'prerequisites': ['Volume usage verification', 'Snapshot creation']
                })
        
        return recommendations
    
    def _generate_reserved_instance_recommendations(self, cost_data: pd.DataFrame) -> List[Dict]:
        """Generate Reserved Instance recommendations"""
        recommendations = []
        
        # EC2 Reserved Instances
        ec2_data = cost_data[cost_data['Service'].str.contains('EC2|Elastic Compute', case=False, na=False)]
        if not ec2_data.empty:
            ec2_cost = ec2_data['Cost'].sum()
            # Assume 60% of EC2 usage is suitable for RIs
            ri_eligible_cost = ec2_cost * 0.6
            ri_savings = ri_eligible_cost * self.savings_estimates['reserved_instances']
            
            if ri_savings > 100:
                recommendations.append({
                    'type': 'Reserved Instances',
                    'service': 'Amazon EC2',
                    'title': 'Purchase EC2 Reserved Instances',
                    'description': 'Convert stable on-demand instances to 1-year reserved instances',
                    'potential_savings': ri_savings,
                    'effort': 'Low',
                    'risk': 'Low',
                    'implementation': 'Purchase reserved instances for predictable workloads',
                    'status': 'Recommended',
                    'priority': 'High',
                    'category': 'Commitment Discounts',
                    'impact': 'Cost Reduction',
                    'timeline': '1 day',
                    'prerequisites': ['Usage pattern analysis', 'Capacity planning']
                })
        
        # RDS Reserved Instances
        rds_data = cost_data[cost_data['Service'].str.contains('RDS|Relational Database', case=False, na=False)]
        if not rds_data.empty:
            rds_cost = rds_data['Cost'].sum()
            rds_ri_savings = rds_cost * self.savings_estimates['reserved_instances']
            
            if rds_ri_savings > 50:
                recommendations.append({
                    'type': 'Reserved Instances',
                    'service': 'Amazon RDS',
                    'title': 'Purchase RDS Reserved Instances',
                    'description': 'Convert production databases to reserved instances',
                    'potential_savings': rds_ri_savings,
                    'effort': 'Low',
                    'risk': 'Low',
                    'implementation': 'Purchase reserved instances for production databases',
                    'status': 'Recommended',
                    'priority': 'High',
                    'category': 'Commitment Discounts',
                    'impact': 'Cost Reduction',
                    'timeline': '1 day',
                    'prerequisites': ['Database usage analysis']
                })
        
        return recommendations
    
    def _generate_cleanup_recommendations(self, cost_data: pd.DataFrame) -> List[Dict]:
        """Generate idle resource cleanup recommendations"""
        recommendations = []
        
        # Simulate idle resource detection
        services_with_idle_resources = ['Amazon EC2', 'Amazon EBS', 'Amazon ELB']
        
        for service in services_with_idle_resources:
            service_data = cost_data[cost_data['Service'].str.contains(service.split()[-1], case=False, na=False)]
            if not service_data.empty:
                service_cost = service_data['Cost'].sum()
                # Assume 5-15% of resources are idle
                idle_percentage = np.random.uniform(0.05, 0.15)
                idle_savings = service_cost * idle_percentage
                
                if idle_savings > 25:
                    resource_type = service.split()[-1]
                    recommendations.append({
                        'type': 'Resource Cleanup',
                        'service': service,
                        'title': f'Remove idle {resource_type} resources',
                        'description': f'Identified {resource_type} resources with no activity for {self.idle_threshold_days}+ days',
                        'potential_savings': idle_savings,
                        'effort': 'Low',
                        'risk': 'Medium',
                        'implementation': f'Review and terminate idle {resource_type} resources after verification',
                        'status': 'Pending Review',
                        'priority': 'Medium',
                        'category': 'Resource Cleanup',
                        'impact': 'Cost Reduction',
                        'timeline': '1 week',
                        'prerequisites': ['Resource usage verification', 'Stakeholder approval']
                    })
        
        return recommendations
    
    def _generate_spot_instance_recommendations(self, cost_data: pd.DataFrame) -> List[Dict]:
        """Generate Spot Instance recommendations"""
        recommendations = []
        
        ec2_data = cost_data[cost_data['Service'].str.contains('EC2|Elastic Compute', case=False, na=False)]
        if not ec2_data.empty:
            ec2_cost = ec2_data['Cost'].sum()
            # Assume 30% of workloads are suitable for spot instances
            spot_eligible_cost = ec2_cost * 0.3
            spot_savings = spot_eligible_cost * self.savings_estimates['spot_instances']
            
            if spot_savings > 75:
                recommendations.append({
                    'type': 'Spot Instances',
                    'service': 'Amazon EC2',
                    'title': 'Migrate suitable workloads to Spot Instances',
                    'description': 'Fault-tolerant and flexible workloads can use Spot Instances for significant savings',
                    'potential_savings': spot_savings,
                    'effort': 'High',
                    'risk': 'Medium',
                    'implementation': 'Identify fault-tolerant workloads and implement Spot Instance strategies',
                    'status': 'Under Evaluation',
                    'priority': 'Medium',
                    'category': 'Compute Optimization',
                    'impact': 'Cost Reduction',
                    'timeline': '4-6 weeks',
                    'prerequisites': ['Workload analysis', 'Fault tolerance design', 'Testing']
                })
        
        return recommendations
    
    def _generate_scaling_recommendations(self, cost_data: pd.DataFrame) -> List[Dict]:
        """Generate auto-scaling and scheduled scaling recommendations"""
        recommendations = []
        
        ec2_data = cost_data[cost_data['Service'].str.contains('EC2|Elastic Compute', case=False, na=False)]
        if not ec2_data.empty:
            ec2_cost = ec2_data['Cost'].sum()
            scaling_savings = ec2_cost * self.savings_estimates['scheduled_scaling']
            
            if scaling_savings > 50:
                recommendations.append({
                    'type': 'Auto Scaling',
                    'service': 'Amazon EC2',
                    'title': 'Implement scheduled scaling for predictable workloads',
                    'description': 'Scale down resources during off-hours and weekends for non-production environments',
                    'potential_savings': scaling_savings,
                    'effort': 'Medium',
                    'risk': 'Low',
                    'implementation': 'Configure Auto Scaling groups with scheduled scaling policies',
                    'status': 'Recommended',
                    'priority': 'Medium',
                    'category': 'Automation',
                    'impact': 'Cost Reduction',
                    'timeline': '2-3 weeks',
                    'prerequisites': ['Workload pattern analysis', 'Auto Scaling setup']
                })
        
        return recommendations
    
    def calculate_total_savings_potential(self, recommendations: List[Dict]) -> Dict:
        """
        Calculate total potential savings from all recommendations
        
        Args:
            recommendations: List of recommendation dictionaries
            
        Returns:
            Dictionary with savings summary
        """
        if not recommendations:
            return {
                'total_potential_savings': 0,
                'monthly_savings': 0,
                'annual_savings': 0,
                'by_category': {},
                'by_priority': {},
                'implementation_effort': {}
            }
        
        total_savings = sum(rec['potential_savings'] for rec in recommendations)
        
        # Group by category
        category_savings = {}
        for rec in recommendations:
            category = rec.get('category', 'Other')
            category_savings[category] = category_savings.get(category, 0) + rec['potential_savings']
        
        # Group by priority
        priority_savings = {}
        for rec in recommendations:
            priority = rec['priority']
            priority_savings[priority] = priority_savings.get(priority, 0) + rec['potential_savings']
        
        # Group by effort
        effort_savings = {}
        for rec in recommendations:
            effort = rec['effort']
            effort_savings[effort] = effort_savings.get(effort, 0) + rec['potential_savings']
        
        return {
            'total_potential_savings': total_savings,
            'monthly_savings': total_savings,  # Assuming monthly data
            'annual_savings': total_savings * 12,
            'by_category': category_savings,
            'by_priority': priority_savings,
            'implementation_effort': effort_savings,
            'recommendation_count': len(recommendations)
        }
    
    def prioritize_recommendations(self, recommendations: List[Dict]) -> List[Dict]:
        """
        Prioritize recommendations based on savings potential, effort, and risk
        
        Args:
            recommendations: List of recommendation dictionaries
            
        Returns:
            Sorted list of recommendations by priority score
        """
        def calculate_priority_score(rec):
            # Scoring factors
            savings_score = min(rec['potential_savings'] / 1000, 10)  # Max 10 points for savings
            
            effort_scores = {'Low': 3, 'Medium': 2, 'High': 1}
            effort_score = effort_scores.get(rec['effort'], 1)
            
            risk_scores = {'Low': 3, 'Medium': 2, 'High': 1}
            risk_score = risk_scores.get(rec['risk'], 1)
            
            priority_scores = {'High': 3, 'Medium': 2, 'Low': 1}
            priority_score = priority_scores.get(rec['priority'], 1)
            
            # Calculate weighted score
            total_score = (savings_score * 0.4 + effort_score * 0.2 + 
                          risk_score * 0.2 + priority_score * 0.2)
            
            return total_score
        
        # Add priority scores
        for rec in recommendations:
            rec['priority_score'] = calculate_priority_score(rec)
        
        # Sort by priority score (descending)
        return sorted(recommendations, key=lambda x: x['priority_score'], reverse=True)
    
    def generate_implementation_plan(self, recommendations: List[Dict]) -> Dict:
        """
        Generate an implementation plan for the recommendations
        
        Args:
            recommendations: List of recommendation dictionaries
            
        Returns:
            Implementation plan with phases and timeline
        """
        # Sort recommendations by priority
        prioritized_recs = self.prioritize_recommendations(recommendations)
        
        # Group into implementation phases
        phases = {
            'Phase 1 (Quick Wins)': [],
            'Phase 2 (Medium Term)': [],
            'Phase 3 (Long Term)': []
        }
        
        for rec in prioritized_recs:
            if rec['effort'] == 'Low' and rec['risk'] == 'Low':
                phases['Phase 1 (Quick Wins)'].append(rec)
            elif rec['effort'] == 'Medium' or rec['risk'] == 'Medium':
                phases['Phase 2 (Medium Term)'].append(rec)
            else:
                phases['Phase 3 (Long Term)'].append(rec)
        
        # Calculate phase savings
        phase_savings = {}
        for phase, recs in phases.items():
            phase_savings[phase] = sum(rec['potential_savings'] for rec in recs)
        
        return {
            'phases': phases,
            'phase_savings': phase_savings,
            'total_recommendations': len(recommendations),
            'estimated_timeline': '3-6 months',
            'implementation_order': [rec['id'] for rec in prioritized_recs]
        }

