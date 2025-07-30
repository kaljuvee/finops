import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import logging

class AnomalyDetector:
    """
    Cost anomaly detection system for AWS spending patterns
    Uses statistical methods and machine learning for anomaly detection
    """
    
    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        """
        Initialize the anomaly detector
        
        Args:
            contamination: Expected proportion of anomalies in the data
            random_state: Random state for reproducibility
        """
        self.contamination = contamination
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=random_state
        )
        self.logger = logging.getLogger(__name__)
        
        # Thresholds for different severity levels
        self.severity_thresholds = {
            'Low': 1.5,      # 50% deviation
            'Medium': 2.0,   # 100% deviation  
            'High': 3.0      # 200% deviation
        }
    
    def detect_cost_anomalies(self, cost_data: pd.DataFrame, 
                            service_column: str = 'Service',
                            cost_column: str = 'Cost',
                            date_column: str = 'Date') -> List[Dict]:
        """
        Detect anomalies in cost data using multiple methods
        
        Args:
            cost_data: DataFrame containing cost data
            service_column: Name of the service column
            cost_column: Name of the cost column
            date_column: Name of the date column
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        # Group by service for individual analysis
        for service in cost_data[service_column].unique():
            service_data = cost_data[cost_data[service_column] == service].copy()
            service_data = service_data.sort_values(date_column)
            
            # Statistical anomaly detection
            stat_anomalies = self._detect_statistical_anomalies(
                service_data, service, cost_column, date_column
            )
            anomalies.extend(stat_anomalies)
            
            # Machine learning anomaly detection
            if len(service_data) >= 10:  # Need sufficient data for ML
                ml_anomalies = self._detect_ml_anomalies(
                    service_data, service, cost_column, date_column
                )
                anomalies.extend(ml_anomalies)
        
        return self._deduplicate_anomalies(anomalies)
    
    def _detect_statistical_anomalies(self, data: pd.DataFrame, service: str,
                                    cost_column: str, date_column: str) -> List[Dict]:
        """
        Detect anomalies using statistical methods (Z-score, IQR)
        """
        anomalies = []
        costs = data[cost_column].values
        
        if len(costs) < 3:
            return anomalies
        
        # Calculate statistics
        mean_cost = np.mean(costs)
        std_cost = np.std(costs)
        q1 = np.percentile(costs, 25)
        q3 = np.percentile(costs, 75)
        iqr = q3 - q1
        
        # Z-score method
        z_scores = np.abs((costs - mean_cost) / std_cost) if std_cost > 0 else np.zeros_like(costs)
        
        # IQR method
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        for idx, (_, row) in enumerate(data.iterrows()):
            cost = row[cost_column]
            date = row[date_column]
            z_score = z_scores[idx]
            
            # Check for anomalies
            is_anomaly = False
            anomaly_type = None
            severity = 'Low'
            
            if z_score > 2.5:  # Z-score threshold
                is_anomaly = True
                anomaly_type = 'Statistical Outlier (Z-score)'
                severity = self._calculate_severity(z_score, 'z_score')
            elif cost > upper_bound or cost < lower_bound:  # IQR threshold
                is_anomaly = True
                anomaly_type = 'Statistical Outlier (IQR)'
                severity = self._calculate_severity(cost, 'iqr', mean_cost)
            
            if is_anomaly:
                anomalies.append({
                    'timestamp': pd.to_datetime(date),
                    'service': service,
                    'anomaly_type': anomaly_type,
                    'severity': severity,
                    'description': f'{service} cost anomaly detected: ${cost:.2f} vs baseline ${mean_cost:.2f}',
                    'baseline_cost': mean_cost,
                    'actual_cost': cost,
                    'deviation': ((cost - mean_cost) / mean_cost) * 100 if mean_cost > 0 else 0,
                    'detection_method': 'Statistical',
                    'confidence': min(z_score / 3.0, 1.0),  # Normalize confidence
                    'status': 'Active'
                })
        
        return anomalies
    
    def _detect_ml_anomalies(self, data: pd.DataFrame, service: str,
                           cost_column: str, date_column: str) -> List[Dict]:
        """
        Detect anomalies using machine learning (Isolation Forest)
        """
        anomalies = []
        
        # Prepare features
        data_copy = data.copy()
        data_copy['day_of_week'] = pd.to_datetime(data_copy[date_column]).dt.dayofweek
        data_copy['day_of_month'] = pd.to_datetime(data_copy[date_column]).dt.day
        
        # Create feature matrix
        features = data_copy[[cost_column, 'day_of_week', 'day_of_month']].values
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Detect anomalies
        anomaly_labels = self.isolation_forest.fit_predict(features_scaled)
        anomaly_scores = self.isolation_forest.decision_function(features_scaled)
        
        for idx, (_, row) in enumerate(data.iterrows()):
            if anomaly_labels[idx] == -1:  # Anomaly detected
                cost = row[cost_column]
                date = row[date_column]
                score = anomaly_scores[idx]
                
                # Calculate baseline (median of normal points)
                normal_costs = data_copy[anomaly_labels == 1][cost_column]
                baseline = normal_costs.median() if len(normal_costs) > 0 else cost
                
                severity = self._calculate_ml_severity(score)
                
                anomalies.append({
                    'timestamp': pd.to_datetime(date),
                    'service': service,
                    'anomaly_type': 'ML Detected Anomaly',
                    'severity': severity,
                    'description': f'{service} unusual spending pattern detected: ${cost:.2f}',
                    'baseline_cost': baseline,
                    'actual_cost': cost,
                    'deviation': ((cost - baseline) / baseline) * 100 if baseline > 0 else 0,
                    'detection_method': 'Machine Learning',
                    'confidence': abs(score),
                    'status': 'Active'
                })
        
        return anomalies
    
    def detect_trend_anomalies(self, cost_data: pd.DataFrame,
                             window_size: int = 7) -> List[Dict]:
        """
        Detect trend-based anomalies (sudden changes in spending patterns)
        
        Args:
            cost_data: DataFrame with cost data
            window_size: Window size for trend calculation
            
        Returns:
            List of trend anomalies
        """
        anomalies = []
        
        if len(cost_data) < window_size * 2:
            return anomalies
        
        # Calculate rolling statistics
        cost_data = cost_data.sort_values('Date')
        cost_data['rolling_mean'] = cost_data['Cost'].rolling(window=window_size).mean()
        cost_data['rolling_std'] = cost_data['Cost'].rolling(window=window_size).std()
        
        # Calculate trend changes
        cost_data['trend_change'] = cost_data['rolling_mean'].pct_change()
        
        # Detect significant trend changes
        trend_threshold = 0.3  # 30% change threshold
        
        for idx, row in cost_data.iterrows():
            if pd.isna(row['trend_change']):
                continue
                
            if abs(row['trend_change']) > trend_threshold:
                direction = 'increase' if row['trend_change'] > 0 else 'decrease'
                severity = 'High' if abs(row['trend_change']) > 0.5 else 'Medium'
                
                anomalies.append({
                    'timestamp': pd.to_datetime(row['Date']),
                    'service': 'Overall',
                    'anomaly_type': f'Trend {direction.title()}',
                    'severity': severity,
                    'description': f'Significant cost trend {direction}: {row["trend_change"]*100:.1f}%',
                    'baseline_cost': row['rolling_mean'] - (row['rolling_mean'] * row['trend_change']),
                    'actual_cost': row['Cost'],
                    'deviation': row['trend_change'] * 100,
                    'detection_method': 'Trend Analysis',
                    'confidence': min(abs(row['trend_change']) / 0.5, 1.0),
                    'status': 'Active'
                })
        
        return anomalies
    
    def _calculate_severity(self, value: float, method: str, baseline: float = None) -> str:
        """Calculate severity based on deviation"""
        if method == 'z_score':
            if value > 3.0:
                return 'High'
            elif value > 2.0:
                return 'Medium'
            else:
                return 'Low'
        elif method == 'iqr' and baseline:
            deviation_ratio = abs(value - baseline) / baseline if baseline > 0 else 0
            if deviation_ratio > 2.0:
                return 'High'
            elif deviation_ratio > 1.0:
                return 'Medium'
            else:
                return 'Low'
        
        return 'Low'
    
    def _calculate_ml_severity(self, score: float) -> str:
        """Calculate severity based on ML anomaly score"""
        # Isolation Forest scores are typically between -1 and 1
        # More negative scores indicate stronger anomalies
        abs_score = abs(score)
        
        if abs_score > 0.6:
            return 'High'
        elif abs_score > 0.3:
            return 'Medium'
        else:
            return 'Low'
    
    def _deduplicate_anomalies(self, anomalies: List[Dict]) -> List[Dict]:
        """Remove duplicate anomalies based on timestamp and service"""
        seen = set()
        unique_anomalies = []
        
        for anomaly in anomalies:
            key = (anomaly['timestamp'], anomaly['service'])
            if key not in seen:
                seen.add(key)
                unique_anomalies.append(anomaly)
        
        return sorted(unique_anomalies, key=lambda x: x['timestamp'], reverse=True)
    
    def create_alert_message(self, anomaly: Dict) -> str:
        """
        Create a formatted alert message for an anomaly
        
        Args:
            anomaly: Anomaly dictionary
            
        Returns:
            Formatted alert message
        """
        severity_emoji = {
            'Low': '🟡',
            'Medium': '🟠', 
            'High': '🔴'
        }
        
        emoji = severity_emoji.get(anomaly['severity'], '⚠️')
        
        message = f"""
{emoji} **{anomaly['severity']} Severity Anomaly Detected**

**Service:** {anomaly['service']}
**Type:** {anomaly['anomaly_type']}
**Time:** {anomaly['timestamp'].strftime('%Y-%m-%d %H:%M')}

**Cost Details:**
- Actual Cost: ${anomaly['actual_cost']:.2f}
- Baseline Cost: ${anomaly['baseline_cost']:.2f}
- Deviation: {anomaly['deviation']:.1f}%

**Description:** {anomaly['description']}

**Detection Method:** {anomaly['detection_method']}
**Confidence:** {anomaly['confidence']:.2f}
        """
        
        return message.strip()
    
    def get_anomaly_summary(self, anomalies: List[Dict]) -> Dict:
        """
        Generate summary statistics for detected anomalies
        
        Args:
            anomalies: List of anomaly dictionaries
            
        Returns:
            Summary statistics
        """
        if not anomalies:
            return {
                'total_anomalies': 0,
                'by_severity': {},
                'by_service': {},
                'total_cost_impact': 0,
                'avg_deviation': 0
            }
        
        # Count by severity
        severity_counts = {}
        for anomaly in anomalies:
            severity = anomaly['severity']
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Count by service
        service_counts = {}
        for anomaly in anomalies:
            service = anomaly['service']
            service_counts[service] = service_counts.get(service, 0) + 1
        
        # Calculate cost impact
        total_impact = sum(abs(anomaly['actual_cost'] - anomaly['baseline_cost']) 
                          for anomaly in anomalies)
        
        # Calculate average deviation
        avg_deviation = np.mean([abs(anomaly['deviation']) for anomaly in anomalies])
        
        return {
            'total_anomalies': len(anomalies),
            'by_severity': severity_counts,
            'by_service': service_counts,
            'total_cost_impact': total_impact,
            'avg_deviation': avg_deviation
        }

