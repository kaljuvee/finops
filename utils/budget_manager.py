import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import logging
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

class BudgetManager:
    """
    Budget management and forecasting system for AWS costs
    Handles budget tracking, alerts, and cost forecasting
    """
    
    def __init__(self):
        """Initialize the budget manager"""
        self.logger = logging.getLogger(__name__)
        
        # Alert thresholds (percentage of budget)
        self.alert_thresholds = {
            'warning': 75,    # 75% of budget
            'critical': 90,   # 90% of budget
            'exceeded': 100   # 100% of budget
        }
        
        # Forecast models
        self.linear_model = LinearRegression()
        self.poly_features = PolynomialFeatures(degree=2)
    
    def create_budget(self, name: str, amount: float, period: str = 'monthly',
                     filters: Dict = None, alert_emails: List[str] = None) -> Dict:
        """
        Create a new budget
        
        Args:
            name: Budget name
            amount: Budget amount
            period: Budget period (monthly, quarterly, annual)
            filters: Cost filters (service, tag, account, etc.)
            alert_emails: List of email addresses for alerts
            
        Returns:
            Budget configuration dictionary
        """
        budget = {
            'id': f"budget_{name.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}",
            'name': name,
            'amount': amount,
            'period': period,
            'filters': filters or {},
            'alert_emails': alert_emails or [],
            'created_date': datetime.now(),
            'status': 'Active',
            'alert_thresholds': self.alert_thresholds.copy(),
            'current_spend': 0.0,
            'forecast_spend': 0.0,
            'utilization': 0.0
        }
        
        return budget
    
    def update_budget_spend(self, budget: Dict, cost_data: pd.DataFrame) -> Dict:
        """
        Update budget with current spending data
        
        Args:
            budget: Budget dictionary
            cost_data: DataFrame with cost data
            
        Returns:
            Updated budget dictionary
        """
        # Apply filters to cost data
        filtered_data = self._apply_budget_filters(cost_data, budget['filters'])
        
        # Calculate current spend
        current_spend = filtered_data['Cost'].sum() if not filtered_data.empty else 0.0
        
        # Update budget
        budget['current_spend'] = current_spend
        budget['utilization'] = (current_spend / budget['amount']) * 100 if budget['amount'] > 0 else 0
        budget['remaining_amount'] = budget['amount'] - current_spend
        budget['last_updated'] = datetime.now()
        
        # Generate forecast
        if not filtered_data.empty:
            budget['forecast_spend'] = self._forecast_budget_spend(filtered_data, budget)
        
        return budget
    
    def _apply_budget_filters(self, cost_data: pd.DataFrame, filters: Dict) -> pd.DataFrame:
        """
        Apply budget filters to cost data
        
        Args:
            cost_data: DataFrame with cost data
            filters: Dictionary of filters to apply
            
        Returns:
            Filtered DataFrame
        """
        filtered_data = cost_data.copy()
        
        # Service filter
        if 'services' in filters and filters['services']:
            service_pattern = '|'.join(filters['services'])
            filtered_data = filtered_data[
                filtered_data['Service'].str.contains(service_pattern, case=False, na=False)
            ]
        
        # Tag filters
        if 'tags' in filters and filters['tags']:
            for tag_key, tag_values in filters['tags'].items():
                if tag_key in filtered_data.columns:
                    filtered_data = filtered_data[
                        filtered_data[tag_key].isin(tag_values)
                    ]
        
        # Account filter
        if 'accounts' in filters and filters['accounts']:
            if 'Account' in filtered_data.columns:
                filtered_data = filtered_data[
                    filtered_data['Account'].isin(filters['accounts'])
                ]
        
        # Region filter
        if 'regions' in filters and filters['regions']:
            if 'Region' in filtered_data.columns:
                filtered_data = filtered_data[
                    filtered_data['Region'].isin(filters['regions'])
                ]
        
        return filtered_data
    
    def _forecast_budget_spend(self, cost_data: pd.DataFrame, budget: Dict) -> float:
        """
        Forecast budget spend for the remaining period
        
        Args:
            cost_data: Historical cost data
            budget: Budget dictionary
            
        Returns:
            Forecasted spend amount
        """
        if cost_data.empty or 'Date' not in cost_data.columns:
            return budget['current_spend']
        
        # Prepare data for forecasting
        cost_data = cost_data.sort_values('Date')
        cost_data['Date'] = pd.to_datetime(cost_data['Date'])
        
        # Group by date and sum costs
        daily_costs = cost_data.groupby('Date')['Cost'].sum().reset_index()
        
        if len(daily_costs) < 3:  # Need at least 3 data points
            # Simple linear extrapolation
            days_in_period = self._get_days_in_period(budget['period'])
            days_elapsed = len(daily_costs)
            avg_daily_cost = budget['current_spend'] / days_elapsed if days_elapsed > 0 else 0
            return avg_daily_cost * days_in_period
        
        # Prepare features for ML model
        daily_costs['day_number'] = range(len(daily_costs))
        X = daily_costs[['day_number']].values
        y = daily_costs['Cost'].values
        
        try:
            # Try polynomial regression first
            X_poly = self.poly_features.fit_transform(X)
            self.linear_model.fit(X_poly, y)
            
            # Forecast remaining days
            days_in_period = self._get_days_in_period(budget['period'])
            remaining_days = max(0, days_in_period - len(daily_costs))
            
            if remaining_days > 0:
                future_days = np.arange(len(daily_costs), days_in_period).reshape(-1, 1)
                future_X_poly = self.poly_features.transform(future_days)
                future_costs = self.linear_model.predict(future_X_poly)
                
                # Ensure non-negative predictions
                future_costs = np.maximum(future_costs, 0)
                
                total_forecast = budget['current_spend'] + np.sum(future_costs)
            else:
                total_forecast = budget['current_spend']
            
        except Exception as e:
            self.logger.warning(f"Forecasting failed, using simple extrapolation: {e}")
            # Fallback to simple extrapolation
            days_in_period = self._get_days_in_period(budget['period'])
            days_elapsed = len(daily_costs)
            avg_daily_cost = budget['current_spend'] / days_elapsed if days_elapsed > 0 else 0
            total_forecast = avg_daily_cost * days_in_period
        
        return max(total_forecast, budget['current_spend'])
    
    def _get_days_in_period(self, period: str) -> int:
        """Get number of days in budget period"""
        period_days = {
            'daily': 1,
            'weekly': 7,
            'monthly': 30,
            'quarterly': 90,
            'annual': 365
        }
        return period_days.get(period.lower(), 30)
    
    def check_budget_alerts(self, budget: Dict) -> List[Dict]:
        """
        Check if budget has triggered any alerts
        
        Args:
            budget: Budget dictionary
            
        Returns:
            List of alert dictionaries
        """
        alerts = []
        utilization = budget['utilization']
        
        for alert_type, threshold in budget['alert_thresholds'].items():
            if utilization >= threshold:
                severity = self._get_alert_severity(alert_type)
                
                alert = {
                    'budget_id': budget['id'],
                    'budget_name': budget['name'],
                    'alert_type': alert_type,
                    'severity': severity,
                    'threshold': threshold,
                    'current_utilization': utilization,
                    'current_spend': budget['current_spend'],
                    'budget_amount': budget['amount'],
                    'remaining_amount': budget.get('remaining_amount', 0),
                    'forecast_spend': budget.get('forecast_spend', 0),
                    'timestamp': datetime.now(),
                    'message': self._generate_alert_message(budget, alert_type, threshold)
                }
                
                alerts.append(alert)
        
        return alerts
    
    def _get_alert_severity(self, alert_type: str) -> str:
        """Get alert severity based on type"""
        severity_map = {
            'warning': 'Medium',
            'critical': 'High',
            'exceeded': 'Critical'
        }
        return severity_map.get(alert_type, 'Low')
    
    def _generate_alert_message(self, budget: Dict, alert_type: str, threshold: float) -> str:
        """Generate alert message"""
        messages = {
            'warning': f"Budget '{budget['name']}' has reached {threshold}% utilization",
            'critical': f"Budget '{budget['name']}' has reached {threshold}% utilization - immediate attention required",
            'exceeded': f"Budget '{budget['name']}' has been exceeded!"
        }
        
        base_message = messages.get(alert_type, f"Budget alert for '{budget['name']}'")
        
        details = f"""
Current Spend: ${budget['current_spend']:.2f}
Budget Amount: ${budget['amount']:.2f}
Utilization: {budget['utilization']:.1f}%
Forecast: ${budget.get('forecast_spend', 0):.2f}
        """
        
        return base_message + details
    
    def generate_budget_report(self, budgets: List[Dict]) -> Dict:
        """
        Generate comprehensive budget report
        
        Args:
            budgets: List of budget dictionaries
            
        Returns:
            Budget report dictionary
        """
        if not budgets:
            return {
                'total_budgets': 0,
                'total_budget_amount': 0,
                'total_spend': 0,
                'overall_utilization': 0,
                'budgets_on_track': 0,
                'budgets_at_risk': 0,
                'budgets_exceeded': 0,
                'summary': {}
            }
        
        total_budget_amount = sum(b['amount'] for b in budgets)
        total_spend = sum(b['current_spend'] for b in budgets)
        overall_utilization = (total_spend / total_budget_amount) * 100 if total_budget_amount > 0 else 0
        
        # Categorize budgets
        on_track = sum(1 for b in budgets if b['utilization'] < 75)
        at_risk = sum(1 for b in budgets if 75 <= b['utilization'] < 100)
        exceeded = sum(1 for b in budgets if b['utilization'] >= 100)
        
        # Budget status summary
        budget_summary = []
        for budget in budgets:
            status = 'On Track'
            if budget['utilization'] >= 100:
                status = 'Exceeded'
            elif budget['utilization'] >= 75:
                status = 'At Risk'
            
            budget_summary.append({
                'name': budget['name'],
                'amount': budget['amount'],
                'spend': budget['current_spend'],
                'utilization': budget['utilization'],
                'forecast': budget.get('forecast_spend', 0),
                'status': status
            })
        
        return {
            'total_budgets': len(budgets),
            'total_budget_amount': total_budget_amount,
            'total_spend': total_spend,
            'overall_utilization': overall_utilization,
            'budgets_on_track': on_track,
            'budgets_at_risk': at_risk,
            'budgets_exceeded': exceeded,
            'budget_summary': budget_summary,
            'generated_at': datetime.now()
        }
    
    def create_cost_forecast(self, cost_data: pd.DataFrame, 
                           forecast_days: int = 30) -> pd.DataFrame:
        """
        Create detailed cost forecast
        
        Args:
            cost_data: Historical cost data
            forecast_days: Number of days to forecast
            
        Returns:
            DataFrame with historical and forecasted costs
        """
        if cost_data.empty or 'Date' not in cost_data.columns:
            return pd.DataFrame()
        
        # Prepare historical data
        cost_data = cost_data.copy()
        cost_data['Date'] = pd.to_datetime(cost_data['Date'])
        cost_data = cost_data.sort_values('Date')
        
        # Group by date
        daily_costs = cost_data.groupby('Date')['Cost'].sum().reset_index()
        
        if len(daily_costs) < 3:
            return daily_costs
        
        # Prepare features
        daily_costs['day_number'] = range(len(daily_costs))
        daily_costs['day_of_week'] = daily_costs['Date'].dt.dayofweek
        daily_costs['day_of_month'] = daily_costs['Date'].dt.day
        
        # Features for modeling
        feature_cols = ['day_number', 'day_of_week', 'day_of_month']
        X = daily_costs[feature_cols].values
        y = daily_costs['Cost'].values
        
        try:
            # Fit polynomial model
            X_poly = self.poly_features.fit_transform(X)
            self.linear_model.fit(X_poly, y)
            
            # Generate future dates
            last_date = daily_costs['Date'].max()
            future_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=forecast_days,
                freq='D'
            )
            
            # Prepare future features
            future_data = pd.DataFrame({
                'Date': future_dates,
                'day_number': range(len(daily_costs), len(daily_costs) + forecast_days),
                'day_of_week': future_dates.dayofweek,
                'day_of_month': future_dates.day
            })
            
            future_X = future_data[feature_cols].values
            future_X_poly = self.poly_features.transform(future_X)
            future_costs = self.linear_model.predict(future_X_poly)
            
            # Ensure non-negative predictions
            future_costs = np.maximum(future_costs, 0)
            
            future_data['Cost'] = future_costs
            future_data['Type'] = 'Forecast'
            
            # Combine historical and forecast data
            daily_costs['Type'] = 'Historical'
            forecast_data = pd.concat([
                daily_costs[['Date', 'Cost', 'Type']],
                future_data[['Date', 'Cost', 'Type']]
            ], ignore_index=True)
            
        except Exception as e:
            self.logger.warning(f"Advanced forecasting failed, using simple method: {e}")
            # Simple moving average forecast
            window_size = min(7, len(daily_costs))
            avg_cost = daily_costs['Cost'].tail(window_size).mean()
            
            last_date = daily_costs['Date'].max()
            future_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=forecast_days,
                freq='D'
            )
            
            future_data = pd.DataFrame({
                'Date': future_dates,
                'Cost': [avg_cost] * forecast_days,
                'Type': 'Forecast'
            })
            
            daily_costs['Type'] = 'Historical'
            forecast_data = pd.concat([
                daily_costs[['Date', 'Cost', 'Type']],
                future_data
            ], ignore_index=True)
        
        return forecast_data
    
    def calculate_budget_variance(self, budget: Dict, actual_costs: pd.DataFrame) -> Dict:
        """
        Calculate budget variance analysis
        
        Args:
            budget: Budget dictionary
            actual_costs: Actual cost data
            
        Returns:
            Variance analysis dictionary
        """
        # Apply budget filters
        filtered_costs = self._apply_budget_filters(actual_costs, budget['filters'])
        actual_spend = filtered_costs['Cost'].sum() if not filtered_costs.empty else 0
        
        # Calculate variances
        budget_variance = actual_spend - budget['amount']
        variance_percentage = (budget_variance / budget['amount']) * 100 if budget['amount'] > 0 else 0
        
        # Determine variance status
        if variance_percentage <= -10:
            status = 'Significantly Under Budget'
        elif variance_percentage <= 0:
            status = 'Under Budget'
        elif variance_percentage <= 10:
            status = 'Over Budget'
        else:
            status = 'Significantly Over Budget'
        
        return {
            'budget_name': budget['name'],
            'budget_amount': budget['amount'],
            'actual_spend': actual_spend,
            'variance_amount': budget_variance,
            'variance_percentage': variance_percentage,
            'status': status,
            'analysis_date': datetime.now()
        }

