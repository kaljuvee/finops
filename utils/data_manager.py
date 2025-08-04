import pandas as pd
import os
import sys
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path

# Add the parent directory to the path to import data_generator
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.data_generator import *

class DataManager:
    """Manages data operations including reading from CSV files and generating sample data"""
    
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Define data file mappings
        self.data_files = {
            'cost_data': 'cost_data.csv',
            'service_breakdown': 'service_breakdown.csv',
            'anomaly_data': 'anomaly_data.csv',
            'multivariate_data': 'multivariate_cost_data.csv',
            'forecast_scenarios': 'forecast_scenarios.csv',
            'optimization_recommendations': 'optimization_recommendations.csv',
            'budget_data': 'budget_data.csv',
            'forecast_data': 'forecast_data.csv',
            'cost_allocation': 'cost_allocation.csv',
            'tag_analysis': 'tag_analysis.csv',
            'historical_cost_data': 'historical_cost_data.csv'
        }
        
        # Generate data if files don't exist
        self._ensure_data_exists()
    
    def _ensure_data_exists(self):
        """Generate CSV files if they don't exist"""
        for data_type, filename in self.data_files.items():
            file_path = self.data_dir / filename
            if not file_path.exists():
                self._generate_and_save_data(data_type, file_path)
    
    def _generate_and_save_data(self, data_type, file_path):
        """Generate and save data to CSV file"""
        try:
            if data_type == 'cost_data':
                data = generate_cost_data(30)
            elif data_type == 'service_breakdown':
                data = generate_service_breakdown()
            elif data_type == 'anomaly_data':
                data = generate_anomaly_data(30)
            elif data_type == 'multivariate_data':
                data = generate_multivariate_cost_data(30)
            elif data_type == 'forecast_scenarios':
                data = generate_forecast_scenarios(90)
            elif data_type == 'optimization_recommendations':
                data = pd.DataFrame(generate_optimization_recommendations())
            elif data_type == 'budget_data':
                budget_info = generate_budget_data()
                data = pd.DataFrame(budget_info['budgets'])
            elif data_type == 'forecast_data':
                data = generate_forecast_data(30)
            elif data_type == 'cost_allocation':
                data = pd.DataFrame(generate_cost_allocation_data())
            elif data_type == 'tag_analysis':
                data = pd.DataFrame(generate_tag_analysis_data())
            elif data_type == 'historical_cost_data':
                data = generate_historical_cost_data(365)
            else:
                raise ValueError(f"Unknown data type: {data_type}")
            
            # Save to CSV
            data.to_csv(file_path, index=False)
            print(f"Generated {file_path}")
            
        except Exception as e:
            print(f"Error generating {data_type}: {e}")
    
    def get_data(self, data_type):
        """Get data from CSV file"""
        if data_type not in self.data_files:
            raise ValueError(f"Unknown data type: {data_type}")
        
        file_path = self.data_dir / self.data_files[data_type]
        
        if not file_path.exists():
            self._generate_and_save_data(data_type, file_path)
        
        return pd.read_csv(file_path)
    
    def get_cost_data(self, days=30):
        """Get cost data with optional regeneration"""
        data = self.get_data('cost_data')
        
        # Convert date column to datetime if it exists
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
        
        return data
    
    def get_service_breakdown(self):
        """Get service breakdown data"""
        return self.get_data('service_breakdown')
    
    def get_anomaly_data(self):
        """Get anomaly detection data"""
        data = self.get_data('anomaly_data')
        
        # Convert date column to datetime if it exists
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
        
        return data
    
    def get_multivariate_data(self):
        """Get multivariate cost data"""
        data = self.get_data('multivariate_data')
        
        # Convert date column to datetime if it exists
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
        
        return data
    
    def get_forecast_scenarios(self):
        """Get forecast scenarios data"""
        data = self.get_data('forecast_scenarios')
        
        # Convert date column to datetime if it exists
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
        
        return data
    
    def get_optimization_recommendations(self):
        """Get optimization recommendations"""
        return self.get_data('optimization_recommendations')
    
    def get_budget_data(self):
        """Get budget data"""
        return self.get_data('budget_data')
    
    def get_forecast_data(self):
        """Get forecast data"""
        data = self.get_data('forecast_data')
        
        # Convert date column to datetime if it exists
        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date'])
        
        return data
    
    def get_cost_allocation_data(self):
        """Get cost allocation data"""
        return self.get_data('cost_allocation')
    
    def get_tag_analysis_data(self):
        """Get tag analysis data"""
        return self.get_data('tag_analysis')
    
    def get_historical_cost_data(self):
        """Get historical cost data"""
        data = self.get_data('historical_cost_data')
        
        # Convert date column to datetime if it exists
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
        
        return data
    
    def regenerate_data(self, data_type=None):
        """Regenerate data files"""
        if data_type:
            if data_type not in self.data_files:
                raise ValueError(f"Unknown data type: {data_type}")
            
            file_path = self.data_dir / self.data_files[data_type]
            self._generate_and_save_data(data_type, file_path)
        else:
            # Regenerate all data
            for data_type, filename in self.data_files.items():
                file_path = self.data_dir / filename
                self._generate_and_save_data(data_type, file_path)
    
    def get_data_info(self):
        """Get information about all available data files"""
        info = []
        for data_type, filename in self.data_files.items():
            file_path = self.data_dir / filename
            if file_path.exists():
                try:
                    data = pd.read_csv(file_path)
                    info.append({
                        'type': data_type,
                        'filename': filename,
                        'rows': len(data),
                        'columns': list(data.columns),
                        'size_mb': file_path.stat().st_size / (1024 * 1024),
                        'last_modified': datetime.fromtimestamp(file_path.stat().st_mtime)
                    })
                except Exception as e:
                    info.append({
                        'type': data_type,
                        'filename': filename,
                        'error': str(e)
                    })
            else:
                info.append({
                    'type': data_type,
                    'filename': filename,
                    'status': 'Not generated'
                })
        
        return info
    
    def download_data(self, data_type, format='csv'):
        """Prepare data for download"""
        data = self.get_data(data_type)
        
        if format.lower() == 'csv':
            return data.to_csv(index=False)
        elif format.lower() == 'json':
            return data.to_json(orient='records', indent=2)
        elif format.lower() == 'excel':
            # For Excel, we'll return the DataFrame and let Streamlit handle it
            return data
        else:
            raise ValueError(f"Unsupported format: {format}")

# Global instance
data_manager = DataManager() 