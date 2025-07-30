import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

# Machine Learning imports
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.svm import OneClassSVM
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.covariance import EllipticEnvelope
    from scipy import stats
    from scipy.signal import find_peaks
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    st.error("Scikit-learn not available. Please install with: pip install scikit-learn scipy")

from data_generator import generate_anomaly_data, generate_multivariate_cost_data

# Page configuration
st.set_page_config(
    page_title="Advanced Anomaly Detection - FinOps",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .anomaly-critical {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-left: 4px solid #dc3545;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .anomaly-warning {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-left: 4px solid #ffc107;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .anomaly-info {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-left: 4px solid #17a2b8;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .model-card {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .performance-summary {
        background: linear-gradient(135deg, #dc3545 0%, #fd7e14 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class AnomalyDetector:
    """Advanced anomaly detection using multiple algorithms"""
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, X):
        """Fit multiple anomaly detection models"""
        if not SKLEARN_AVAILABLE:
            return
        
        # Scale the data
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialize models
        self.models = {
            'Isolation Forest': IsolationForest(contamination=0.1, random_state=42),
            'One-Class SVM': OneClassSVM(nu=0.1),
            'Elliptic Envelope': EllipticEnvelope(contamination=0.1),
            'DBSCAN': DBSCAN(eps=0.5, min_samples=5)
        }
        
        # Fit models
        for name, model in self.models.items():
            if name == 'DBSCAN':
                model.fit(X_scaled)
            else:
                model.fit(X_scaled)
        
        self.is_fitted = True
    
    def predict(self, X):
        """Predict anomalies using ensemble of models"""
        if not self.is_fitted or not SKLEARN_AVAILABLE:
            return np.zeros(len(X))
        
        X_scaled = self.scaler.transform(X)
        predictions = {}
        
        for name, model in self.models.items():
            if name == 'DBSCAN':
                pred = model.fit_predict(X_scaled)
                # Convert DBSCAN output (-1 for outliers, >=0 for normal)
                pred = np.where(pred == -1, -1, 1)
            else:
                pred = model.predict(X_scaled)
            
            predictions[name] = pred
        
        # Ensemble prediction (majority vote)
        pred_array = np.array(list(predictions.values()))
        ensemble_pred = np.where(np.sum(pred_array == -1, axis=0) >= 2, -1, 1)
        
        return ensemble_pred, predictions
    
    def get_anomaly_scores(self, X):
        """Get anomaly scores from different models"""
        if not self.is_fitted or not SKLEARN_AVAILABLE:
            return {}
        
        X_scaled = self.scaler.transform(X)
        scores = {}
        
        for name, model in self.models.items():
            if hasattr(model, 'decision_function'):
                score = model.decision_function(X_scaled)
            elif hasattr(model, 'score_samples'):
                score = model.score_samples(X_scaled)
            else:
                score = np.zeros(len(X))
            
            scores[name] = score
        
        return scores

def detect_statistical_anomalies(data, method='zscore', threshold=3):
    """Detect anomalies using statistical methods"""
    if method == 'zscore':
        z_scores = np.abs(stats.zscore(data))
        return z_scores > threshold
    elif method == 'iqr':
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (data < lower_bound) | (data > upper_bound)
    elif method == 'modified_zscore':
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        modified_z_scores = 0.6745 * (data - median) / mad
        return np.abs(modified_z_scores) > threshold
    else:
        return np.zeros(len(data), dtype=bool)

def main():
    st.title("🔍 Advanced Anomaly Detection")
    st.markdown("Multi-algorithm anomaly detection using machine learning and statistical methods")
    
    # Sidebar for detection controls
    with st.sidebar:
        st.header("🔧 Detection Controls")
        
        # Algorithm selection
        st.subheader("🤖 Algorithm Selection")
        
        detection_methods = st.multiselect(
            "Detection Methods",
            ["Isolation Forest", "One-Class SVM", "Elliptic Envelope", "DBSCAN", "Statistical (Z-Score)", "Statistical (IQR)"],
            default=["Isolation Forest", "Statistical (Z-Score)"]
        )
        
        # Parameters
        st.subheader("⚙️ Parameters")
        
        contamination_rate = st.slider(
            "Expected Contamination Rate",
            min_value=0.01,
            max_value=0.3,
            value=0.1,
            step=0.01,
            format="%.2f"
        )
        
        statistical_threshold = st.slider(
            "Statistical Threshold (Z-Score)",
            min_value=1.0,
            max_value=5.0,
            value=3.0,
            step=0.1
        )
        
        # Feature selection
        st.subheader("📊 Feature Selection")
        
        features_to_use = st.multiselect(
            "Features for Detection",
            ["Cost", "Usage Hours", "Request Count", "Data Transfer", "API Calls"],
            default=["Cost", "Usage Hours"]
        )
        
        # Time window
        time_window = st.selectbox(
            "Analysis Time Window",
            ["Last 7 days", "Last 30 days", "Last 90 days", "Last 6 months"],
            index=1
        )
        
        window_days = {
            "Last 7 days": 7,
            "Last 30 days": 30,
            "Last 90 days": 90,
            "Last 6 months": 180
        }[time_window]
        
        st.markdown("---")
        
        # Real-time settings
        st.subheader("⚡ Real-time Settings")
        
        auto_refresh = st.checkbox("Auto Refresh", value=True)
        refresh_interval = st.selectbox(
            "Refresh Interval",
            ["1 minute", "5 minutes", "15 minutes", "1 hour"],
            index=1
        )
        
        alert_threshold = st.selectbox(
            "Alert Threshold",
            ["Low", "Medium", "High", "Critical"],
            index=1
        )
        
        st.markdown("---")
        
        # Model tuning
        st.subheader("🔧 Model Tuning")
        
        if st.button("🎯 Auto-tune Models"):
            st.success("Models auto-tuned successfully!")
        
        if st.button("🔄 Retrain Models"):
            st.info("Retraining models with latest data...")
    
    # Generate sample data
    anomaly_data = generate_anomaly_data(days=window_days)
    multivariate_data = generate_multivariate_cost_data(days=window_days)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    # Mock anomaly statistics
    total_anomalies = np.random.randint(15, 35)
    critical_anomalies = np.random.randint(2, 8)
    detection_accuracy = np.random.uniform(92, 98)
    false_positive_rate = np.random.uniform(2, 8)
    
    with col1:
        st.metric(
            label="🚨 Total Anomalies",
            value=str(total_anomalies),
            delta=f"+{np.random.randint(1, 5)} from yesterday"
        )
    
    with col2:
        st.metric(
            label="🔴 Critical Anomalies",
            value=str(critical_anomalies),
            delta=f"+{np.random.randint(0, 3)} from yesterday"
        )
    
    with col3:
        st.metric(
            label="🎯 Detection Accuracy",
            value=f"{detection_accuracy:.1f}%",
            delta=f"+{np.random.uniform(0.1, 1.5):.1f}%"
        )
    
    with col4:
        st.metric(
            label="⚠️ False Positive Rate",
            value=f"{false_positive_rate:.1f}%",
            delta=f"-{np.random.uniform(0.1, 0.8):.1f}%"
        )
    
    st.markdown("---")
    
    # Main detection results
    st.subheader("🔍 Anomaly Detection Results")
    
    if SKLEARN_AVAILABLE:
        # Prepare data for ML models
        feature_data = multivariate_data[['cost', 'usage_hours', 'request_count']].values
        
        # Initialize and fit detector
        detector = AnomalyDetector()
        detector.fit(feature_data)
        
        # Get predictions
        ensemble_pred, individual_preds = detector.predict(feature_data)
        anomaly_scores = detector.get_anomaly_scores(feature_data)
        
        # Visualization
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Main anomaly timeline
            fig = go.Figure()
            
            # Normal points
            normal_mask = ensemble_pred == 1
            fig.add_trace(go.Scatter(
                x=multivariate_data.loc[normal_mask, 'date'],
                y=multivariate_data.loc[normal_mask, 'cost'],
                mode='markers',
                name='Normal',
                marker=dict(color='#28a745', size=6),
                hovertemplate='Date: %{x}<br>Cost: $%{y:.2f}<br>Status: Normal<extra></extra>'
            ))
            
            # Anomalous points
            anomaly_mask = ensemble_pred == -1
            fig.add_trace(go.Scatter(
                x=multivariate_data.loc[anomaly_mask, 'date'],
                y=multivariate_data.loc[anomaly_mask, 'cost'],
                mode='markers',
                name='Anomaly',
                marker=dict(color='#dc3545', size=10, symbol='x'),
                hovertemplate='Date: %{x}<br>Cost: $%{y:.2f}<br>Status: Anomaly<extra></extra>'
            ))
            
            fig.update_layout(
                title="Anomaly Detection Timeline",
                xaxis_title="Date",
                yaxis_title="Cost ($)",
                height=500,
                hovermode='closest'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Model performance comparison
            st.markdown("**🏆 Model Performance**")
            
            # Mock performance metrics for each model
            model_performance = {
                'Isolation Forest': {'Precision': 0.89, 'Recall': 0.92, 'F1': 0.90},
                'One-Class SVM': {'Precision': 0.85, 'Recall': 0.88, 'F1': 0.86},
                'Elliptic Envelope': {'Precision': 0.82, 'Recall': 0.85, 'F1': 0.83},
                'DBSCAN': {'Precision': 0.78, 'Recall': 0.81, 'F1': 0.79}
            }
            
            for model, metrics in model_performance.items():
                st.markdown(f"""
                <div class="model-card">
                    <strong>{model}</strong><br>
                    Precision: {metrics['Precision']:.2f}<br>
                    Recall: {metrics['Recall']:.2f}<br>
                    F1-Score: {metrics['F1']:.2f}
                </div>
                """, unsafe_allow_html=True)
    
    else:
        # Fallback to statistical methods
        st.warning("Machine learning libraries not available. Using statistical methods.")
        
        # Statistical anomaly detection
        cost_data = anomaly_data['cost'].values
        z_score_anomalies = detect_statistical_anomalies(cost_data, 'zscore', statistical_threshold)
        iqr_anomalies = detect_statistical_anomalies(cost_data, 'iqr')
        
        # Visualization
        fig = go.Figure()
        
        # Normal points
        normal_mask = ~(z_score_anomalies | iqr_anomalies)
        fig.add_trace(go.Scatter(
            x=anomaly_data.loc[normal_mask, 'date'],
            y=anomaly_data.loc[normal_mask, 'cost'],
            mode='markers',
            name='Normal',
            marker=dict(color='#28a745', size=6)
        ))
        
        # Anomalous points
        anomaly_mask = z_score_anomalies | iqr_anomalies
        fig.add_trace(go.Scatter(
            x=anomaly_data.loc[anomaly_mask, 'date'],
            y=anomaly_data.loc[anomaly_mask, 'cost'],
            mode='markers',
            name='Anomaly',
            marker=dict(color='#dc3545', size=10, symbol='x')
        ))
        
        fig.update_layout(
            title="Statistical Anomaly Detection",
            xaxis_title="Date",
            yaxis_title="Cost ($)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed analysis tabs
    st.markdown("---")
    st.subheader("📊 Detailed Anomaly Analysis")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 Anomaly Details",
        "📈 Feature Analysis", 
        "🎯 Model Comparison",
        "⚙️ Threshold Tuning",
        "📋 Investigation Tools"
    ])
    
    with tab1:
        st.markdown("**Recent Anomalies**")
        
        # Generate mock anomaly details
        anomaly_details = []
        for i in range(10):
            severity = np.random.choice(['Critical', 'High', 'Medium', 'Low'], p=[0.1, 0.2, 0.4, 0.3])
            service = np.random.choice(['EC2', 'S3', 'RDS', 'Lambda', 'CloudFront'])
            
            anomaly_details.append({
                'Timestamp': datetime.now() - timedelta(hours=np.random.randint(1, 48)),
                'Service': service,
                'Severity': severity,
                'Anomaly Score': np.random.uniform(-3, -0.5),
                'Expected': np.random.uniform(100, 500),
                'Actual': np.random.uniform(600, 1200),
                'Deviation': np.random.uniform(150, 400),
                'Status': np.random.choice(['New', 'Investigating', 'Resolved'])
            })
        
        anomaly_df = pd.DataFrame(anomaly_details)
        
        # Display anomalies with color coding
        for _, anomaly in anomaly_df.head(5).iterrows():
            if anomaly['Severity'] == 'Critical':
                css_class = "anomaly-critical"
                icon = "🔴"
            elif anomaly['Severity'] in ['High', 'Medium']:
                css_class = "anomaly-warning"
                icon = "🟡"
            else:
                css_class = "anomaly-info"
                icon = "🔵"
            
            deviation_pct = (anomaly['Deviation'] / anomaly['Expected']) * 100
            
            st.markdown(f"""
            <div class="{css_class}">
                {icon} <strong>{anomaly['Severity']} Severity</strong><br>
                <strong>Service:</strong> {anomaly['Service']}<br>
                <strong>Expected:</strong> ${anomaly['Expected']:.2f} | <strong>Actual:</strong> ${anomaly['Actual']:.2f}<br>
                <strong>Deviation:</strong> +{deviation_pct:.1f}% (${anomaly['Deviation']:.2f})<br>
                <strong>Score:</strong> {anomaly['Anomaly Score']:.2f} | <strong>Status:</strong> {anomaly['Status']}<br>
                <small>{anomaly['Timestamp'].strftime('%Y-%m-%d %H:%M:%S')}</small>
            </div>
            """, unsafe_allow_html=True)
        
        # Full anomaly table
        st.markdown("**📋 Complete Anomaly Log**")
        
        st.dataframe(
            anomaly_df,
            use_container_width=True,
            column_config={
                'Timestamp': st.column_config.DatetimeColumn('Timestamp'),
                'Expected': st.column_config.NumberColumn('Expected ($)', format='$%.2f'),
                'Actual': st.column_config.NumberColumn('Actual ($)', format='$%.2f'),
                'Deviation': st.column_config.NumberColumn('Deviation ($)', format='$%.2f'),
                'Anomaly Score': st.column_config.NumberColumn('Score', format='%.2f')
            }
        )
    
    with tab2:
        st.markdown("**Feature Importance Analysis**")
        
        if SKLEARN_AVAILABLE:
            # Feature importance for anomaly detection
            col1, col2 = st.columns(2)
            
            with col1:
                # Mock feature importance
                features = ['Cost', 'Usage Hours', 'Request Count', 'Data Transfer', 'API Calls']
                importance = [0.35, 0.28, 0.18, 0.12, 0.07]
                
                fig_importance = px.bar(
                    x=features,
                    y=importance,
                    title="Feature Importance for Anomaly Detection",
                    labels={'x': 'Features', 'y': 'Importance Score'}
                )
                
                st.plotly_chart(fig_importance, use_container_width=True)
            
            with col2:
                # Feature correlation with anomalies
                st.markdown("**🔗 Feature Correlations**")
                
                correlation_data = {
                    'Feature': features,
                    'Correlation with Anomalies': [0.78, 0.65, 0.52, 0.41, 0.23],
                    'Statistical Significance': ['***', '***', '**', '*', 'ns']
                }
                
                corr_df = pd.DataFrame(correlation_data)
                st.dataframe(corr_df, use_container_width=True)
                
                st.markdown("""
                **Significance Levels:**
                - *** p < 0.001 (highly significant)
                - ** p < 0.01 (significant)  
                - * p < 0.05 (marginally significant)
                - ns: not significant
                """)
        
        # Feature distribution analysis
        st.markdown("**📊 Feature Distribution Analysis**")
        
        # Generate sample feature data
        normal_cost = np.random.normal(300, 50, 1000)
        anomaly_cost = np.random.normal(800, 100, 100)
        
        fig_dist = go.Figure()
        
        fig_dist.add_trace(go.Histogram(
            x=normal_cost,
            name='Normal',
            opacity=0.7,
            nbinsx=30,
            marker_color='#28a745'
        ))
        
        fig_dist.add_trace(go.Histogram(
            x=anomaly_cost,
            name='Anomalous',
            opacity=0.7,
            nbinsx=30,
            marker_color='#dc3545'
        ))
        
        fig_dist.update_layout(
            title="Cost Distribution: Normal vs Anomalous",
            xaxis_title="Cost ($)",
            yaxis_title="Frequency",
            barmode='overlay'
        )
        
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with tab3:
        st.markdown("**Model Performance Comparison**")
        
        if SKLEARN_AVAILABLE:
            # Detailed model comparison
            models = ['Isolation Forest', 'One-Class SVM', 'Elliptic Envelope', 'DBSCAN', 'Ensemble']
            metrics = ['Precision', 'Recall', 'F1-Score', 'AUC-ROC']
            
            # Mock performance data
            performance_data = np.random.uniform(0.75, 0.95, (len(models), len(metrics)))
            performance_data[-1] = np.mean(performance_data[:-1], axis=0) + 0.02  # Ensemble slightly better
            
            performance_df = pd.DataFrame(performance_data, index=models, columns=metrics)
            
            # Heatmap
            fig_heatmap = px.imshow(
                performance_df.values,
                x=metrics,
                y=models,
                color_continuous_scale='RdYlGn',
                title="Model Performance Heatmap"
            )
            
            fig_heatmap.update_layout(height=400)
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Performance table
            st.dataframe(
                performance_df.round(3),
                use_container_width=True
            )
        
        # Model execution time comparison
        st.markdown("**⏱️ Execution Time Analysis**")
        
        execution_times = {
            'Model': ['Isolation Forest', 'One-Class SVM', 'Elliptic Envelope', 'DBSCAN', 'Statistical'],
            'Training Time (s)': [2.3, 15.7, 1.8, 0.9, 0.1],
            'Prediction Time (ms)': [12, 45, 8, 25, 1],
            'Memory Usage (MB)': [45, 120, 35, 25, 5]
        }
        
        exec_df = pd.DataFrame(execution_times)
        st.dataframe(exec_df, use_container_width=True)
    
    with tab4:
        st.markdown("**Threshold Optimization**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🎯 Current Thresholds**")
            
            current_thresholds = {
                'Isolation Forest': -0.1,
                'One-Class SVM': 0.0,
                'Statistical (Z-Score)': statistical_threshold,
                'Ensemble Vote': 2
            }
            
            for method, threshold in current_thresholds.items():
                st.metric(method, f"{threshold:.2f}")
        
        with col2:
            st.markdown("**📊 Threshold Impact**")
            
            # Threshold sensitivity analysis
            thresholds = np.linspace(1, 5, 50)
            precision_scores = 1 / (1 + np.exp(-(thresholds - 3) * 2))  # Sigmoid curve
            recall_scores = 1 - precision_scores + 0.1
            
            fig_threshold = go.Figure()
            
            fig_threshold.add_trace(go.Scatter(
                x=thresholds,
                y=precision_scores,
                mode='lines',
                name='Precision',
                line=dict(color='#007bff')
            ))
            
            fig_threshold.add_trace(go.Scatter(
                x=thresholds,
                y=recall_scores,
                mode='lines',
                name='Recall',
                line=dict(color='#28a745')
            ))
            
            fig_threshold.update_layout(
                title="Precision-Recall vs Threshold",
                xaxis_title="Threshold",
                yaxis_title="Score"
            )
            
            st.plotly_chart(fig_threshold, use_container_width=True)
        
        # Optimal threshold recommendation
        st.markdown("**💡 Threshold Recommendations**")
        
        recommendations = [
            {"Method": "Isolation Forest", "Current": -0.1, "Recommended": -0.05, "Reason": "Reduce false positives"},
            {"Method": "Statistical Z-Score", "Current": 3.0, "Recommended": 2.5, "Reason": "Increase sensitivity"},
            {"Method": "Ensemble Vote", "Current": 2, "Recommended": 2, "Reason": "Optimal balance"}
        ]
        
        rec_df = pd.DataFrame(recommendations)
        st.dataframe(rec_df, use_container_width=True)
    
    with tab5:
        st.markdown("**Anomaly Investigation Tools**")
        
        # Anomaly drill-down
        st.markdown("**🔍 Anomaly Drill-down**")
        
        selected_anomaly = st.selectbox(
            "Select Anomaly to Investigate",
            ["2024-07-30 14:23:45 - EC2 Cost Spike", "2024-07-30 09:15:22 - S3 Usage Anomaly", "2024-07-29 18:47:33 - Lambda Invocation Spike"]
        )
        
        if selected_anomaly:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📊 Anomaly Context**")
                
                context_data = {
                    'Metric': ['Service', 'Account', 'Region', 'Resource ID', 'Cost Center'],
                    'Value': ['Amazon EC2', 'Production', 'us-east-1', 'i-0123456789abcdef0', 'Engineering'],
                    'Normal Range': ['$200-400', 'N/A', 'N/A', 'N/A', '$1000-2000'],
                    'Anomaly Value': ['$850', 'N/A', 'N/A', 'N/A', '$2500']
                }
                
                context_df = pd.DataFrame(context_data)
                st.dataframe(context_df, use_container_width=True)
            
            with col2:
                st.markdown("**🕒 Timeline Analysis**")
                
                # Mock timeline data
                timeline_hours = list(range(-6, 7))
                timeline_costs = [200, 210, 195, 205, 850, 820, 780, 300, 250, 220, 200, 195, 190]
                
                fig_timeline = px.line(
                    x=timeline_hours,
                    y=timeline_costs,
                    title="Cost Timeline (Hours from Anomaly)",
                    labels={'x': 'Hours from Anomaly', 'y': 'Cost ($)'}
                )
                
                fig_timeline.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Anomaly")
                
                st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Root cause analysis
        st.markdown("**🔍 Root Cause Analysis**")
        
        root_causes = [
            {"Factor": "Instance Type Change", "Likelihood": "High", "Impact": "High", "Evidence": "m5.large → m5.2xlarge"},
            {"Factor": "Auto Scaling Event", "Likelihood": "Medium", "Impact": "Medium", "Evidence": "Scale out from 2 to 8 instances"},
            {"Factor": "Data Processing Job", "Likelihood": "Low", "Impact": "High", "Evidence": "No unusual job activity"},
            {"Factor": "Network Traffic Spike", "Likelihood": "Medium", "Impact": "Low", "Evidence": "15% increase in data transfer"}
        ]
        
        root_cause_df = pd.DataFrame(root_causes)
        
        st.dataframe(
            root_cause_df,
            use_container_width=True,
            column_config={
                'Likelihood': st.column_config.SelectboxColumn(
                    'Likelihood',
                    options=['Low', 'Medium', 'High']
                ),
                'Impact': st.column_config.SelectboxColumn(
                    'Impact', 
                    options=['Low', 'Medium', 'High']
                )
            }
        )
        
        # Investigation actions
        st.markdown("**⚡ Quick Actions**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("📧 Alert Team"):
                st.success("Team notified!")
        
        with col2:
            if st.button("🔒 Create Ticket"):
                st.success("Ticket created!")
        
        with col3:
            if st.button("📊 Generate Report"):
                st.success("Report generated!")
        
        with col4:
            if st.button("✅ Mark Resolved"):
                st.success("Anomaly marked as resolved!")

if __name__ == "__main__":
    main()

