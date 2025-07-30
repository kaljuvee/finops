# FinOps Dashboard - AWS Cost Management MVP

A comprehensive Financial Operations (FinOps) Streamlit application for AWS cost management, monitoring, and optimization.

## 🚀 Features

### 📊 Real-time Cost Monitoring
- Interactive cost visualization dashboards
- Multi-dimensional filtering (Service, Account, Region, Time)
- Daily, weekly, and monthly cost trends
- Service-wise cost breakdown and analysis
- Export capabilities for reports

### 🏷️ Cost Allocation & Tagging
- Automated cost allocation across departments and projects
- Tag-based cost management and reporting
- Resource ownership tracking
- Cost center attribution and analysis

### 🚨 Anomaly Detection & Alerts
- Real-time cost anomaly detection using statistical models
- Configurable alert thresholds and sensitivity levels
- Multi-channel notifications (Email, Slack, SNS)
- Historical anomaly analysis and trends
- Automated incident response workflows

### 💡 Optimization Recommendations
- AI-powered cost optimization suggestions
- Right-sizing recommendations for EC2 instances
- Reserved Instance and Savings Plan analysis
- Unused resource identification
- ROI analysis and implementation tracking

### 📈 Budget Management & Forecasting
- Comprehensive budget tracking and management
- Predictive cost forecasting using machine learning
- Variance analysis and budget alerts
- Multi-level budget hierarchies
- Automated budget approval workflows

## 🛠️ Technology Stack

- **Frontend**: Streamlit 1.29.0
- **Backend**: Python 3.11+
- **Data Visualization**: Plotly, Pandas
- **AWS Integration**: Boto3 (ready for AWS Cost Explorer API)
- **Machine Learning**: Scikit-learn (for anomaly detection)
- **Styling**: Custom CSS with responsive design

## 📁 Project Structure

```
finops-app/
├── Home.py                 # Main dashboard and entry point
├── pages/                  # Streamlit pages
│   ├── 1_Cost_Monitoring.py
│   ├── 2_Cost_Allocation.py
│   ├── 3_Anomaly_Detection.py
│   ├── 4_Optimization.py
│   └── 5_Budget_Management.py
├── utils/                  # Core business logic
│   ├── __init__.py
│   ├── cost_monitor.py     # Cost monitoring utilities
│   ├── data_generator.py   # Sample data generation
│   ├── anomaly_detector.py # Anomaly detection algorithms
│   ├── optimizer.py        # Cost optimization engine
│   └── budget_manager.py   # Budget management logic
├── tests/                  # Test files and results
│   └── test_results.md
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- pip package manager
- AWS CLI configured (for production use)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/finops-dashboard.git
   cd finops-dashboard
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run Home.py
   ```

4. **Access the dashboard**
   Open your browser and navigate to `http://localhost:8501`

### Docker Deployment (Optional)

```bash
# Build the Docker image
docker build -t finops-dashboard .

# Run the container
docker run -p 8501:8501 finops-dashboard
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the root directory:

```env
# AWS Configuration
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=us-east-1

# Alert Configuration
SLACK_WEBHOOK_URL=your_slack_webhook
EMAIL_SMTP_SERVER=smtp.gmail.com
EMAIL_USERNAME=your_email@company.com
EMAIL_PASSWORD=your_app_password

# Application Settings
DEBUG=False
ANOMALY_SENSITIVITY=medium
DEFAULT_CURRENCY=USD
```

### AWS Permissions

The application requires the following AWS permissions:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "ce:GetCostAndUsage",
                "ce:GetDimensionValues",
                "ce:GetReservationCoverage",
                "ce:GetReservationPurchaseRecommendation",
                "ce:GetReservationUtilization",
                "ce:GetRightsizingRecommendation",
                "ce:GetUsageReport",
                "organizations:ListAccounts",
                "budgets:ViewBudget",
                "budgets:ModifyBudget"
            ],
            "Resource": "*"
        }
    ]
}
```

## 📊 Usage Guide

### 1. Dashboard Overview
- View key metrics: monthly spend, daily average, budget utilization
- Monitor cost trends and service breakdowns
- Check recent alerts and optimization opportunities

### 2. Cost Monitoring
- Filter costs by service, account, region, and time period
- Analyze spending patterns and trends
- Export cost reports for stakeholders

### 3. Anomaly Detection
- Configure detection sensitivity and thresholds
- Monitor real-time anomaly alerts
- Investigate cost spikes and unusual patterns

### 4. Optimization
- Review AI-generated cost optimization recommendations
- Track implementation progress and ROI
- Plan optimization roadmap by priority and effort

### 5. Budget Management
- Create and manage budgets across departments/projects
- Set up automated alerts and notifications
- Forecast future costs and variance analysis

## 🔌 AWS Integration

### Production Setup

1. **Replace mock data with AWS Cost Explorer API**
   ```python
   # In utils/cost_monitor.py
   import boto3
   
   ce_client = boto3.client('ce')
   response = ce_client.get_cost_and_usage(
       TimePeriod={
           'Start': '2024-01-01',
           'End': '2024-01-31'
       },
       Granularity='DAILY',
       Metrics=['BlendedCost']
   )
   ```

2. **Configure real-time alerts**
   ```python
   # In utils/anomaly_detector.py
   import boto3
   
   sns_client = boto3.client('sns')
   sns_client.publish(
       TopicArn='arn:aws:sns:us-east-1:123456789012:finops-alerts',
       Message='Cost anomaly detected',
       Subject='FinOps Alert'
   )
   ```

## 🧪 Testing

Run the test suite:

```bash
# Unit tests
python -m pytest tests/

# Integration tests
python -m pytest tests/integration/

# Load tests
python -m pytest tests/load/
```

View test results:
```bash
cat tests/test_results.md
```

## 🚀 Deployment

### Streamlit Cloud

1. Push code to GitHub repository
2. Connect repository to Streamlit Cloud
3. Configure environment variables
4. Deploy with one click

### AWS ECS/Fargate

1. Build and push Docker image to ECR
2. Create ECS task definition
3. Deploy to Fargate cluster
4. Configure load balancer and auto-scaling

### Kubernetes

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: finops-dashboard
spec:
  replicas: 3
  selector:
    matchLabels:
      app: finops-dashboard
  template:
    metadata:
      labels:
        app: finops-dashboard
    spec:
      containers:
      - name: finops-dashboard
        image: your-registry/finops-dashboard:latest
        ports:
        - containerPort: 8501
```

## 🔒 Security

- **Authentication**: Implement OAuth 2.0 or SAML integration
- **Authorization**: Role-based access control (RBAC)
- **Data Encryption**: Encrypt sensitive data at rest and in transit
- **Audit Logging**: Log all user actions and system events
- **Network Security**: Use VPC, security groups, and WAF

## 📈 Monitoring & Observability

- **Application Metrics**: Response time, error rate, user sessions
- **Business Metrics**: Cost savings, budget accuracy, alert response time
- **Infrastructure Metrics**: CPU, memory, network usage
- **Logging**: Structured logging with correlation IDs

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Write comprehensive tests for new features
- Update documentation for API changes
- Use semantic versioning for releases

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: [Wiki](https://github.com/your-username/finops-dashboard/wiki)
- **Issues**: [GitHub Issues](https://github.com/your-username/finops-dashboard/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/finops-dashboard/discussions)
- **Email**: finops-support@company.com

## 🗺️ Roadmap

### Version 2.0 (Q2 2025)
- [ ] Multi-cloud support (Azure, GCP)
- [ ] Advanced ML models for cost prediction
- [ ] Custom dashboard builder
- [ ] API for third-party integrations

### Version 3.0 (Q4 2025)
- [ ] Real-time streaming cost data
- [ ] Advanced governance workflows
- [ ] Mobile application
- [ ] Enterprise SSO integration

## 🙏 Acknowledgments

- AWS Cost Explorer API documentation
- Streamlit community and documentation
- FinOps Foundation best practices
- Open source contributors

## 📊 Screenshots

### Dashboard Overview
![Dashboard](screenshots/dashboard.png)

### Cost Monitoring
![Cost Monitoring](screenshots/cost-monitoring.png)

### Anomaly Detection
![Anomaly Detection](screenshots/anomaly-detection.png)

### Budget Management
![Budget Management](screenshots/budget-management.png)

---

**Built with ❤️ for the FinOps community**

