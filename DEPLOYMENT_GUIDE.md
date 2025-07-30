# FinOps Dashboard Deployment Guide

## 🚀 Deployment Options

### Option 1: Streamlit Cloud (Recommended for MVP)

1. **Push to GitHub**
   ```bash
   # Create a new repository on GitHub
   # Then push your local repository
   git remote add origin https://github.com/your-username/finops-dashboard.git
   git branch -M main
   git push -u origin main
   ```

2. **Deploy to Streamlit Cloud**
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub account
   - Select the repository and branch
   - Set the main file path: `Home.py`
   - Click "Deploy"

3. **Configure Environment Variables** (if needed)
   - Add AWS credentials in Streamlit Cloud secrets
   - Configure alert settings

### Option 2: Local Development

```bash
# Clone and run locally
git clone https://github.com/your-username/finops-dashboard.git
cd finops-dashboard
pip install -r requirements.txt
streamlit run Home.py
```

### Option 3: Docker Deployment

1. **Create Dockerfile**
   ```dockerfile
   FROM python:3.11-slim
   
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   
   COPY . .
   
   EXPOSE 8501
   
   HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
   
   ENTRYPOINT ["streamlit", "run", "Home.py", "--server.port=8501", "--server.address=0.0.0.0"]
   ```

2. **Build and Run**
   ```bash
   docker build -t finops-dashboard .
   docker run -p 8501:8501 finops-dashboard
   ```

### Option 4: AWS ECS/Fargate

1. **Build and Push to ECR**
   ```bash
   aws ecr create-repository --repository-name finops-dashboard
   docker build -t finops-dashboard .
   docker tag finops-dashboard:latest 123456789012.dkr.ecr.us-east-1.amazonaws.com/finops-dashboard:latest
   docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/finops-dashboard:latest
   ```

2. **Create ECS Task Definition**
   ```json
   {
     "family": "finops-dashboard",
     "networkMode": "awsvpc",
     "requiresCompatibilities": ["FARGATE"],
     "cpu": "256",
     "memory": "512",
     "executionRoleArn": "arn:aws:iam::123456789012:role/ecsTaskExecutionRole",
     "containerDefinitions": [
       {
         "name": "finops-dashboard",
         "image": "123456789012.dkr.ecr.us-east-1.amazonaws.com/finops-dashboard:latest",
         "portMappings": [
           {
             "containerPort": 8501,
             "protocol": "tcp"
           }
         ],
         "essential": true,
         "logConfiguration": {
           "logDriver": "awslogs",
           "options": {
             "awslogs-group": "/ecs/finops-dashboard",
             "awslogs-region": "us-east-1",
             "awslogs-stream-prefix": "ecs"
           }
         }
       }
     ]
   }
   ```

3. **Create ECS Service**
   ```bash
   aws ecs create-service \
     --cluster finops-cluster \
     --service-name finops-dashboard \
     --task-definition finops-dashboard \
     --desired-count 2 \
     --launch-type FARGATE \
     --network-configuration "awsvpcConfiguration={subnets=[subnet-12345,subnet-67890],securityGroups=[sg-abcdef],assignPublicIp=ENABLED}"
   ```

## 🔧 Production Configuration

### Environment Variables

```bash
# AWS Configuration
export AWS_ACCESS_KEY_ID="your_access_key"
export AWS_SECRET_ACCESS_KEY="your_secret_key"
export AWS_DEFAULT_REGION="us-east-1"

# Application Settings
export STREAMLIT_SERVER_PORT="8501"
export STREAMLIT_SERVER_ADDRESS="0.0.0.0"
export STREAMLIT_BROWSER_GATHER_USAGE_STATS="false"

# Alert Configuration
export SLACK_WEBHOOK_URL="your_slack_webhook"
export EMAIL_SMTP_SERVER="smtp.gmail.com"
export EMAIL_USERNAME="alerts@company.com"
export EMAIL_PASSWORD="your_app_password"
```

### Streamlit Configuration

Create `.streamlit/config.toml`:

```toml
[server]
port = 8501
address = "0.0.0.0"
maxUploadSize = 200
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false
serverAddress = "your-domain.com"
serverPort = 443

[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
```

### Security Configuration

1. **Enable HTTPS**
   ```bash
   # Using nginx as reverse proxy
   server {
       listen 443 ssl;
       server_name your-domain.com;
       
       ssl_certificate /path/to/certificate.crt;
       ssl_certificate_key /path/to/private.key;
       
       location / {
           proxy_pass http://localhost:8501;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;
       }
   }
   ```

2. **Authentication Setup**
   ```python
   # Add to Home.py for basic auth
   import streamlit_authenticator as stauth
   
   authenticator = stauth.Authenticate(
       names=['Admin', 'FinOps Team'],
       usernames=['admin', 'finops'],
       passwords=['hashed_password1', 'hashed_password2'],
       cookie_name='finops_auth',
       key='random_signature_key',
       cookie_expiry_days=30
   )
   
   name, authentication_status, username = authenticator.login('Login', 'main')
   
   if authentication_status == False:
       st.error('Username/password is incorrect')
   elif authentication_status == None:
       st.warning('Please enter your username and password')
   elif authentication_status:
       # Main application code here
       pass
   ```

## 📊 Monitoring Setup

### Application Monitoring

1. **Health Check Endpoint**
   ```python
   # Add to Home.py
   @st.cache_data
   def health_check():
       return {"status": "healthy", "timestamp": datetime.now().isoformat()}
   
   if st.query_params.get("health"):
       st.json(health_check())
       st.stop()
   ```

2. **Metrics Collection**
   ```python
   # Add to utils/monitoring.py
   import time
   import logging
   
   def track_page_view(page_name):
       logging.info(f"Page viewed: {page_name}")
   
   def track_user_action(action, details=None):
       logging.info(f"User action: {action}, Details: {details}")
   ```

### Infrastructure Monitoring

1. **CloudWatch Logs**
   ```bash
   # Install CloudWatch agent
   aws logs create-log-group --log-group-name /aws/ecs/finops-dashboard
   ```

2. **Application Metrics**
   ```python
   # Add to requirements.txt
   boto3==1.34.0
   
   # Add to utils/metrics.py
   import boto3
   
   cloudwatch = boto3.client('cloudwatch')
   
   def put_metric(metric_name, value, unit='Count'):
       cloudwatch.put_metric_data(
           Namespace='FinOps/Dashboard',
           MetricData=[
               {
                   'MetricName': metric_name,
                   'Value': value,
                   'Unit': unit
               }
           ]
       )
   ```

## 🔄 CI/CD Pipeline

### GitHub Actions

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy FinOps Dashboard

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: '3.11'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest
    - name: Run tests
      run: |
        pytest tests/

  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
    - uses: actions/checkout@v3
    - name: Deploy to Streamlit Cloud
      run: |
        # Trigger Streamlit Cloud deployment
        curl -X POST "${{ secrets.STREAMLIT_WEBHOOK_URL }}"
```

## 🚨 Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Kill process using port 8501
   lsof -ti:8501 | xargs kill -9
   ```

2. **Memory Issues**
   ```bash
   # Increase Docker memory limit
   docker run --memory=2g -p 8501:8501 finops-dashboard
   ```

3. **AWS Permissions**
   ```bash
   # Test AWS credentials
   aws sts get-caller-identity
   aws ce get-cost-and-usage --time-period Start=2024-01-01,End=2024-01-02 --granularity DAILY --metrics BlendedCost
   ```

### Performance Optimization

1. **Caching**
   ```python
   @st.cache_data(ttl=300)  # Cache for 5 minutes
   def load_cost_data():
       # Expensive data loading operation
       pass
   ```

2. **Lazy Loading**
   ```python
   # Load data only when needed
   if 'cost_data' not in st.session_state:
       st.session_state.cost_data = load_cost_data()
   ```

## 📞 Support

- **Documentation**: Check README.md for detailed setup instructions
- **Issues**: Report bugs on GitHub Issues
- **Community**: Join discussions on GitHub Discussions
- **Enterprise Support**: Contact finops-support@company.com

## 🎯 Next Steps

1. **Deploy to Streamlit Cloud** for immediate access
2. **Configure AWS integration** for real data
3. **Set up monitoring and alerts**
4. **Gather user feedback** for improvements
5. **Plan production deployment** strategy

---

**Ready to deploy your FinOps Dashboard! 🚀**

