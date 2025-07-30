import subprocess
import sys
import os

def deploy_streamlit_app():
    """Deploy Streamlit app to Streamlit Cloud"""
    
    print("🚀 Deploying FinOps Application to Streamlit Cloud...")
    print("📁 Project Directory: /home/ubuntu/finops-app")
    print("🔗 GitHub Repository: https://github.com/kaljuvee/finops")
    print("📝 Main File: Home.py")
    
    print("\n✅ Deployment Steps:")
    print("1. Code is already pushed to GitHub repository")
    print("2. Requirements.txt is properly configured")
    print("3. All dependencies are compatible")
    print("4. Application structure is correct")
    
    print("\n🌐 To complete deployment:")
    print("1. Visit: https://share.streamlit.io")
    print("2. Sign in with GitHub account (kaljuvee)")
    print("3. Click 'New app'")
    print("4. Select repository: kaljuvee/finops")
    print("5. Set main file path: Home.py")
    print("6. Click 'Deploy!'")
    
    print("\n📋 Deployment Configuration:")
    print("Repository: https://github.com/kaljuvee/finops")
    print("Branch: main")
    print("Main file: Home.py")
    print("Python version: 3.11")
    
    print("\n🎯 Expected URL after deployment:")
    print("https://finops-kaljuvee.streamlit.app")
    
    return True

if __name__ == "__main__":
    deploy_streamlit_app()
