"""
Deployment script for the AI Interviewer Agent to Azure Machine Learning
"""

import os
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, command
from azure.ai.ml.entities import (
    Environment,
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    CodeConfiguration,
)
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Azure ML configuration
SUBSCRIPTION_ID = os.environ.get("AZURE_SUBSCRIPTION_ID")
RESOURCE_GROUP = os.environ.get("AZURE_RESOURCE_GROUP")
WORKSPACE_NAME = os.environ.get("AZURE_ML_WORKSPACE")

# Endpoint configuration
ENDPOINT_NAME = "ai-interviewer-endpoint"
DEPLOYMENT_NAME = "ai-interviewer-deployment"

def main():
    # Check for required environment variables
    if not all([SUBSCRIPTION_ID, RESOURCE_GROUP, WORKSPACE_NAME]):
        print("ERROR: Missing required Azure environment variables.")
        print("Please set AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, and AZURE_ML_WORKSPACE.")
        return
    
    # Get Azure ML credentials
    credential = DefaultAzureCredential()
    
    # Connect to Azure ML workspace
    ml_client = MLClient(
        credential=credential,
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME
    )
    
    print(f"Connected to Azure ML workspace: {WORKSPACE_NAME}")
    
    # Create or update the endpoint
    endpoint = ManagedOnlineEndpoint(
        name=ENDPOINT_NAME,
        description="AI Interviewer Agent Endpoint",
        auth_mode="key"
    )
    
    try:
        ml_client.online_endpoints.begin_create_or_update(endpoint).result()
        print(f"Endpoint '{ENDPOINT_NAME}' created or updated successfully.")
    except Exception as e:
        print(f"Error creating endpoint: {str(e)}")
        return
    
    # Create the deployment
    deployment = ManagedOnlineDeployment(
        name=DEPLOYMENT_NAME,
        endpoint_name=ENDPOINT_NAME,
        model=None,  # No model to register, using custom container
        environment=Environment(
            image="mcr.microsoft.com/azureml/curated/pytorch-1.13-cuda11.7:latest",
            conda_file="conda.yml"
        ),
        code_configuration=CodeConfiguration(
            code=".",  # Use current directory
            scoring_script="main.py"
        ),
        instance_type="Standard_DS3_v2",  # Choose appropriate VM size
        instance_count=1
    )
    
    try:
        ml_client.online_deployments.begin_create_or_update(deployment).result()
        print(f"Deployment '{DEPLOYMENT_NAME}' created or updated successfully.")
    except Exception as e:
        print(f"Error creating deployment: {str(e)}")
        return
    
    # Set the deployment as the default for the endpoint
    try:
        ml_client.online_endpoints.begin_update(
            name=ENDPOINT_NAME,
            traffic={DEPLOYMENT_NAME: 100}
        ).result()
        print(f"Traffic set to 100% for deployment '{DEPLOYMENT_NAME}'.")
    except Exception as e:
        print(f"Error updating traffic: {str(e)}")
    
    # Get the endpoint URL
    endpoint_info = ml_client.online_endpoints.get(name=ENDPOINT_NAME)
    print(f"Endpoint URL: {endpoint_info.scoring_uri}")

if __name__ == "__main__":
    main() 