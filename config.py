import os
from pathlib import Path
from dotenv import load_dotenv
import logging
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

def get_aws_parameter(name, region='us-east-1'):
    """Retrieve a parameter from AWS Parameter Store"""
    ssm = boto3.client('ssm', region_name=region)
    try:
        response = ssm.get_parameter(Name=name, WithDecryption=True)
        return response['Parameter']['Value']
    except ClientError as e:
        logger.error(f"Error getting AWS parameter {name}: {e}")
        return None

def load_configuration():
    """
    Load configuration from AWS Parameter Store in production, fallback to .env file in development
    """
    config = {}
    
    # Check if we're running in production (EC2)
    try:
        import requests
        r = requests.get('http://169.254.169.254/latest/meta-data/instance-id', timeout=0.1)
        is_ec2 = r.status_code == 200
    except:
        is_ec2 = False
    
    if is_ec2:
        logger.info("Running in EC2, loading configuration from Parameter Store")
        # Load from AWS Parameter Store
        parameters = [
            '/visualsearch/prod/woocommerce_url',
            '/visualsearch/prod/consumer_key',
            '/visualsearch/prod/consumer_secret'
        ]
        
        for param_name in parameters:
            key = param_name.split('/')[-1].upper()
            value = get_aws_parameter(param_name)
            if value:
                config[key] = value
            else:
                raise EnvironmentError(f"Missing required parameter: {param_name}")
    else:
        logger.info("Running locally, loading configuration from .env file")
        # Load from .env file
        env_path = Path('.') / '.env'
        if env_path.exists():
            load_dotenv(env_path)
        
        config = {
            'WOOCOMMERCE_URL': os.getenv('WOOCOMMERCE_URL'),
            'CONSUMER_KEY': os.getenv('CONSUMER_KEY'),
            'CONSUMER_SECRET': os.getenv('CONSUMER_SECRET'),
        }
    
    # Validate configuration
    missing_vars = [key for key, value in config.items() if not value]
    if missing_vars:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing_vars)}\n"
            "Please set these variables in your environment or .env file"
        )
    
    return config

def get_woocommerce_api():
    """
    Get WooCommerce API configuration securely
    """
    config = load_configuration()
    return {
        'url': config['WOOCOMMERCE_URL'],
        'consumer_key': config['CONSUMER_KEY'],
        'consumer_secret': config['CONSUMER_SECRET']
    }
