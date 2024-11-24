import boto3
import os
from botocore.exceptions import ClientError

def get_secret():
    """Retrieve secrets from AWS Parameter Store"""
    ssm = boto3.client('ssm', region_name='us-east-1')  # Change region as needed
    
    try:
        # Get parameters
        params = [
            '/visualsearch/woocommerce_url',
            '/visualsearch/consumer_key',
            '/visualsearch/consumer_secret'
        ]
        
        response = ssm.get_parameters(
            Names=params,
            WithDecryption=True
        )
        
        # Set environment variables
        for param in response['Parameters']:
            name = param['Name'].split('/')[-1].upper()
            os.environ[name] = param['Value']
            
        return True
        
    except ClientError as e:
        print(f"Error retrieving parameters: {e}")
        return False

if __name__ == "__main__":
    get_secret()
