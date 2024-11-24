import boto3
import argparse
import json
from botocore.exceptions import ClientError

def create_parameter(ssm_client, name, value, description):
    """Create a secure parameter in Parameter Store"""
    try:
        ssm_client.put_parameter(
            Name=name,
            Value=value,
            Type='SecureString',
            Description=description,
            Overwrite=True
        )
        print(f"Successfully created/updated parameter: {name}")
    except ClientError as e:
        print(f"Error creating parameter {name}: {e}")
        raise

def setup_parameters():
    """Set up all required parameters in AWS Parameter Store"""
    parser = argparse.ArgumentParser(description='Set up AWS Parameter Store')
    parser.add_argument('--region', default='us-east-1', help='AWS region')
    parser.add_argument('--env-file', default='.env', help='Environment file path')
    args = parser.parse_args()

    # Initialize AWS client
    ssm_client = boto3.client('ssm', region_name=args.region)

    # Read environment variables from file
    try:
        with open(args.env_file, 'r') as f:
            env_vars = {}
            for line in f:
                if line.strip() and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    env_vars[key] = value

        # Create parameters in Parameter Store
        parameters = {
            '/visualsearch/prod/woocommerce_url': {
                'value': env_vars.get('WOOCOMMERCE_URL', ''),
                'description': 'WooCommerce API URL'
            },
            '/visualsearch/prod/consumer_key': {
                'value': env_vars.get('CONSUMER_KEY', ''),
                'description': 'WooCommerce Consumer Key'
            },
            '/visualsearch/prod/consumer_secret': {
                'value': env_vars.get('CONSUMER_SECRET', ''),
                'description': 'WooCommerce Consumer Secret'
            }
        }

        for name, config in parameters.items():
            create_parameter(ssm_client, name, config['value'], config['description'])

        print("Successfully set up all parameters in AWS Parameter Store")

    except FileNotFoundError:
        print(f"Error: Environment file {args.env_file} not found")
    except Exception as e:
        print(f"Error setting up parameters: {e}")

if __name__ == '__main__':
    setup_parameters()
