import boto3
import argparse

def setup_parameters(woocommerce_url, consumer_key, consumer_secret, region='us-east-1'):
    """Set up secure parameters in AWS Parameter Store"""
    ssm = boto3.client('ssm', region_name=region)
    
    parameters = [
        {
            'name': '/visualsearch/woocommerce_url',
            'value': woocommerce_url,
            'type': 'SecureString',
            'description': 'WooCommerce URL for Visual Search'
        },
        {
            'name': '/visualsearch/consumer_key',
            'value': consumer_key,
            'type': 'SecureString',
            'description': 'WooCommerce API Consumer Key'
        },
        {
            'name': '/visualsearch/consumer_secret',
            'value': consumer_secret,
            'type': 'SecureString',
            'description': 'WooCommerce API Consumer Secret'
        }
    ]
    
    for param in parameters:
        try:
            ssm.put_parameter(
                Name=param['name'],
                Value=param['value'],
                Type=param['type'],
                Description=param['description'],
                Overwrite=True
            )
            print(f"Successfully set parameter: {param['name']}")
        except Exception as e:
            print(f"Error setting parameter {param['name']}: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Set up AWS Parameter Store values')
    parser.add_argument('--url', required=True, help='WooCommerce URL')
    parser.add_argument('--key', required=True, help='Consumer Key')
    parser.add_argument('--secret', required=True, help='Consumer Secret')
    parser.add_argument('--region', default='us-east-1', help='AWS Region')
    
    args = parser.parse_args()
    setup_parameters(args.url, args.key, args.secret, args.region)
