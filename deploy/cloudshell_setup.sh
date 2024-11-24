#!/bin/bash

# Create necessary directories
mkdir -p ~/visualsearch/deploy

# Create .env file template (you'll need to fill in the values)
cat > ~/visualsearch/.env << EOF
WOOCOMMERCE_URL=your_woocommerce_url
CONSUMER_KEY=your_consumer_key
CONSUMER_SECRET=your_consumer_secret
EOF

# Create Parameter Store script
cat > ~/visualsearch/deploy/aws_parameter_store.py << EOF
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
EOF

# Create EC2 role setup script
cat > ~/visualsearch/deploy/setup_ec2_role.py << EOF
import boto3
import json

def create_ec2_role():
    """Create IAM role for EC2 to access Parameter Store"""
    iam = boto3.client('iam')
    
    try:
        role_name = 'VisualSearchEC2Role'
        
        # Define trust relationship policy
        trust_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {
                        "Service": "ec2.amazonaws.com"
                    },
                    "Action": "sts:AssumeRole"
                }
            ]
        }
        
        # Create the IAM role
        iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description='Role for Visual Search EC2 to access Parameter Store'
        )
        
        # Define parameter store access policy
        parameter_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": [
                        "ssm:GetParameter",
                        "ssm:GetParameters",
                        "ssm:GetParametersByPath"
                    ],
                    "Resource": [
                        f"arn:aws:ssm:*:*:parameter/visualsearch/*"
                    ]
                }
            ]
        }
        
        # Create and attach the policy
        policy_name = 'VisualSearchParameterAccess'
        iam.create_policy(
            PolicyName=policy_name,
            PolicyDocument=json.dumps(parameter_policy)
        )
        
        # Get the policy ARN
        policy_arn = f"arn:aws:iam::{iam.get_user()['User']['Arn'].split(':')[4]}:policy/{policy_name}"
        
        # Attach the policy to the role
        iam.attach_role_policy(
            RoleName=role_name,
            PolicyArn=policy_arn
        )
        
        # Create instance profile and add role to it
        iam.create_instance_profile(InstanceProfileName=role_name)
        iam.add_role_to_instance_profile(
            InstanceProfileName=role_name,
            RoleName=role_name
        )
        
        print(f"Successfully created role {role_name} with parameter store access")
        return role_name
        
    except iam.exceptions.EntityAlreadyExistsException:
        print(f"Role {role_name} already exists")
        return role_name
    except Exception as e:
        print(f"Error creating role: {e}")
        raise

if __name__ == '__main__':
    role_name = create_ec2_role()
    print(f"Use this role name when launching your EC2 instance: {role_name}")
EOF

# Make scripts executable
chmod +x ~/visualsearch/deploy/*.py

# Install required packages
pip install boto3

echo "CloudShell setup complete! Now you need to:"
echo "1. Edit ~/visualsearch/.env with your actual API credentials"
echo "2. Run: cd ~/visualsearch"
echo "3. Run: python deploy/aws_parameter_store.py --env-file .env"
echo "4. Run: python deploy/setup_ec2_role.py"
