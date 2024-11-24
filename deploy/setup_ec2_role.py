import boto3
import json
import time

def create_ec2_role():
    """Create IAM role for EC2 to access Parameter Store"""
    iam = boto3.client('iam')
    
    # Create IAM role
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
