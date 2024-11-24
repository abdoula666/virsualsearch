#!/bin/bash

# Create environment file with restricted permissions
touch /home/ubuntu/app/.env
chmod 600 /home/ubuntu/app/.env

# Read credentials securely from AWS Parameter Store
WOOCOMMERCE_URL=$(aws ssm get-parameter --name "/visual_search/WOOCOMMERCE_URL" --with-decryption --query "Parameter.Value" --output text)
CONSUMER_KEY=$(aws ssm get-parameter --name "/visual_search/CONSUMER_KEY" --with-decryption --query "Parameter.Value" --output text)
CONSUMER_SECRET=$(aws ssm get-parameter --name "/visual_search/CONSUMER_SECRET" --with-decryption --query "Parameter.Value" --output text)

# Write to environment file
cat > /home/ubuntu/app/.env << EOL
WOOCOMMERCE_URL=${WOOCOMMERCE_URL}
CONSUMER_KEY=${CONSUMER_KEY}
CONSUMER_SECRET=${CONSUMER_SECRET}
EOL
