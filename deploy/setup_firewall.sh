#!/bin/bash

# Install UFW if not already installed
sudo apt-get update
sudo apt-get install -y ufw

# Reset UFW to default settings
sudo ufw --force reset

# Set default policies
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Allow SSH (limit to prevent brute force attacks)
sudo ufw limit ssh

# Allow HTTP and HTTPS
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Allow specific IP addresses if needed
# sudo ufw allow from YOUR_IP_ADDRESS to any port 22

# Enable UFW
sudo ufw --force enable

# Show status
sudo ufw status verbose
