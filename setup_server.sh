#!/bin/bash

# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install Python and dependencies
sudo apt-get install -y python3-pip python3-dev build-essential libssl-dev libffi-dev python3-setuptools nginx

# Create a Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python packages
pip install -r requirements.txt
pip install gunicorn

# Setup systemd service
sudo bash -c 'cat > /etc/systemd/system/visualsearch.service << EOL
[Unit]
Description=Visual Search Flask App
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/visual_search
Environment="PATH=/home/ubuntu/visual_search/venv/bin"
ExecStart=/home/ubuntu/visual_search/venv/bin/gunicorn --workers 2 --bind 0.0.0.0:59106 app:app

[Install]
WantedBy=multi-user.target
EOL'

# Setup Nginx
sudo bash -c 'cat > /etc/nginx/sites-available/visualsearch << EOL
server {
    listen 80;
    server_name YOUR_IP_OR_DOMAIN;

    location / {
        proxy_pass http://localhost:59106;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
    }
}
EOL'

# Enable the Nginx site
sudo ln -s /etc/nginx/sites-available/visualsearch /etc/nginx/sites-enabled

# Start services
sudo systemctl start visualsearch
sudo systemctl enable visualsearch
sudo systemctl restart nginx

# Setup firewall
sudo ufw allow 80
sudo ufw allow 443
sudo ufw allow 59106
