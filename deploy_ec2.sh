#!/bin/bash

# Update system packages
sudo apt-get update
sudo apt-get upgrade -y

# Install Python and required system packages
sudo apt-get install -y python3-pip python3-dev build-essential nginx

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python packages
pip install -r requirements.txt

# Setup Nginx
sudo rm -f /etc/nginx/sites-enabled/default
sudo bash -c 'cat > /etc/nginx/sites-available/visual_search' << EOL
server {
    listen 80;
    server_name _;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
    }
}
EOL

sudo ln -s /etc/nginx/sites-available/visual_search /etc/nginx/sites-enabled
sudo systemctl restart nginx

# Setup systemd service
sudo bash -c 'cat > /etc/systemd/system/visual_search.service' << EOL
[Unit]
Description=Visual Search Flask Application
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/app
Environment="PATH=/home/ubuntu/app/venv/bin"
ExecStart=/home/ubuntu/app/venv/bin/gunicorn -w 4 -b 127.0.0.1:8000 app:app
Restart=always

[Install]
WantedBy=multi-user.target
EOL

# Start and enable the service
sudo systemctl start visual_search
sudo systemctl enable visual_search
