#!/bin/bash

# Update system packages
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv nginx git

# Clone the repository
git clone https://github.com/abdoula666/virsualsearch.git
cd virsualsearch

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
pip install gunicorn

# Configure Nginx
sudo bash -c 'cat > /etc/nginx/sites-available/visual_search << EOL
server {
    listen 80;
    server_name 44.203.252.48;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
    }
}
EOL'

# Create symbolic link and remove default site
sudo ln -sf /etc/nginx/sites-available/visual_search /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default

# Test Nginx configuration
sudo nginx -t

# Restart Nginx
sudo systemctl restart nginx

# Create systemd service file
sudo bash -c 'cat > /etc/systemd/system/visual_search.service << EOL
[Unit]
Description=Visual Search Application
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/virsualsearch
Environment="PATH=/home/ubuntu/virsualsearch/venv/bin"
ExecStart=/home/ubuntu/virsualsearch/venv/bin/gunicorn --workers 3 --bind 127.0.0.1:8000 app:app

[Install]
WantedBy=multi-user.target
EOL'

# Reload systemd
sudo systemctl daemon-reload

# Start and enable the service
sudo systemctl start visual_search
sudo systemctl enable visual_search

# Show service status
sudo systemctl status visual_search
