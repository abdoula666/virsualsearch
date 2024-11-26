#!/bin/bash

# Update package list and install required packages
apt update
apt install -y python3-pip python3-dev build-essential libssl-dev libffi-dev nginx

# Create directory and set permissions
mkdir -p /home/ubuntu/visual_search
chown -R ubuntu:ubuntu /home/ubuntu/visual_search

# Copy files to the correct location
cp -r /root/visual_search/* /home/ubuntu/visual_search/
chown -R ubuntu:ubuntu /home/ubuntu/visual_search

# Install Python packages globally
pip install flask flask-cors tensorflow pillow requests apscheduler scikit-learn opencv-python-headless gunicorn

# Configure Nginx
cat > /etc/nginx/sites-available/visualsearch << 'EOL'
server {
    listen 80;
    server_name _;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_connect_timeout 300s;
        proxy_read_timeout 300s;
        client_max_body_size 16M;
    }
}
EOL

# Enable the Nginx site
rm -f /etc/nginx/sites-enabled/default
ln -sf /etc/nginx/sites-available/visualsearch /etc/nginx/sites-enabled/

# Create systemd service
cat > /etc/systemd/system/visualsearch.service << 'EOL'
[Unit]
Description=Visual Search Gunicorn App
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/visual_search
ExecStart=/usr/local/bin/gunicorn -w 4 -b 127.0.0.1:8000 app:app
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOL

# Set proper permissions
chown -R ubuntu:ubuntu /home/ubuntu/visual_search
chmod -R 755 /home/ubuntu/visual_search

# Reload systemd and start services
systemctl daemon-reload
systemctl enable visualsearch
systemctl restart visualsearch
systemctl restart nginx
