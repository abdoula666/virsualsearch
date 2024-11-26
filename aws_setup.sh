#!/bin/bash

# Update package list and install required packages
apt update
apt install -y python3.12-venv python3-pip python3-dev build-essential libssl-dev libffi-dev nginx

# Create directory and set permissions
mkdir -p /home/ubuntu/visual_search
chown -R ubuntu:ubuntu /home/ubuntu/visual_search

# Copy files to the correct location
cp -r /root/visual_search/* /home/ubuntu/visual_search/
chown -R ubuntu:ubuntu /home/ubuntu/visual_search

# Create and set up virtual environment
cd /home/ubuntu/visual_search
rm -rf venv
python3 -m venv venv
chown -R ubuntu:ubuntu venv

# Install Python packages
sudo -u ubuntu bash << 'EOF'
cd /home/ubuntu/visual_search
source venv/bin/activate
pip install --no-cache-dir flask flask-cors tensorflow pillow requests apscheduler scikit-learn opencv-python-headless waitress
deactivate
EOF

# Configure firewall
ufw allow 80
ufw allow 443
ufw allow 5000

# Configure Nginx
cat > /etc/nginx/sites-available/visualsearch << 'EOL'
server {
    listen 80;
    server_name _;

    location / {
        proxy_pass http://127.0.0.1:5000;
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
Description=Visual Search Flask App
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/visual_search
Environment="PATH=/home/ubuntu/visual_search/venv/bin"
ExecStart=/home/ubuntu/visual_search/venv/bin/python app.py
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
