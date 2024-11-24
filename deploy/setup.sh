#!/bin/bash

# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install required packages
sudo apt-get install -y python3-pip python3-dev nginx certbot python3-certbot-nginx

# Install AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Install required Python packages
pip3 install -r requirements.txt
pip3 install awscli boto3 gunicorn

# Create service user
sudo useradd -r -s /bin/false visualsearch

# Create application directories
sudo mkdir -p /opt/visualsearch
sudo chown visualsearch:visualsearch /opt/visualsearch

# Copy application files
sudo cp -r . /opt/visualsearch/
sudo chown -R visualsearch:visualsearch /opt/visualsearch/

# Setup systemd service
sudo tee /etc/systemd/system/visualsearch.service << EOF
[Unit]
Description=Visual Search Application
After=network.target

[Service]
User=visualsearch
Group=visualsearch
WorkingDirectory=/opt/visualsearch
Environment="PATH=/opt/visualsearch/venv/bin"
ExecStart=/opt/visualsearch/venv/bin/gunicorn --workers 3 --bind unix:visualsearch.sock -m 007 app:app

[Install]
WantedBy=multi-user.target
EOF

# Configure Nginx with SSL
sudo tee /etc/nginx/sites-available/visualsearch << EOF
server {
    listen 80;
    listen [::]:80;
    server_name \$DOMAIN_NAME;
    
    # Redirect all HTTP traffic to HTTPS
    location / {
        return 301 https://\$host\$request_uri;
    }
}

server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name \$DOMAIN_NAME;

    # SSL configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_prefer_server_ciphers on;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-ECDSA-CHACHA20-POLY1305:ECDHE-RSA-CHACHA20-POLY1305:DHE-RSA-AES128-GCM-SHA256:DHE-RSA-AES256-GCM-SHA384;
    ssl_session_timeout 1d;
    ssl_session_cache shared:SSL:50m;
    ssl_stapling on;
    ssl_stapling_verify on;
    
    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options SAMEORIGIN;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    
    location / {
        proxy_pass http://unix:/opt/visualsearch/visualsearch.sock;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF

# Enable the site
sudo ln -s /etc/nginx/sites-available/visualsearch /etc/nginx/sites-enabled

# Install Let's Encrypt certificate
sudo certbot --nginx -d \$DOMAIN_NAME --non-interactive --agree-tos --email \$EMAIL --redirect

# Start services
sudo systemctl start visualsearch
sudo systemctl enable visualsearch
sudo systemctl reload nginx
