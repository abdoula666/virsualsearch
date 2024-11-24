#!/bin/bash

# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install required packages
sudo apt-get install -y python3-pip python3-dev nginx

# Create application directory
sudo mkdir -p /opt/visualsearch
cd /opt/visualsearch

# Create secure environment file
sudo bash -c 'cat > /opt/visualsearch/.env' << 'EOF'
WOOCOMMERCE_URL=your_woocommerce_url
CONSUMER_KEY=your_consumer_key
CONSUMER_SECRET=your_consumer_secret
EOF

# Secure the environment file
sudo chmod 600 /opt/visualsearch/.env
sudo chown www-data:www-data /opt/visualsearch/.env

# Clone your application (replace with your repository URL)
git clone https://github.com/your-username/your-repo.git .

# Install Python requirements
pip3 install -r requirements.txt

# Create systemd service
sudo bash -c 'cat > /etc/systemd/system/visualsearch.service' << 'EOF'
[Unit]
Description=Visual Search Application
After=network.target

[Service]
User=www-data
WorkingDirectory=/opt/visualsearch
Environment="PATH=/opt/visualsearch/venv/bin"
EnvironmentFile=/opt/visualsearch/.env
ExecStart=/usr/bin/python3 app.py

[Install]
WantedBy=multi-user.target
EOF

# Start the service
sudo systemctl start visualsearch
sudo systemctl enable visualsearch

# Configure Nginx
sudo bash -c 'cat > /etc/nginx/sites-available/visualsearch' << 'EOF'
server {
    listen 80;
    server_name your_domain.com;

    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
EOF

# Enable the site
sudo ln -s /etc/nginx/sites-available/visualsearch /etc/nginx/sites-enabled
sudo systemctl restart nginx

echo "Setup complete! Don't forget to:"
echo "1. Edit /opt/visualsearch/.env with your actual API credentials"
echo "2. Update the Nginx configuration with your actual domain"
echo "3. Restart the application: sudo systemctl restart visualsearch"
