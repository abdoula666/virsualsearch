#!/bin/bash

# Update system packages
sudo apt-get update
sudo apt-get install -y python3-pip git

# Clone the repository
git clone https://github.com/abdoula666/virsualsearch.git
cd virsualsearch

# Install Python dependencies
pip3 install -r requirements.txt

# Run the application
nohup python3 app.py > app.log 2>&1 &
