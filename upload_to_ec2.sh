#!/bin/bash

# Create directory on EC2
ssh -i chat.pem ubuntu@44.203.252.48 "mkdir -p ~/visual_search"

# Upload files
scp -i chat.pem -r ./* ubuntu@44.203.252.48:~/visual_search/

# Upload deployment script
scp -i chat.pem deploy.sh ubuntu@44.203.252.48:~/visual_search/
ssh -i chat.pem ubuntu@44.203.252.48 "chmod +x ~/visual_search/deploy.sh"
