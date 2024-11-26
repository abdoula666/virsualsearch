# EC2 instance details
$EC2_IP = "34.204.8.40"
$KEY_FILE = "aws_key.pem"
$USERNAME = "ubuntu"

Write-Host "Starting deployment to AWS EC2..."

# Create remote directory
Write-Host "Creating remote directory..."
ssh -i $KEY_FILE -o StrictHostKeyChecking=no "${USERNAME}@${EC2_IP}" "mkdir -p ~/visual_search"

# Upload application files
Write-Host "Uploading application files..."
$files = @(
    "app.py",
    "requirements.txt",
    "api_key.txt",
    "aws_setup.sh",
    "templates",
    "product_images",
    "featurevector.pkl",
    "filenames.pkl",
    "product_ids.pkl"
)

foreach ($file in $files) {
    Write-Host "Uploading $file..."
    scp -i $KEY_FILE -r $file "${USERNAME}@${EC2_IP}:~/visual_search/"
}

# Make setup script executable and run it
Write-Host "Running setup script..."
ssh -i $KEY_FILE "${USERNAME}@${EC2_IP}" "chmod +x ~/visual_search/aws_setup.sh && cd ~/visual_search && sudo ./aws_setup.sh"

Write-Host "Deployment complete!"
Write-Host "Your application should be accessible at:"
Write-Host "http://${EC2_IP}"
Write-Host "http://ec2-34-204-8-40.compute-1.amazonaws.com"

# Check service status
Write-Host "`nChecking service status..."
ssh -i $KEY_FILE "${USERNAME}@${EC2_IP}" "sudo systemctl status visualsearch"
