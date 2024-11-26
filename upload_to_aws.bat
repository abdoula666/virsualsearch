@echo off
if "%~1"=="" (
    echo Please provide your AWS instance public IP
    echo Usage: upload_to_aws.bat YOUR_EC2_IP
    exit /b 1
)

echo Uploading files to AWS EC2...
scp -i aws_key.pem -r app.py requirements.txt templates static aws_setup.sh ubuntu@%1:~/visual_search/
ssh -i aws_key.pem ubuntu@%1 "chmod +x ~/visual_search/aws_setup.sh && cd ~/visual_search && ./aws_setup.sh"

echo Setup complete! Your server should be running at http://%1
pause
