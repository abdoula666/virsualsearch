@echo off
echo Installing required package...
pip install psutil
echo.
echo Starting Visual Search Server...
start /min cmd /c "python server_runner.py"
echo Server started! The server will keep running at http://localhost:59106
echo Check server.log for status and details.
echo.
echo Server is accessible at:
ipconfig | findstr "IPv4"
timeout /t 5
