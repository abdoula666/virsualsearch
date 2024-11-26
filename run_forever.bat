@echo off
echo Starting Visual Search Server...
start /min cmd /c "python keep_alive.py"
echo Server monitor started! The server will keep running at http://localhost:59106
echo Check server_status.log for details.
timeout /t 5
