@echo off
start cmd /k "python app.py"
timeout /t 5
start cmd /k "ngrok http 59106 --basic-auth \"visual:search123\""
