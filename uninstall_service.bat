@echo off
echo Stopping Visual Search Service...
python visual_search_service.py stop
echo.
echo Removing Visual Search Service...
python visual_search_service.py remove
echo.
echo Service uninstallation complete!
pause
