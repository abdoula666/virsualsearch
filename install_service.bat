@echo off
echo Installing required Python packages...
pip install pywin32
echo.
echo Installing Visual Search Service...
python visual_search_service.py install
echo.
echo Starting Visual Search Service...
python visual_search_service.py start
echo.
echo Service installation complete!
echo The Visual Search server will now run automatically when Windows starts.
pause
