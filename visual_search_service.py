import win32serviceutil
import win32service
import win32event
import servicemanager
import socket
import sys
import os
import subprocess
import logging

class VisualSearchService(win32serviceutil.ServiceFramework):
    _svc_name_ = "VisualSearchService"
    _svc_display_name_ = "Visual Search Service"
    _svc_description_ = "Keeps the Visual Search server running continuously"

    def __init__(self, args):
        win32serviceutil.ServiceFramework.__init__(self, args)
        self.stop_event = win32event.CreateEvent(None, 0, 0, None)
        self.process = None

    def SvcStop(self):
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        win32event.SetEvent(self.stop_event)
        if self.process:
            self.process.terminate()

    def SvcDoRun(self):
        try:
            # Set up logging
            log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'service_log.txt')
            logging.basicConfig(filename=log_file, level=logging.INFO,
                            format='%(asctime)s - %(message)s')
            
            # Get the directory of the script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            app_path = os.path.join(script_dir, 'app.py')
            
            logging.info('Starting Visual Search Service')
            
            # Start the Flask application
            self.process = subprocess.Popen([sys.executable, app_path],
                                        cwd=script_dir,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE)
            
            logging.info('Service started successfully')
            
            # Wait for the stop event
            win32event.WaitForSingleObject(self.stop_event, win32event.INFINITE)
            
        except Exception as e:
            logging.error(f'Service error: {str(e)}')
            raise

if __name__ == '__main__':
    if len(sys.argv) == 1:
        servicemanager.Initialize()
        servicemanager.PrepareToHostSingle(VisualSearchService)
        servicemanager.StartServiceCtrlDispatcher()
    else:
        win32serviceutil.HandleCommandLine(VisualSearchService)
