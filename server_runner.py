import subprocess
import sys
import time
import os
import psutil
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler('server.log'),
        logging.StreamHandler()
    ]
)

def kill_process_on_port(port):
    for proc in psutil.process_iter(['pid', 'name', 'connections']):
        try:
            connections = proc.connections()
            for conn in connections:
                if conn.laddr.port == port:
                    logging.info(f"Killing process {proc.pid} using port {port}")
                    psutil.Process(proc.pid).terminate()
                    time.sleep(2)  # Wait for process to terminate
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

def start_server():
    # Get the directory of this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(current_dir, 'app.py')
    
    while True:
        try:
            # Kill any process using port 59106
            kill_process_on_port(59106)
            
            # Start the server
            logging.info("Starting server...")
            process = subprocess.Popen(
                [sys.executable, app_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=current_dir
            )
            
            # Log the start time
            start_time = datetime.now()
            logging.info(f"Server started at {start_time}")
            
            # Monitor the process
            while True:
                # Check if process is still running
                if process.poll() is not None:
                    logging.error("Server process died, restarting...")
                    break
                
                # Log uptime every hour
                if (datetime.now() - start_time).seconds % 3600 == 0:
                    logging.info("Server running normally")
                
                time.sleep(5)  # Check every 5 seconds
                
        except Exception as e:
            logging.error(f"Error occurred: {str(e)}")
            time.sleep(5)  # Wait before retrying

if __name__ == "__main__":
    logging.info("=== Server Runner Started ===")
    try:
        start_server()
    except KeyboardInterrupt:
        logging.info("Server Runner stopped by user")
        sys.exit(0)
