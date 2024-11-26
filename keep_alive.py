import subprocess
import time
import sys
import os
import signal
import requests
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    filename='server_status.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

def is_server_running():
    try:
        response = requests.get('http://localhost:59106/', timeout=5)
        return response.status_code == 200
    except:
        return False

def run_server():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(script_dir, 'app.py')
    
    while True:
        if not is_server_running():
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n[{current_time}] Starting server...")
            logging.info("Starting server...")
            
            # Start the server process
            process = subprocess.Popen([sys.executable, app_path],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
            
            # Wait for server to start
            time.sleep(10)
            
            if is_server_running():
                print(f"[{current_time}] Server is running successfully at http://localhost:59106")
                logging.info("Server started successfully")
            else:
                print(f"[{current_time}] Server failed to start, retrying...")
                logging.error("Server failed to start")
                try:
                    process.terminate()
                except:
                    pass
                
        # Check every 30 seconds
        time.sleep(30)

if __name__ == "__main__":
    print("\nStarting Visual Search Server Keep-Alive Monitor")
    print("Press Ctrl+C to stop\n")
    
    try:
        run_server()
    except KeyboardInterrupt:
        print("\nShutting down monitor...")
        logging.info("Monitor stopped by user")
        sys.exit(0)
