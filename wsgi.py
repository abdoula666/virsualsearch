import sys
import os

# Add your project directory to Python path
project_dir = os.path.dirname(os.path.abspath(__file__))
if project_dir not in sys.path:
    sys.path.append(project_dir)

# Import your Flask app
from app import app as application

# This is the WSGI entry point
if __name__ == '__main__':
    application.run()
