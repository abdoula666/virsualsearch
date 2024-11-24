import os
from dotenv import load_dotenv

# Load environment variables from .env file in development
load_dotenv()

def get_woocommerce_api():
    """Get WooCommerce API configuration"""
    return {
        'url': os.environ.get('WOOCOMMERCE_URL'),
        'consumer_key': os.environ.get('CONSUMER_KEY'),
        'consumer_secret': os.environ.get('CONSUMER_SECRET')
    }
