from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D
from PIL import Image
import io
import logging
import threading
import time
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, timedelta
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import requests
from base64 import b64encode
from dotenv import load_dotenv
from woocommerce import API
import json
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# WooCommerce configuration
WOOCOMMERCE_URL = os.environ.get('WOOCOMMERCE_URL', '').rstrip('/')  # Remove trailing slash
CONSUMER_KEY = os.environ.get('CONSUMER_KEY', '')
CONSUMER_SECRET = os.environ.get('CONSUMER_SECRET', '')

# Initialize WooCommerce API
wcapi = API(
    url=WOOCOMMERCE_URL,
    consumer_key=CONSUMER_KEY,
    consumer_secret=CONSUMER_SECRET,
    version="wc/v3",
    verify_ssl=False,
    query_string_auth=True,
    wp_api=True,
    timeout=30
)

def test_woocommerce_connection():
    """Test the WooCommerce API connection"""
    try:
        logger.info("Testing WooCommerce connection...")
        logger.info(f"WooCommerce URL: {WOOCOMMERCE_URL}")
        
        # Try to get store info first
        store_response = wcapi.get("")
        logger.info(f"Store info response status: {store_response.status_code}")
        logger.info(f"Store info response: {store_response.text[:500]}")
        
        # Try to get products
        products_response = wcapi.get("products")
        logger.info(f"Products response status: {products_response.status_code}")
        logger.info(f"Products response: {products_response.text[:500]}")
        
        if store_response.status_code == 200 or products_response.status_code == 200:
            logger.info("Successfully connected to WooCommerce API")
            return True
        else:
            logger.error(f"Failed to connect to WooCommerce.")
            logger.error(f"Store response: {store_response.text[:500]}")
            logger.error(f"Products response: {products_response.text[:500]}")
            return False
    except Exception as e:
        logger.error(f"Error testing WooCommerce connection: {str(e)}")
        return False

# Test connection immediately
connection_test = test_woocommerce_connection()
if not connection_test:
    logger.error("Initial WooCommerce connection test failed")

# Test connection before starting
if not test_woocommerce_connection():
    logger.error("Failed to establish WooCommerce connection. Check your credentials and URL.")

logger.info("Initializing application...")
logger.info(f"WooCommerce URL: {WOOCOMMERCE_URL}")

logger.info(f"Starting API tests with base URL: {WOOCOMMERCE_URL}")

def test_url_connection(url, description=""):
    """Test connection to a URL with detailed logging"""
    try:
        response = requests.get(url, verify=False)
        logger.info(f"Response code: {response.status_code}")
        logger.info(f"Response headers: {response.headers}")
        if response.status_code == 200:
            logger.info(f"Response content: \n{response.text[:200]}...")
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Error testing {description}: {str(e)}")
        return False

# Test basic site connectivity
test_url_connection(WOOCOMMERCE_URL, "base site")
test_url_connection(f"{WOOCOMMERCE_URL}/wp-admin", "WordPress admin")

# Test WordPress REST API endpoints
api_endpoints = [
    ("/wp-json", "WordPress REST API root"),
    ("/wp-json/wp/v2/posts", "WordPress posts API"),
    ("/wp-json/wc/v3/products", "WooCommerce products API")
]

for endpoint, desc in api_endpoints:
    url = f"{WOOCOMMERCE_URL}{endpoint}"
    params = {'consumer_key': CONSUMER_KEY, 'consumer_secret': CONSUMER_SECRET} if 'wc/v3' in endpoint else {}
    try:
        logger.info(f"\nTesting {desc}")
        logger.info(f"URL: {url}")
        logger.info(f"Params: {params}")
        response = requests.get(url, params=params, verify=False, timeout=10)
        logger.info(f"Status: {response.status_code}")
        logger.info(f"Headers: {dict(response.headers)}")
        if response.status_code != 404:
            logger.info(f"Content: {response.text[:200]}...")
    except Exception as e:
        logger.error(f"Error testing {desc}: {str(e)}")

class ProductManager:
    def __init__(self):
        self.products = []
        self.feature_vectors = []
        self.last_update = None
        self.wcapi = wcapi
        self.initialization_status = "not_started"
        self.initialization_error = None
        self.last_check = None
        self.features = {}
        self.model = None
        self.category_weights = {}  # Store category importance weights
        # Immediately load products
        self.initialize()
        
    def initialize(self):
        """Initialize the product manager and load initial products"""
        try:
            self.initialization_status = "in_progress"
            logger.info("Starting initial product load...")
            
            # Fetch initial products
            products = self.fetch_products()
            if products:
                self.products = products
                logger.info(f"Successfully loaded {len(products)} initial products")
                
                # Process feature vectors for initial products
                for product in self.products:
                    try:
                        if product.get('images') and len(product['images']) > 0:
                            image_url = product['images'][0]['src']
                            response = requests.get(image_url, verify=False)
                            if response.status_code == 200:
                                img = Image.open(io.BytesIO(response.content))
                                feature_vector = extract_features(img)
                                if feature_vector is not None:
                                    self.features[product['id']] = feature_vector
                                    logger.info(f"Processed features for product {product['id']}")
                    except Exception as e:
                        logger.error(f"Error processing product {product.get('id', 'unknown')}: {str(e)}")
                        continue
                
                self.initialization_status = "completed"
                self.last_update = datetime.now()
                self.last_check = datetime.now()
            else:
                self.initialization_status = "failed"
                self.initialization_error = "Failed to fetch initial products"
                logger.error("Failed to load initial products")
        except Exception as e:
            self.initialization_status = "failed"
            self.initialization_error = str(e)
            logger.error(f"Error during initialization: {str(e)}")
    
    def get_initialization_status(self):
        """Get the current initialization status"""
        return {
            "status": self.initialization_status,
            "error": self.initialization_error,
            "product_count": len(self.products),
            "feature_count": len(self.features),
            "last_update": self.last_update.isoformat() if self.last_update else None
        }
        
    def fetch_products(self):
        """Fetch products from WooCommerce API"""
        try:
            params = {
                'per_page': 100,
                'status': 'publish',
                'orderby': 'date',
                'order': 'desc'
            }
            logger.info(f"Requesting products with params: {params}")
            
            # Try the standard WooCommerce API endpoint first
            response = self.wcapi.get("products", params=params)
            logger.info(f"Products API Response Status: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    products = response.json()
                    if isinstance(products, list):
                        logger.info(f"Successfully fetched {len(products)} products")
                        return products
                    elif isinstance(products, dict) and 'products' in products:
                        # Some WooCommerce installations wrap the products in an object
                        products_list = products['products']
                        logger.info(f"Successfully fetched {len(products_list)} products from wrapped response")
                        return products_list
                    else:
                        logger.error(f"Unexpected response format: {products}")
                        return None
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON response: {str(e)}")
                    logger.error(f"Response content: {response.text[:500]}")
                    return None
            elif response.status_code == 404:
                # Try alternative endpoint
                logger.info("Products endpoint not found, trying alternative endpoint...")
                response = self.wcapi.get("wp-json/wc/v3/products", params=params)
                
                if response.status_code == 200:
                    try:
                        products = response.json()
                        if isinstance(products, list):
                            logger.info(f"Successfully fetched {len(products)} products from alternative endpoint")
                            return products
                        else:
                            logger.error(f"Unexpected response format from alternative endpoint: {products}")
                            return None
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse JSON response from alternative endpoint: {str(e)}")
                        return None
                else:
                    logger.error(f"Alternative endpoint failed. Status code: {response.status_code}")
                    logger.error(f"Response content: {response.text[:500]}")
                    return None
            else:
                logger.error(f"Failed to fetch products. Status code: {response.status_code}")
                logger.error(f"Response content: {response.text[:500]}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching products: {str(e)}")
            return None
            
    def check_new_products(self):
        """Check for new products and update the local cache"""
        try:
            logger.info("Checking for new products...")
            
            # Fetch new products
            new_products = self.fetch_products()
            
            if new_products:
                # Update products list
                self.products = new_products
                logger.info(f"Updated products list with {len(new_products)} products")
                
                # Process new products
                for product in new_products:
                    if product['id'] not in self.features:
                        try:
                            if product.get('images') and len(product['images']) > 0:
                                image_url = product['images'][0]['src']
                                response = requests.get(image_url, verify=False)
                                if response.status_code == 200:
                                    img = Image.open(io.BytesIO(response.content))
                                    feature_vector = extract_features(img)
                                    if feature_vector is not None:
                                        self.features[product['id']] = feature_vector
                                        logger.info(f"Processed features for new product {product['id']}")
                        except Exception as e:
                            logger.error(f"Error processing new product {product.get('id', 'unknown')}: {str(e)}")
                            continue
                
                self.last_check = datetime.now()
                logger.info("Successfully updated products and features")
            else:
                logger.error("Failed to fetch new products")
                
        except Exception as e:
            logger.error(f"Error checking for new products: {str(e)}")
            
    def update_category_weights(self):
        """Update category importance weights based on product distribution"""
        category_counts = {}
        total_products = len(self.products)
        
        # Count products in each category
        for product in self.products:
            for category in product['categories']:
                category_counts[category] = category_counts.get(category, 0) + 1
        
        # Calculate inverse frequency weights
        for category, count in category_counts.items():
            self.category_weights[category] = np.log(total_products / (count + 1))
    
    def update_features(self):
        if not self.model:
            self.model = feature_extractor
        
        for product in self.products:
            if product['id'] not in self.features and product['image_path']:
                try:
                    response = requests.get(product['image_path'])
                    img = Image.open(io.BytesIO(response.content))
                    img = img.convert('RGB')
                    img = img.resize((224, 224))
                    img_array = image.img_to_array(img)
                    img_array = np.expand_dims(img_array, axis=0)
                    img_array = preprocess_input(img_array)
                    
                    # Extract features and normalize
                    features = self.model.predict(img_array, verbose=0)
                    normalized_features = normalize(features)
                    self.features[product['id']] = normalized_features.flatten()
                    
                except Exception as e:
                    logger.error(f"Error processing image for product {product['id']}: {str(e)}")

# Initialize ResNet model with custom top layer
base_model = ResNet50(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
feature_extractor = tf.keras.Model(inputs=base_model.input, outputs=x)

def extract_features(img):
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = feature_extractor.predict(img_array, verbose=0)
    return features.flatten()

# Initialize scheduler
scheduler = BackgroundScheduler()
scheduler.start()

# Initialize product manager
product_manager = ProductManager()

# Add job to check for new products every 5 minutes
scheduler.add_job(
    product_manager.check_new_products,
    'interval',
    minutes=5,
    id='check_new_products',
    replace_existing=True
)

# Add initialization retry job if initial load fails
def retry_initialization():
    if product_manager.initialization_status == "failed":
        logger.info("Retrying product initialization...")
        product_manager.initialize()

scheduler.add_job(
    retry_initialization,
    'interval',
    minutes=1,
    id='retry_initialization',
    replace_existing=True
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/products')
def get_products():
    try:
        # Get products from WooCommerce
        products = product_manager.fetch_products()
        return jsonify(products)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/test_connection')
def test_connection():
    try:
        # Test WooCommerce connection
        response = product_manager.fetch_products()
        if response:
            return jsonify({"status": "success", "message": "Connected to WooCommerce successfully"})
        else:
            return jsonify({"status": "error", "message": "Connection failed"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/status')
def get_status():
    """Get the application status including product initialization"""
    try:
        # Test WooCommerce connection
        test_result = test_woocommerce_connection()
        
        # Get initialization status
        init_status = product_manager.get_initialization_status()
        
        status = {
            'status': 'healthy' if test_result else 'unhealthy',
            'timestamp': datetime.now().isoformat(),
            'woocommerce': {
                'connection': 'ok' if test_result else 'failed',
                'url': WOOCOMMERCE_URL,
                'has_credentials': bool(CONSUMER_KEY and CONSUMER_SECRET)
            },
            'initialization': init_status,
            'environment': {
                'python_version': sys.version,
                'platform': sys.platform,
                'tensorflow_version': tf.__version__
            }
        }
        
        return jsonify(status), 200 if test_result else 503
    except Exception as e:
        logger.error(f"Error in status endpoint: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/health_check')
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    }), 200

@app.route('/search', methods=['POST'])
def search():
    """Search for products using an uploaded image"""
    try:
        if not product_manager.products:
            return jsonify({
                'error': 'Product database is empty',
                'status': product_manager.get_initialization_status()
            }), 503
        
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
            
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
            
        try:
            # Process the uploaded image
            img = Image.open(file.stream)
            query_features = extract_features(img)
            
            # Calculate similarities
            similarities = []
            for product in product_manager.products:
                if product['id'] in product_manager.features:
                    product_features = product_manager.features[product['id']]
                    similarity = cosine_similarity([query_features], [product_features])[0][0]
                    similarities.append((similarity, product))
            
            # Sort by similarity
            similarities.sort(reverse=True)
            
            # Return top 10 results
            results = []
            for similarity, product in similarities[:10]:
                result = {
                    'id': product['id'],
                    'name': product.get('name', 'Unknown'),
                    'similarity': float(similarity),
                    'image_url': product['images'][0]['src'] if product.get('images') else None,
                    'product_url': product.get('permalink'),
                    'price': product.get('price', '0.00'),
                    'categories': [cat['name'] for cat in product.get('categories', [])]
                }
                results.append(result)
            
            return jsonify({
                'results': results,
                'total_products': len(product_manager.products),
                'processed_products': len(similarities)
            })
            
        except Exception as e:
            logger.error(f"Error processing search: {str(e)}")
            return jsonify({'error': 'Error processing image search'}), 500
            
    except Exception as e:
        logger.error(f"Error in search endpoint: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

def calculate_similarity_score(query_features, product_features, product_categories, query_category=None):
    """Calculate weighted similarity score considering visual features and categories"""
    # Base visual similarity using cosine similarity
    visual_similarity = np.dot(query_features, product_features) / (
        np.linalg.norm(query_features) * np.linalg.norm(product_features)
    )
    
    # Category bonus
    category_bonus = 0
    if query_category:
        for category in product_categories:
            if category.lower() in query_category.lower() or query_category.lower() in category.lower():
                category_bonus += product_manager.category_weights.get(category, 0.1)
    
    # Combine scores (70% visual, 30% category when category is provided)
    final_score = visual_similarity * (0.7 if query_category else 1.0) + category_bonus * 0.3
    
    return final_score

if __name__ == '__main__':
    # In production, let the production server handle the port
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
