from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
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
import requests
from base64 import b64encode
from waitress import serve
from functools import wraps
import secrets
import tempfile
import pathlib

# Load environment variables
load_dotenv()

# Environment variable validation
WOOCOMMERCE_BASE_URL = os.getenv('WOOCOMMERCE_BASE_URL')
CONSUMER_KEY = os.getenv('WOOCOMMERCE_CONSUMER_KEY')
CONSUMER_SECRET = os.getenv('WOOCOMMERCE_CONSUMER_SECRET')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info(f"Environment Check:")
logger.info(f"Base URL: {WOOCOMMERCE_BASE_URL}")
logger.info(f"Consumer Key: {'Set' if CONSUMER_KEY else 'Not Set'}")
logger.info(f"Consumer Secret: {'Set' if CONSUMER_SECRET else 'Not Set'}")

if not WOOCOMMERCE_BASE_URL:
    WOOCOMMERCE_BASE_URL = "https://cgbshop1.com"
    logger.warning(f"WOOCOMMERCE_BASE_URL not set, using default: {WOOCOMMERCE_BASE_URL}")

if not CONSUMER_KEY or not CONSUMER_SECRET:
    logger.error("WooCommerce credentials not set. Please configure WOOCOMMERCE_CONSUMER_KEY and WOOCOMMERCE_CONSUMER_SECRET")
    CONSUMER_KEY = "ck_da1507a982310e8a29d704df57b4e886b26d528a"
    CONSUMER_SECRET = "cs_2917aeffff79c6bb2427849b617f0c992959f301"
    logger.warning("Using default credentials for development")

# Ensure base URL has correct format
if not WOOCOMMERCE_BASE_URL.startswith(('http://', 'https://')):
    WOOCOMMERCE_BASE_URL = f"https://{WOOCOMMERCE_BASE_URL}"
logger.info(f"Final WooCommerce Base URL: {WOOCOMMERCE_BASE_URL}")

WOOCOMMERCE_API_URL = f"{WOOCOMMERCE_BASE_URL}/wp-json/wc/v3"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

if os.environ.get('RENDER'):
    UPLOAD_FOLDER = tempfile.gettempdir()
else:
    UPLOAD_FOLDER = 'uploads'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Security configuration
ALLOWED_ORIGINS = [
    'http://localhost:59106',
    'http://127.0.0.1:59106',
    'http://192.168.21.2:59106',
    'https://cgbshop1.com',
    'https://visual-search-3gq1.onrender.com'
]

cors = CORS(app, resources={
    r"/*": {
        "origins": ALLOWED_ORIGINS,
        "methods": ["GET", "POST"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if request.endpoint == 'index' or request.endpoint == 'static':
            return f(*args, **kwargs)
        
        api_key = request.headers.get('Authorization')
        if api_key and api_key.startswith('Bearer '):
            api_key = api_key.split('Bearer ')[1]
        
        if not api_key or api_key != os.getenv('API_KEY'):
            return jsonify({'error': 'Invalid or missing API key'}), 401
        
        return f(*args, **kwargs)
    return decorated_function

app.before_request(require_api_key)

@app.errorhandler(Exception)
def handle_error(error):
    logger.error(f"Unhandled error: {str(error)}", exc_info=True)
    return jsonify({'error': str(error)}), 500

# Initialize ResNet model with custom top layer
base_model = ResNet50(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
feature_extractor = tf.keras.Model(inputs=base_model.input, outputs=x)

class ProductManager:
    def __init__(self):
        self.last_check = None
        self.products = []
        self.features = {}
        self.model = None
        self.scheduler = BackgroundScheduler()
        self.category_weights = {}
        self.processing_lock = threading.Lock()
        
        # Initialize immediately but don't block
        threading.Thread(target=self.check_new_products).start()
        
        # Start scheduler after initial load
        self.scheduler.add_job(self.check_new_products, 'interval', seconds=30)
        self.scheduler.start()
        
        logger.info("ProductManager initialized")

    def check_new_products(self):
        """Check for new products and update the database"""
        try:
            with self.processing_lock:
                logger.info("Starting product check...")
                logger.info(f"Using WooCommerce API URL: {WOOCOMMERCE_API_URL}")
                
                # Test API connection first
                try:
                    test_url = f"{WOOCOMMERCE_API_URL}/products/categories"
                    logger.info(f"Testing API connection with URL: {test_url}")
                    
                    test_response = requests.get(
                        test_url,
                        auth=(CONSUMER_KEY, CONSUMER_SECRET),
                        timeout=10,
                        verify=True
                    )
                    test_response.raise_for_status()
                    logger.info("API connection successful")
                except requests.exceptions.RequestException as e:
                    logger.error(f"API connection test failed: {str(e)}")
                    return
                
                params = {
                    'per_page': 100,
                    'status': 'publish',
                    'orderby': 'date',
                    'order': 'desc'
                }
                
                url = f"{WOOCOMMERCE_API_URL}/products"
                response = requests.get(
                    url,
                    params=params,
                    auth=(CONSUMER_KEY, CONSUMER_SECRET),
                    timeout=30,
                    verify=True
                )
                response.raise_for_status()
                
                products = response.json()
                if not products:
                    logger.warning("No products found")
                    return
                
                logger.info(f"Retrieved {len(products)} products from API")
                
                current_ids = set()
                updated = False
                
                for product in products:
                    try:
                        current_ids.add(product['id'])
                        
                        if not product.get('images'):
                            logger.warning(f"Product {product['id']} has no images")
                            continue
                            
                        product_data = {
                            'id': product['id'],
                            'name': product['name'],
                            'price': product['price'],
                            'image_path': product['images'][0]['src'],
                            'product_url': product['permalink'],
                            'categories': [cat['name'] for cat in product.get('categories', [])],
                            'attributes': product.get('attributes', [])
                        }
                        
                        # Add or update product
                        if product['id'] not in [p['id'] for p in self.products]:
                            self.products.append(product_data)
                            updated = True
                            logger.info(f"Added new product: {product_data['name']}")
                        
                        # Update features if needed
                        if product['id'] not in self.features:
                            try:
                                img_response = requests.get(
                                    product_data['image_path'],
                                    timeout=10,
                                    verify=True
                                )
                                img_response.raise_for_status()
                                
                                img = Image.open(io.BytesIO(img_response.content))
                                img = img.convert('RGB')
                                img = img.resize((224, 224))
                                img_array = image.img_to_array(img)
                                img_array = np.expand_dims(img_array, axis=0)
                                img_array = preprocess_input(img_array)
                                
                                features = feature_extractor.predict(img_array)
                                self.features[product['id']] = features.flatten()
                                logger.info(f"Extracted features for product: {product_data['name']}")
                            except Exception as e:
                                logger.error(f"Error processing image for {product_data['name']}: {str(e)}")
                    
                    except Exception as e:
                        logger.error(f"Error processing product: {str(e)}")
                        continue
                
                # Clean up old products
                self.products = [p for p in self.products if p['id'] in current_ids]
                self.features = {k: v for k, v in self.features.items() if k in current_ids}
                
                if updated:
                    self.update_category_weights()
                    logger.info(f"Updated category weights. Total products: {len(self.products)}")
                
                self.last_check = datetime.now()
        
        except Exception as e:
            logger.error(f"Unexpected error in check_new_products: {str(e)}", exc_info=True)

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

product_manager = ProductManager()

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

@app.route('/')
def index():
    return render_template('standalone.html', api_key=os.getenv('API_KEY'))

@app.route('/status')
def get_status():
    """Get the current status of the product database"""
    return jsonify({
        'total_products': len(product_manager.products),
        'products_with_features': len(product_manager.features),
        'categories': list(product_manager.category_weights.keys()),
        'last_check': str(product_manager.last_check) if product_manager.last_check else None
    })

@app.route('/search', methods=['POST'])
@require_api_key
def search():
    if len(product_manager.products) == 0:
        logger.warning("Search attempted but no products are loaded")
        return jsonify({
            'error': 'Product database is empty',
            'status': {
                'total_products': len(product_manager.products),
                'products_with_features': len(product_manager.features),
                'last_check': str(product_manager.last_check) if product_manager.last_check else None
            }
        }), 503

    if 'image' not in request.files:
        logger.error("No image file in request")
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        logger.error("Empty filename in request")
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Process the uploaded image
        img = Image.open(file.stream)
        img = img.convert('RGB')
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Extract features
        query_features = feature_extractor.predict(img_array).flatten()

        # Get query category if provided
        query_category = request.form.get('category', None)

        # Calculate similarities and sort products
        results = []
        for product in product_manager.products:
            if product['id'] in product_manager.features:
                similarity = calculate_similarity_score(
                    query_features,
                    product_manager.features[product['id']],
                    product.get('categories', []),
                    query_category
                )
                results.append({
                    'product': product,
                    'similarity': float(similarity)
                })

        # Sort by similarity
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        logger.info(f"Search completed successfully. Found {len(results)} matches")
        
        return jsonify({
            'results': results[:10],  # Return top 10 results
            'total_matches': len(results),
            'status': {
                'total_products': len(product_manager.products),
                'products_with_features': len(product_manager.features)
            }
        })

    except Exception as e:
        logger.error(f"Error processing search request: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'Error processing image',
            'details': str(e),
            'status': {
                'total_products': len(product_manager.products),
                'products_with_features': len(product_manager.features)
            }
        }), 500

@app.route('/webhook/product-update', methods=['POST'])
@require_api_key
def product_webhook():
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data received'}), 400
        
        # Get the product ID from the webhook
        product_id = data.get('id')
        if not product_id:
            return jsonify({'error': 'No product ID in webhook data'}), 400
        
        # Fetch the specific product
        response = requests.get(
            f"{WOOCOMMERCE_API_URL}/products/{product_id}",
            auth=(CONSUMER_KEY, CONSUMER_SECRET)
        )
        
        if response.status_code == 200:
            product_data = response.json()
            with product_manager.processing_lock:
                product_manager._process_new_products([product_data])
            return jsonify({'success': True}), 200
        else:
            return jsonify({'error': 'Failed to fetch product data'}), 500
            
    except Exception as e:
        logger.error(f"Webhook error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Initialize product manager and start loading products
    product_manager.check_new_products()
    
    # Get port from environment variable or use default
    port = int(os.environ.get('PORT', 10000))
    host = os.environ.get('HOST', '0.0.0.0')
    
    if os.environ.get('RENDER'):
        # Running on Render
        app.run(host=host, port=port)
    else:
        # Local development
        print(f"Server starting on http://{host}:{port}")
        serve(app, host=host, port=port, threads=4)
