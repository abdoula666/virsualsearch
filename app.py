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
import requests
from base64 import b64encode
from dotenv import load_dotenv
from woocommerce import API

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
    version="v3",
    verify_ssl=False,
    query_string_auth=True  # Use query string authentication
)

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
            
            response = self.wcapi.get("products", params=params)
            
            if response.status_code == 200:
                products = response.json()
                logger.info(f"Successfully fetched {len(products)} products")
                return products
            else:
                logger.error(f"Failed to fetch products. Status code: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return None
        except Exception as e:
            logger.error(f"Error fetching products: {str(e)}")
            return None

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
        self.scheduler.add_job(self.check_new_products, 'interval', minutes=5)
        self.scheduler.start()
        self.category_weights = {}  # Store category importance weights
    
    def check_new_products(self):
        try:
            logger.info("Starting to fetch products from WooCommerce...")
            params = {
                'per_page': 100,
                'status': 'publish',  # Only get published products
                'orderby': 'date',
                'order': 'desc'
            }
            
            if self.last_check:
                params['after'] = self.last_check.isoformat()
            
            logger.info(f"Requesting products with params: {params}")
            response = self.fetch_products()
            
            if response:
                new_products = response
                logger.info(f"Fetched {len(new_products)} products from WooCommerce")
                
                if new_products:
                    for product in new_products:
                        if 'id' not in product:
                            logger.warning(f"Product missing ID: {product}")
                            continue
                            
                        if product['id'] not in [p['id'] for p in self.products]:
                            try:
                                product_data = {
                                    'id': product['id'],
                                    'name': product.get('name', 'Unnamed Product'),
                                    'price': product.get('price', '0.00'),
                                    'image_path': product['images'][0]['src'] if product.get('images') else None,
                                    'product_url': product.get('permalink', ''),
                                    'categories': [cat['name'] for cat in product.get('categories', [])],
                                    'attributes': product.get('attributes', [])
                                }
                                
                                if product_data['image_path']:
                                    logger.info(f"Adding product: {product_data['name']} (ID: {product_data['id']})")
                                    self.products.append(product_data)
                                else:
                                    logger.warning(f"Skipping product {product_data['id']} - No image available")
                            except Exception as e:
                                logger.error(f"Error processing product {product.get('id', 'unknown')}: {str(e)}")
                                continue
                    
                    if self.products:
                        logger.info("Updating features for new products...")
                        self.update_features()
                        logger.info("Updating category weights...")
                        self.update_category_weights()
                        logger.info(f"Product updates completed. Total products: {len(self.products)}")
                    else:
                        logger.warning("No valid products found to process")
                else:
                    logger.info("No new products found")
                
                self.last_check = datetime.now()
            else:
                logger.error(f"Failed to fetch products.")
        except Exception as e:
            logger.error(f"Error checking for new products: {str(e)}", exc_info=True)
    
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

# Initialize product manager and load products immediately
logger.info("Initializing ProductManager and loading initial products...")
product_manager = ProductManager()
product_manager.check_new_products()  # Load products immediately
logger.info(f"Initial product load complete. Loaded {len(product_manager.products)} products.")

@app.route('/')
def home():
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
    with threading.Lock():
        products_loaded = len(product_manager.products)
        features_loaded = len(product_manager.features)
        is_ready = products_loaded > 0 and products_loaded == features_loaded
        return jsonify({
            'ready': is_ready,
            'product_count': products_loaded
        })

@app.route('/health_check')
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    }), 200

@app.route('/search', methods=['POST'])
def search():
    if len(product_manager.products) == 0:
        return jsonify({'error': 'Products still loading, please wait'}), 503

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Get category filter from request if provided
        category_filter = request.form.get('category', None)
        
        # Process uploaded image
        image_bytes = file.read()
        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert('RGB')
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # Extract and normalize query features
        query_features = feature_extractor.predict(img_array, verbose=0)
        query_features = normalize(query_features).flatten()
        
        # Calculate similarities with improved scoring
        similarities = []
        for product in product_manager.products:
            if product['id'] in product_manager.features:
                similarity = calculate_similarity_score(
                    query_features,
                    product_manager.features[product['id']],
                    product['categories'],
                    category_filter
                )
                similarities.append((similarity, product))
        
        # Sort by similarity and get top results
        similarities.sort(key=lambda x: x[0], reverse=True)
        top_results = similarities[:30]
        
        # Format results
        results = []
        for _, product in top_results:
            try:
                price_value = float(product['price'])
                formatted_price = '{:,.0f}'.format(price_value).replace(',', ' ')
                price_display = f"{formatted_price} FCFA"
            except (ValueError, TypeError):
                price_display = product['price']

            results.append({
                'name': product['name'],
                'price': price_display,
                'image_path': product['image_path'],
                'product_url': product['product_url'],
                'categories': product['categories']
            })
        
        return jsonify({'results': results})

    except Exception as e:
        logger.error(f"Error processing search: {str(e)}")
        return jsonify({'error': 'Error processing image'}), 500

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
    # Start the background scheduler
    scheduler = BackgroundScheduler()
    scheduler.add_job(product_manager.check_new_products, 'interval', minutes=5)
    scheduler.start()
    
    # In production, let the production server handle the port
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
