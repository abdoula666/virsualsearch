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
from waitress import serve
from functools import wraps
import secrets
import tempfile
import pathlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

if os.environ.get('RENDER'):
    UPLOAD_FOLDER = tempfile.gettempdir()
else:
    UPLOAD_FOLDER = 'uploads'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# WooCommerce API Configuration
WOOCOMMERCE_URL = "https://cgbshop1.com/wp-json/wc/v3"
CONSUMER_KEY = "ck_da1507a982310e8a29d704df57b4e886b26d528a"
CONSUMER_SECRET = "cs_2917aeffff79c6bb2427849b617f0c992959f301"

# Generate a secure API key if it doesn't exist
API_KEY_FILE = "api_key.txt"
if not os.path.exists(API_KEY_FILE):
    API_KEY = secrets.token_urlsafe(32)
    with open(API_KEY_FILE, "w") as f:
        f.write(API_KEY)
else:
    with open(API_KEY_FILE, "r") as f:
        API_KEY = f.read().strip()

# Security configuration
ALLOWED_ORIGINS = [
    'http://localhost:59106',
    'http://127.0.0.1:59106',
    'http://192.168.21.2:59106',
<<<<<<< HEAD
    'https://cgbshop1.com'
]

cors = CORS(app, resources={
    r"/*": {
        "origins": ALLOWED_ORIGINS,
        "methods": ["GET", "POST"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})
=======
    'https://cgbshop1.com',
    'http://34.204.8.40',
    'http://ec2-34-204-8-40.compute-1.amazonaws.com',
    '*'  # Allow all origins when using public server
}

cors = CORS(app, origins=ALLOWED_ORIGINS)
>>>>>>> cf967bfa3abd668c28d4d5e380ac46a1635f8e54

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if request.endpoint == 'index' or request.endpoint == 'static':
            return f(*args, **kwargs)
        
        api_key = request.headers.get('Authorization')
        if api_key and api_key.startswith('Bearer '):
            api_key = api_key.split('Bearer ')[1]
        
        if not api_key or api_key != API_KEY:
            return jsonify({'error': 'Invalid or missing API key'}), 401
        
        return f(*args, **kwargs)
    return decorated_function

app.before_request(require_api_key)

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
        # Check every 30 seconds
        self.scheduler.add_job(self.check_new_products, 'interval', seconds=30)
        self.scheduler.start()
        self.category_weights = {}
        self.processing_lock = threading.Lock()
        # Initialize products immediately
        self.check_new_products()
        
    def check_new_products(self):
        with self.processing_lock:
            try:
                logger.info("Checking for new products...")
                params = {
                    'per_page': 100,
                    'orderby': 'modified',
                    'order': 'desc'
                }
                
                response = requests.get(
                    f"{WOOCOMMERCE_URL}/products",
                    params=params,
                    auth=(CONSUMER_KEY, CONSUMER_SECRET),
                    timeout=30  # Add timeout
                )
                
                if response.status_code == 200:
                    new_products = response.json()
                    logger.info(f"Found {len(new_products)} products")
                    if new_products:
                        # Track current product IDs
                        current_ids = set()
                        updated = False
                        
                        for product in new_products:
                            current_ids.add(product['id'])
                            product_data = {
                                'id': product['id'],
                                'name': product['name'],
                                'price': product['price'],
                                'image_path': product['images'][0]['src'] if product['images'] else None,
                                'product_url': product['permalink'],
                                'categories': [cat['name'] for cat in product.get('categories', [])],
                                'attributes': product.get('attributes', [])
                            }
                            
                            # Check if we need to update this product
                            existing_product = next((p for p in self.products if p['id'] == product['id']), None)
                            if not existing_product:
                                self.products.append(product_data)
                                updated = True
                                logger.info(f"Added new product: {product_data['name']}")
                            
                            # Update features if needed
                            if product['id'] not in self.features and product_data['image_path']:
                                try:
                                    # Download and process image
                                    img_response = requests.get(product_data['image_path'])
                                    if img_response.status_code == 200:
                                        img = Image.open(io.BytesIO(img_response.content))
                                        img = img.convert('RGB')
                                        img = img.resize((224, 224))
                                        img_array = image.img_to_array(img)
                                        img_array = np.expand_dims(img_array, axis=0)
                                        img_array = preprocess_input(img_array)
                                        
                                        # Extract features
                                        features = feature_extractor.predict(img_array)
                                        self.features[product['id']] = features.flatten()
                                        updated = True
                                        logger.info(f"Extracted features for product: {product_data['name']}")
                                except Exception as e:
                                    logger.error(f"Error processing image for product {product_data['name']}: {str(e)}")
                        
                        # Remove products that no longer exist
                        self.products = [p for p in self.products if p['id'] in current_ids]
                        
                        if updated:
                            self.update_category_weights()
                            logger.info("Updated category weights")
                else:
                    logger.error(f"Failed to fetch products. Status code: {response.status_code}")
                    
            except Exception as e:
                logger.error(f"Error in check_new_products: {str(e)}")
                
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
    with open('api_key.txt', 'r') as f:
        api_key = f.read().strip()
    return render_template('standalone.html', api_key=api_key)

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
            f"{WOOCOMMERCE_URL}/products/{product_id}",
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
