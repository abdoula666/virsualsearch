from flask import Flask, render_template, request, jsonify
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
import traceback
import json
from dotenv import load_dotenv
from search import search_bp
from flask_cors import CORS

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["https://your-github-username.github.io"]}})
app.register_blueprint(search_bp)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# WooCommerce API Configuration
WOOCOMMERCE_URL = os.getenv('WOOCOMMERCE_URL')
CONSUMER_KEY = os.getenv('WOOCOMMERCE_CONSUMER_KEY')
CONSUMER_SECRET = os.getenv('WOOCOMMERCE_CONSUMER_SECRET')

# Initialize ResNet model
try:
    base_model = ResNet50(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    feature_extractor = tf.keras.Model(inputs=base_model.input, outputs=x)
    logger.info("Successfully initialized ResNet50 model")
except Exception as e:
    logger.error(f"Error initializing ResNet50 model: {str(e)}")
    raise

class ProductManager:
    def __init__(self):
        self.last_check = None
        self.products = []
        self.features = {}
        self.model = feature_extractor
        self.scheduler = BackgroundScheduler()
        self.scheduler.add_job(self.check_new_products, 'interval', minutes=5)
        self.scheduler.start()
        self.category_weights = {}
        self.processing_lock = threading.Lock()
        self.is_initialized = False
        self.initialization_error = None
        logger.info("ProductManager initialized")
        
        # Immediately check for products
        self.check_new_products()

    def test_woocommerce_connection(self):
        """Test the WooCommerce API connection"""
        try:
            response = requests.get(
                f"{WOOCOMMERCE_URL}/products",
                auth=(CONSUMER_KEY, CONSUMER_SECRET),
                params={'per_page': 1},
                timeout=10
            )
            response.raise_for_status()
            logger.info("WooCommerce connection test successful")
            return True
        except Exception as e:
            logger.error(f"WooCommerce connection test failed: {str(e)}")
            self.initialization_error = f"WooCommerce connection error: {str(e)}"
            return False

    def process_new_product(self, product_data):
        """Process a single new product"""
        try:
            with self.processing_lock:
                product_id = product_data['id']
                logger.info(f"Processing product ID: {product_id}")
                
                if product_id not in [p['id'] for p in self.products]:
                    image_url = product_data['images'][0]['src'] if product_data.get('images') else None
                    if not image_url:
                        logger.warning(f"No image found for product: {product_data.get('name')} (ID: {product_id})")
                        return False

                    try:
                        logger.info(f"Downloading image from: {image_url}")
                        response = requests.get(image_url, timeout=10)
                        response.raise_for_status()
                        
                        img = Image.open(io.BytesIO(response.content))
                        img = img.convert('RGB')
                        img = img.resize((224, 224))
                        img_array = image.img_to_array(img)
                        img_array = np.expand_dims(img_array, axis=0)
                        img_array = preprocess_input(img_array)
                        
                        logger.info("Extracting features...")
                        features = self.model.predict(img_array, verbose=0)
                        self.features[product_id] = features[0]
                        
                        # Format price
                        try:
                            price_value = float(product_data.get('price', 0))
                            price_display = str(int(price_value))
                        except (ValueError, TypeError):
                            price_display = 'Price not available'
                        
                        product_info = {
                            'id': product_id,
                            'name': product_data.get('name', 'Unknown'),
                            'price': price_display,
                            'image_path': image_url,
                            'product_url': product_data.get('permalink', ''),
                            'categories': [cat['name'] for cat in product_data.get('categories', [])],
                            'attributes': product_data.get('attributes', [])
                        }
                        
                        self.products.append(product_info)
                        logger.info(f"Successfully processed product: {product_info['name']} (ID: {product_id})")
                        return True
                    except requests.RequestException as e:
                        logger.error(f"Error downloading image for product {product_id}: {str(e)}")
                    except Exception as e:
                        logger.error(f"Error processing product {product_id}: {str(e)}")
                        logger.debug(traceback.format_exc())
                return False
        except Exception as e:
            logger.error(f"Error in process_new_product: {str(e)}")
            logger.debug(traceback.format_exc())
            return False

    def check_new_products(self):
        """Check for new products from WooCommerce"""
        logger.info("Starting product check...")
        
        if not self.test_woocommerce_connection():
            logger.error("Failed to connect to WooCommerce")
            return
        
        try:
            params = {
                'per_page': 100,
                'status': 'publish'
            }
            
            response = requests.get(
                f"{WOOCOMMERCE_URL}/products",
                params=params,
                auth=(CONSUMER_KEY, CONSUMER_SECRET),
                timeout=30
            )
            response.raise_for_status()
            
            products = response.json()
            logger.info(f"Found {len(products)} products")
            
            processed_count = 0
            for product in products:
                if self.process_new_product(product):
                    processed_count += 1
            
            if processed_count > 0:
                self.update_category_weights()
                logger.info(f"Processed {processed_count} new products")
            
            self.last_check = datetime.now()
            if len(self.products) > 0:
                self.is_initialized = True
                logger.info("Product manager initialization completed")
            else:
                logger.warning("No products were processed")
            
            logger.info(f"Product check completed. Total products: {len(self.products)}")
            
        except requests.RequestException as e:
            logger.error(f"Error fetching products from WooCommerce: {str(e)}")
            self.initialization_error = str(e)
        except Exception as e:
            logger.error(f"Error in check_new_products: {str(e)}")
            logger.debug(traceback.format_exc())
            self.initialization_error = str(e)

    def update_category_weights(self):
        """Update category importance weights"""
        try:
            category_counts = {}
            total_products = len(self.products)
            
            for product in self.products:
                for category in product['categories']:
                    category_counts[category] = category_counts.get(category, 0) + 1
            
            for category, count in category_counts.items():
                self.category_weights[category] = np.log(total_products / (count + 1))
            
            logger.info("Category weights updated successfully")
        except Exception as e:
            logger.error(f"Error updating category weights: {str(e)}")

def calculate_similarity_score(query_features, product_features, product_categories, query_category=None):
    """Calculate similarity score between query and product"""
    try:
        visual_similarity = np.dot(query_features, product_features) / (
            np.linalg.norm(query_features) * np.linalg.norm(product_features)
        )
        
        category_boost = 1.0
        if query_category and query_category in product_categories:
            category_boost = 1.5
        
        return float(visual_similarity * category_boost)
    except Exception as e:
        logger.error(f"Error calculating similarity score: {str(e)}")
        return 0.0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/status')
def get_status():
    """Get the current status of the product manager"""
    status = {
        'initialized': product_manager.is_initialized,
        'total_products': len(product_manager.products),
        'products_with_features': len(product_manager.features),
        'last_check': product_manager.last_check.isoformat() if product_manager.last_check else None,
        'initialization_error': product_manager.initialization_error,
        'ready': len(product_manager.products) > 0 and len(product_manager.features) > 0
    }
    
    # Test WooCommerce connection
    woo_status = "Connected" if product_manager.test_woocommerce_connection() else "Not Connected"
    status['woocommerce_status'] = woo_status
    
    return jsonify(status)

@app.route('/search', methods=['POST'])
def search():
    """Handle image search requests"""
    if not product_manager.is_initialized or len(product_manager.products) == 0:
        status = {
            'error': 'Products still loading, please wait',
            'total_products': len(product_manager.products),
            'products_with_features': len(product_manager.features),
            'initialization_error': product_manager.initialization_error,
            'woocommerce_status': "Connected" if product_manager.test_woocommerce_connection() else "Not Connected"
        }
        return jsonify(status), 503

    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Process the image
        img = Image.open(file.stream)
        img = img.convert('RGB')
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # Extract features
        query_features = feature_extractor.predict(img_array, verbose=0)
        query_features = normalize(query_features).flatten()
        
        # Calculate similarities
        similarities = []
        for product in product_manager.products:
            if product['id'] in product_manager.features:
                score = calculate_similarity_score(
                    query_features,
                    product_manager.features[product['id']],
                    product['categories']
                )
                
                # Format price
                try:
                    price_value = float(product['price'])
                    price_display = str(int(price_value))
                except (ValueError, TypeError):
                    price_display = 'Price not available'
                
                similarities.append({
                    'id': product['id'],
                    'name': product['name'],
                    'price': price_display,
                    'image_url': product['image_path'],
                    'product_url': product['product_url'],
                    'score': score,
                    'categories': product['categories']
                })
        
        # Sort by similarity score
        similarities.sort(key=lambda x: x['score'], reverse=True)
        
        return jsonify({
            'results': similarities[:50],
            'total_products': len(product_manager.products),
            'products_with_features': len(product_manager.features)
        })

    except Exception as e:
        logger.error(f"Error processing search: {str(e)}")
        logger.debug(traceback.format_exc())
        return jsonify({'error': 'Error processing image'}), 500

@app.route('/search_products')
def search_products():
    try:
        query = request.args.get('query', '')
        if not query:
            return jsonify([])
        
        # Search WooCommerce products
        params = {
            'search': query,
            'per_page': 10,
            'status': 'publish'
        }
        
        response = requests.get(
            f"{WOOCOMMERCE_URL}/products",
            auth=(CONSUMER_KEY, CONSUMER_SECRET),
            params=params
        )
        response.raise_for_status()
        
        products = response.json()
        return jsonify(products)
        
    except Exception as e:
        logger.error(f"Error searching products: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
