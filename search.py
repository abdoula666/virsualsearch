from flask import Blueprint, jsonify, request
import requests
import logging
from deep_translator import GoogleTranslator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# WooCommerce API Configuration
WOOCOMMERCE_URL = "https://cgbshop1.com/wp-json/wc/v3"
CONSUMER_KEY = "ck_da1507a982310e8a29d704df57b4e886b26d528a"
CONSUMER_SECRET = "cs_2917aeffff79c6bb2427849b617f0c992959f301"

# Create Blueprint
search_bp = Blueprint('search', __name__)

# Common fashion terms translation dictionary
FASHION_TERMS = {
    'shoes': 'chaussures',
    'dress': 'robe',
    'shirt': 'chemise',
    'pants': 'pantalon',
    'jacket': 'veste',
    'hat': 'chapeau',
    'skirt': 'jupe',
    'boots': 'bottes',
    'sandals': 'sandales',
    'coat': 'manteau',
    'sweater': 'pull',
    'jeans': 'jean',
    'socks': 'chaussettes',
    'scarf': 'écharpe',
    'gloves': 'gants',
    'belt': 'ceinture',
    'bag': 'sac',
    'wallet': 'portefeuille',
    'jewelry': 'bijoux',
    'watch': 'montre',
    'sneakers': 'baskets',
    'suit': 'costume',
    'tie': 'cravate',
    'blouse': 'chemisier',
    'shorts': 'short',
    'hoodie': 'sweat à capuche',
    'underwear': 'sous-vêtements',
    'pajamas': 'pyjama',
    'swimsuit': 'maillot de bain',
    'cardigan': 'cardigan'
}

def translate_to_french(query):
    """Translate query to French using dictionary or Google Translate"""
    try:
        # Check if the query is in our fashion terms dictionary
        query_lower = query.lower().strip()
        if query_lower in FASHION_TERMS:
            translated = FASHION_TERMS[query_lower]
            logger.info(f"Dictionary translation: {query} -> {translated}")
            return translated, True

        # If not in dictionary, use Google Translate
        translator = GoogleTranslator(source='auto', target='fr')
        translated = translator.translate(query)
        logger.info(f"Google translation: {query} -> {translated}")
        return translated, True

    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        return query, False

@search_bp.route('/text_search')
def text_search():
    try:
        query = request.args.get('query', '').strip()
        if not query:
            return jsonify({'results': [], 'translation': None})

        # Translate query to French
        french_query, was_translated = translate_to_french(query)

        # Search with French query
        params = {
            'search': french_query,
            'per_page': 20,
            'status': 'publish'
        }
        
        response = requests.get(
            f"{WOOCOMMERCE_URL}/products",
            auth=(CONSUMER_KEY, CONSUMER_SECRET),
            params=params
        )
        response.raise_for_status()
        products = response.json()

        # If no results with French query and it was translated, try original query
        if not products and was_translated:
            params['search'] = query
            response = requests.get(
                f"{WOOCOMMERCE_URL}/products",
                auth=(CONSUMER_KEY, CONSUMER_SECRET),
                params=params
            )
            response.raise_for_status()
            products = response.json()

        # Format the response
        formatted_products = []
        for product in products:
            formatted_product = {
                'id': product['id'],
                'name': product['name'],
                'price': product['price'],
                'permalink': product['permalink'],
                'images': product['images'],
                'categories': [cat['name'] for cat in product['categories']]
            }
            formatted_products.append(formatted_product)

        # Return results with translation information
        return jsonify({
            'results': formatted_products,
            'translation': french_query if was_translated and french_query != query else None
        })

    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return jsonify({'error': str(e)}), 500
