# Visual Search Application

This is a visual search application that allows users to search for products by uploading images. The application uses deep learning to find visually similar products in your WooCommerce store.

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/abdoula666/virsualsearch.git
cd virsualsearch
```

2. Create a virtual environment and activate it:
```bash
python -m venv .venv
.venv\Scripts\activate  # On Windows
source .venv/bin/activate  # On Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Fill in your WooCommerce credentials in `.env`:
     ```
     WOOCOMMERCE_URL=your_woocommerce_url
     WOOCOMMERCE_CONSUMER_KEY=your_consumer_key
     WOOCOMMERCE_CONSUMER_SECRET=your_consumer_secret
     ```

5. Create required directories:
```bash
mkdir -p product_images uploads
```

6. Add your product images to the `product_images` folder:
   - Name format: `productID_description.jpg`
   - Make sure the productID matches your WooCommerce product IDs

7. Generate feature vectors:
```bash
python feature_extraction.py
```

8. Run the application:
```bash
python app.py
```

9. Open your browser and go to `http://localhost:5000`

## Project Structure
```
virsualsearch/
├── .env                    # Environment variables (create this)
├── .env.example           # Example environment file
├── app.py                 # Main application file
├── feature_extraction.py  # Feature extraction script
├── search.py             # Search functionality
├── requirements.txt      # Python dependencies
├── product_images/      # Store your product images here
├── uploads/            # Temporary storage for uploaded images
└── templates/         # HTML templates
```

## Security Notes
- Never commit the `.env` file to version control
- Keep your API credentials secure
- Use environment variables for all sensitive information

## Troubleshooting

1. If you get import errors:
   - Make sure you've activated the virtual environment
   - Verify all dependencies are installed: `pip install -r requirements.txt`

2. If feature extraction fails:
   - Ensure product_images directory exists and contains images
   - Check image format (JPG/JPEG recommended)
   - Verify image naming format (productID_description.jpg)

3. If WooCommerce connection fails:
   - Verify your credentials in .env file
   - Check if your WooCommerce site is accessible
   - Ensure API access is enabled in WooCommerce settings
