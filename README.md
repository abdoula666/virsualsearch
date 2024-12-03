# Visual Search Application

This is a visual search application that allows users to search for products by uploading images. The application uses deep learning to find visually similar products in your WooCommerce store.

## Environment Setup

1. Clone the repository
2. Create a virtual environment and activate it:
```bash
python -m venv .venv
.venv\Scripts\activate  # On Windows
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

5. Run the application:
```bash
python app.py
```

## Security Notes
- Never commit the `.env` file to version control
- Keep your API credentials secure
- Use environment variables for all sensitive information

## Setup Instructions

1. Prepare your product images:
   - Create a folder named `product_images`
   - Add your product images to this folder
   - Name your images in the format: `productID_description.jpg`
   - Example: `123_blue_shirt.jpg` where 123 is the WooCommerce product ID

2. Generate the feature vectors:
```bash
python feature_extraction.py
```
This will create three files:
- `featurevector.pkl`: Contains the extracted image features
- `filenames.pkl`: Contains the paths to your images
- `product_ids.pkl`: Contains the WooCommerce product IDs

3. Open your browser and go to `http://localhost:5000`

## Using the Application

1. Click the "Take Photo" button to use your device's camera
2. Or click "Choose Image" to upload an image from your device
3. The application will show visually similar products from your store
4. Click "View Details" to see the full product information on your WooCommerce store

## Features

- Mobile-friendly interface
- Camera integration for direct photo capture
- Image upload support
- Real-time visual search
- Integration with WooCommerce API
- Display of product names and prices
- Direct links to product pages
