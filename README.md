# Visual Search Application

This is a visual search application that allows users to search for products by uploading images. The application uses deep learning to find visually similar products in your WooCommerce store.

## Setup Instructions

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare your product images:
   - Create a folder named `product_images`
   - Add your product images to this folder
   - Name your images in the format: `productID_description.jpg`
   - Example: `123_blue_shirt.jpg` where 123 is the WooCommerce product ID

3. Generate the feature vectors:
```bash
python feature_extraction.py
```
This will create three files:
- `featurevector.pkl`: Contains the extracted image features
- `filenames.pkl`: Contains the paths to your images
- `product_ids.pkl`: Contains the WooCommerce product IDs

4. Run the application:
```bash
python app.py
```

5. Open your browser and go to `http://localhost:5000`

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
