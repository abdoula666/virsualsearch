from setuptools import setup, find_packages

setup(
    name="visual-search",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "flask>=2.0.1",
        "flask-cors>=4.0.0",
        "numpy>=1.22.0",
        "tensorflow-cpu>=2.13.0",
        "Pillow>=8.3.1",
        "requests>=2.31.0",
        "APScheduler>=3.10.1",
        "scikit-learn>=0.24.2",
        "opencv-python-headless>=4.5.3.56",
        "python-dotenv>=1.0.0",
        "WooCommerce>=3.0.0",
        "gunicorn>=20.1.0",
    ],
    python_requires=">=3.8",
)
