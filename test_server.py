from flask import Flask
from waitress import serve

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello, World!'

if __name__ == '__main__':
    print("Server starting on http://localhost:8080")
    serve(app, host='127.0.0.1', port=8080)
