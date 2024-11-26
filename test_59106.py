from flask import Flask
from waitress import serve

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello, World! Server is running on port 59106'

if __name__ == '__main__':
    print("Starting server on http://localhost:59106")
    serve(app, host='localhost', port=59106, threads=1)
