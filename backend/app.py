from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import shutil
from pathlib import Path
import requests
from gradio_client import Client, file
from datetime import datetime
from time import sleep
from typing import Optional

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Define exact paths based on your directory structure
BASE_DIR = Path(r"C:\Users\deeks\fashion-recommendation-sys")
BACKEND_DIR = BASE_DIR / "backend"
UPLOADS_DIR = BACKEND_DIR / "uploads"
PUBLIC_DIR = BASE_DIR / "frontend" / "public"

# Print paths for verification
print("\nDirectory Configuration:")
print(f"Base Directory: {BASE_DIR}")
print(f"Backend Directory: {BACKEND_DIR}")
print(f"Uploads Directory: {UPLOADS_DIR}")
print(f"Public Directory: {PUBLIC_DIR}")

# Create necessary directories
UPLOADS_DIR.mkdir(exist_ok=True, parents=True)
PUBLIC_DIR.mkdir(exist_ok=True, parents=True)

def initialize_gradio_client(url: str, max_retries: int = 3) -> Optional[Client]:
    """Initialize a Gradio client with retries"""
    for attempt in range(max_retries):
        try:
            client = Client(url)
            print(f"Successfully connected to {url}")
            return client
        except Exception as e:
            print(f"Attempt {attempt + 1}/{max_retries} failed for {url}: {e}")
            if attempt < max_retries - 1:
                sleep(2)  # Wait 2 seconds before retrying
    return None

# Define client URLs with fallbacks
CLIENTS = {
    "virtual_tryon": {
        "urls": [
            "https://7395458a587bc50ec3.gradio.live/",
            "https://virtual-try-on.hf.space/"
        ]
    },
    "chatbot": {
        "urls": [
            "https://fe81ff40040ecfff3c.gradio.live/",
            "https://fashion-chatbot.hf.space/"
        ]
    },
    "text_to_dress": {
        "urls": [
            "dhaan-ish/text-to-cloth"
        ]
    },
    "occasion": {
        "urls": [
            "https://8c8e6f96c1fe2aefb7.gradio.live/",
            "https://fashion-occasion.hf.space/"
        ]
    }
}

# Initialize clients with fallbacks
clients = {}
for service_name, config in CLIENTS.items():
    for url in config["urls"]:
        client = initialize_gradio_client(url)
        if client:
            clients[service_name] = client
            print(f"Successfully initialized {service_name} client")
            break
    if service_name not in clients:
        print(f"Failed to initialize {service_name} client")

def generate_unique_filename(original_name):
    """Generate a unique filename using timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{original_name}"

def safe_copy(src, dst):
    """Safely copy a file with error handling"""
    try:
        shutil.copy(src, dst)
        print(f"Successfully copied {src} to {dst}")
        return True
    except Exception as e:
        print(f"Error copying file: {e}")
        return False

def download_image(image_url):
    """Download image from URL and save it locally"""
    try:
        image_url = image_url.strip()
        filename = "downloaded_image.png"
        file_path = UPLOADS_DIR / filename
        
        response = requests.get(image_url, stream=True)
        if response.status_code == 200:
            with open(file_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=128):
                    file.write(chunk)
            print(f"Image downloaded successfully and saved as {file_path}")
            return str(file_path)
        else:
            print(f"Failed to download image. Status code: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error downloading image: {e}")
        return None

@app.errorhandler(503)
def service_unavailable(error):
    return jsonify({
        'error': 'Service temporarily unavailable',
        'message': str(error)
    }), 503

@app.errorhandler(500)
def internal_server_error(error):
    return jsonify({
        'error': 'Internal server error',
        'message': str(error)
    }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'directories': {
            'public': str(PUBLIC_DIR),
            'uploads': str(UPLOADS_DIR),
        },
        'directories_exist': {
            'public': PUBLIC_DIR.exists(),
            'uploads': UPLOADS_DIR.exists(),
        },
        'services': {
            name: 'available' if name in clients else 'unavailable'
            for name in CLIENTS.keys()
        }
    })

@app.route("/predict", methods=["POST"])
def predict():
    """Endpoint for chatbot predictions"""
    try:
        if "chatbot" not in clients:
            return jsonify({'error': 'Chatbot service not available'}), 503

        data = request.get_json()
        text_input = data["text"]
        print(f"Received input: {text_input}")
        
        result = clients["chatbot"].predict(
            text_input,
            api_name="/predict"
        )
        print(f"Prediction result: {result}")
        return jsonify({"result": result})
    except Exception as e:
        print(f"Error in predict: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/uploadocassion', methods=['POST'])
def upload_ocassion():
    """Endpoint for occasion-based virtual try-on"""
    try:
        if "virtual_tryon" not in clients:
            return jsonify({'error': 'Virtual try-on service not available'}), 503

        if 'uploadedFile' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        uploaded_file = request.files['uploadedFile']
        url = request.form['url']
        
        downloaded_path = download_image(url)
        if not downloaded_path:
            return jsonify({'error': 'Failed to download image'}), 500
        
        if uploaded_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        upload_path = UPLOADS_DIR / 'upload.png'
        uploaded_file.save(upload_path)
        
        result = clients["virtual_tryon"].predict(
            file(downloaded_path),
            file(str(upload_path)),
            api_name="/predict"
        )
        
        result_filename = generate_unique_filename("result.png")
        destination = PUBLIC_DIR / result_filename
        
        if safe_copy(result, str(destination)):
            return jsonify({
                'message': 'Result image copied successfully.',
                'filename': result_filename
            }), 200
        else:
            return jsonify({'error': 'Failed to save result image'}), 500
            
    except Exception as e:
        print(f"Error in upload_ocassion: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_files():
    """Endpoint for regular virtual try-on"""
    try:
        if "virtual_tryon" not in clients:
            return jsonify({'error': 'Virtual try-on service not available'}), 503

        if 'uploadedFile' not in request.files:
            return jsonify({'error': 'No file part'}), 400
            
        uploaded_file = request.files['uploadedFile']
        if uploaded_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        upload_path = UPLOADS_DIR / 'upload.png'
        uploaded_file.save(upload_path)
        
        result = clients["virtual_tryon"].predict(
            file(str(PUBLIC_DIR / "image.JPEG")),
            file(str(upload_path)),
            api_name="/predict"
        )
        
        result_filename = generate_unique_filename("result.png")
        destination = PUBLIC_DIR / result_filename
        
        if safe_copy(result, str(destination)):
            return jsonify({
                'message': 'Result image copied successfully.',
                'filename': result_filename
            }), 200
        else:
            return jsonify({'error': 'Failed to save result image'}), 500
            
    except Exception as e:
        print(f"Error in upload_files: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/handleprompt', methods=['POST'])
def handle_prompt():
    """Endpoint for text-to-dress generation"""
    try:
        if "text_to_dress" not in clients:
            return jsonify({'error': 'Text-to-dress service not available'}), 503

        data = request.get_json()
        prompt = data.get('prompt')
        if not prompt:
            return jsonify({'error': 'No prompt provided'}), 400
            
        print(f"Received prompt: {prompt}")
        
        result = clients["text_to_dress"].predict(
            prompt,
            api_name="/predict"
        )
        
        result_filename = generate_unique_filename("generated_dress.png")
        destination = PUBLIC_DIR / result_filename
        
        if safe_copy(result, str(destination)):
            return jsonify({
                'message': 'Success',
                'filename': result_filename
            }), 200
        else:
            return jsonify({'error': 'Failed to save generated image'}), 500
            
    except Exception as e:
        print(f"Error in handle_prompt: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/handleocassion', methods=['POST'])
def handleocassion():
    """Endpoint for occasion-based recommendations"""
    try:
        if "occasion" not in clients:
            return jsonify({'error': 'Occasion service not available'}), 503

        data = request.json
        color = data.get('color')
        selected_occasion = data.get('selectedOccasion')
        
        if not color or not selected_occasion:
            return jsonify({'error': 'Color and occasion are required'}), 400
            
        print(f"Color: {color}, Occasion: {selected_occasion}")
        
        result = clients["occasion"].predict(
            f"{color} shirt for {selected_occasion}",
            api_name="/predict"
        )
        
        new_items = result.split(",")
        
        return jsonify({
            'newItems': new_items,
            'showRecommendations': True
        })
    except Exception as e:
        print(f"Error in handleocassion: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Verify directories exist before starting server
    print("\nVerifying directories before start:")
    print(f"Uploads directory exists: {UPLOADS_DIR.exists()}")
    print(f"Public directory exists: {PUBLIC_DIR.exists()}")
    
    app.run(debug=True)
