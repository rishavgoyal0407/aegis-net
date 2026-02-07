from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from services.inference import InferenceService
from config import Config

app = Flask(__name__)
CORS(app)
inference_service = InferenceService()

@app.route('/heatmaps/<path:filename>')
def serve_heatmap(filename):
    return send_from_directory(Config.HEATMAP_FOLDER, filename)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    # Save temp file or process in memory
    # ...
    
    result = inference_service.predict(file)
    return jsonify(result)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
