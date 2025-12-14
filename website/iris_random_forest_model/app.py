
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

# Create Flask app
app = Flask(__name__)
CORS(app)  # Allow requests from web browsers

# Load the trained model
model = joblib.load('../output/iris_random_forest_model.pkl')

# Flower type names
FLOWER_NAMES = ['Setosa', 'Versicolor', 'Virginica']

@app.route('/')
def home():
    """Home page with API information"""
    return """
    <h1>🌸 Iris Flower Prediction API</h1>
    <p>Send POST request to /predict with flower measurements</p>
    <h3>Example:</h3>
    <pre>
    POST /predict
    {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    </pre>
    """

@app.route('/predict', methods=['POST'])
def predict():
    """Predict flower type from measurements"""
    try:
        # Get data from request
        data = request.get_json()

        # Extract features
        features = [
            float(data['sepal_length']),
            float(data['sepal_width']),
            float(data['petal_length']),
            float(data['petal_width'])
        ]

        # Make prediction
        prediction = model.predict([features])[0]
        probabilities = model.predict_proba([features])[0]

        # Prepare response
        response = {
            'prediction': FLOWER_NAMES[prediction],
            'prediction_id': int(prediction),
            'confidence': {
                'Setosa': float(probabilities[0]),
                'Versicolor': float(probabilities[1]),
                'Virginica': float(probabilities[2])
            },
            'input': {
                'sepal_length': features[0],
                'sepal_width': features[1],
                'petal_length': features[2],
                'petal_width': features[3]
            }
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    print("Starting Iris Prediction API...")
    print("API running at: http://localhost:5000")
    print("View docs at: http://localhost:5000")
    app.run(debug=True, port=5000)
