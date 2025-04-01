from flask import Flask, request, jsonify
import pickle
import numpy as np
from flask_cors import CORS  # To allow requests from PHP

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Requests

# Load trained ML model
model = pickle.load(open('ML1', 'rb'))

# Prediction API
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json  # Get JSON data from PHP request

        # Extract values
        price = float(data['price'])
        open_price = float(data['open'])
        high = float(data['high'])
        low = float(data['low'])

        # Prepare input for prediction
        input_data = np.array([[price, open_price, high, low]])

        # Predict future price
        prediction = model.predict(input_data)[0]

        return jsonify({'prediction': prediction})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
