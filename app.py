from flask import Flask, render_template, request
import joblib
import numpy as np
from collections import Counter

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('fraud_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get comma-separated values from input
        input_str = request.form['data']
        input_data = [float(i) for i in input_str.split(',')]

        # Ensure exactly 30 features
        if len(input_data) != 30:
            raise ValueError("Please enter exactly 30 values.")

        # Custom fraud detection logic
        negative_count = sum(1 for x in input_data if x < 0)
        repeated_values = Counter(input_data)
        has_repeated = any(count > 5 for count in repeated_values.values())

        if negative_count > 10 or has_repeated:
            prediction = "Fraud (Rule-Based)"
        else:
            # Scale and use model to predict
            scaled_input = scaler.transform([input_data])
            result = model.predict(scaled_input)[0]
            prediction = 'Fraud' if result == 1 else 'Safe'

        return render_template('index.html', prediction=prediction)

    except Exception as e:
        return render_template('index.html', prediction='Error: ' + str(e))

if __name__ == '__main__':
    app.run(debug=True)
