import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
MODEL_PATH = "fraud_detection_model.pkl"  # Ensure this file exists
try:
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found. Ensure it exists in the correct directory.")

# Home route - renders the frontend page
@app.route("/")
def home():
    return render_template("index.html")

# API route for making predictions  
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get data from form submission
        time = float(request.form["time"])
        amount = float(request.form["amount"])
        transaction_hour = float(request.form["transaction_hour"])  

        # Convert to numpy array
        data = np.array([time, amount, transaction_hour]).reshape(1, -1)

        # Get fraud probability instead of direct class prediction
        probability = model.predict_proba(data)[0][1]  # Probability of fraud
        threshold = 0.1  # Adjusted threshold for better fraud detection
        result = 1 if probability > threshold else 0  

        return jsonify({"prediction": result, "fraud_probability": probability})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
