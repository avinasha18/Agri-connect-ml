import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import json
import re

# Initialize Flask app and CORS
app = Flask(__name__)
CORS(app)

# Load the saved model from the joblib file
crop_model = None  # Initialize the crop_model variable

try:
    crop_model = joblib.load('crop_recommendation_five.joblib')
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")

# Configure Google Generative AI
genai_api_key = 'YOUR_API_KEY_HERE'  # Replace with your actual API key
genai.configure(api_key=genai_api_key)

# Initialize the generative model
model = genai.GenerativeModel('gemini-1.5-flash')

def recommend_top_crops(model, X_test, top_n=3):
    """Recommends top N crops based on their predicted probabilities."""
    probabilities = model.predict_proba(X_test)
    crop_labels = model.classes_

    top_crops = []
    for i in range(len(probabilities)):
        top_indices = probabilities[i].argsort()[-top_n:][::-1]
        top_crops.append(crop_labels[top_indices])

    return top_crops

def get_crop_info(crop_name):
    """Fetches detailed information about a crop using Google Generative AI."""
    prompt = f"""Provide detailed information about cultivating {crop_name} in the following JSON format:
    {{
        "cropName": "{crop_name}",
        "stages": [
            {{
                "name": "Preparation",
                "description": "Steps to prepare for planting, including soil preparation and timing."
            }},
            {{
                "name": "Sowing",
                "description": "How to sow the seeds, including depth, spacing, and any special techniques."
            }},
            {{
                "name": "Growth",
                "description": "Care during the growth phase, including watering, fertilizing, and general maintenance."
            }},
            {{
                "name": "Harvesting",
                "description": "When and how to harvest the crop, including signs of readiness and harvesting techniques."
            }}
        ],
        "optimalConditions": {{
            "soil": "Describe ideal soil conditions",
            "temperature": "Ideal temperature range",
            "water": "Water requirements",
            "sunlight": "Sunlight needs"
        }},
        "commonIssues": [
            {{
                "problem": "Name of a common disease or pest",
                "symptoms": "How to identify the problem",
                "solution": "Steps to address the issue"
            }}
        ],
        "bestPractices": [
            "List of best practices for maximizing yield and crop health"
        ]
    }}
    Ensure all information is accurate and specific to {crop_name}. Provide at least three common issues and five best practices."""

    response = model.generate_content(prompt)
    try:
        cleaned_response = re.sub(r'```json\n|\n```', '', response.text).strip()
        crop_info = json.loads(cleaned_response)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        print(f"Response text: {response.text}")
        crop_info = {}

    return crop_info

@app.route('/predict', methods=['POST'])
def predict():
    """Handles prediction requests."""
    if crop_model is None:
        return jsonify({"error": "Model not loaded, check server logs"}), 500
    
    data = request.json
    df = pd.DataFrame([data])  # Convert incoming JSON to DataFrame
    
    # Ensure the input features match the training data structure
    try:
        top_crops = recommend_top_crops(crop_model, df, top_n=3)
        print(top_crops[0])
        crop_infos = [get_crop_info(crop) for crop in top_crops[0]]

        return jsonify({
            'recommendedCrops': top_crops[0].tolist(),
            'cropInfos': crop_infos
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
