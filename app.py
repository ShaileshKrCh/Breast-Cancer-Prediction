import os
import sys
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

MODEL_PATH = 'cancer_model.pkl'
REQUIRED_FEATURES = ["mean_radius", "mean_texture", "mean_perimeter", "mean_area", "mean_smoothness"]

FEATURE_MAPPING = {
    "mean_radius": "mean radius",
    "mean_texture": "mean texture",
    "mean_perimeter": "mean perimeter",
    "mean_area": "mean area",
    "mean_smoothness": "mean smoothness"
}

# Approximate normal scaling maximums from the Wisconsin dataset to normalize radar inputs (0 to 100)
DATASET_MAXIMUMS = {
    "mean_radius": 28.11,
    "mean_texture": 39.28,
    "mean_perimeter": 188.5,
    "mean_area": 2501.0,
    "mean_smoothness": 0.163
}

# Baseline averages for Malignant vs Benign populations to plot as reference layers
BENIGN_BASLINES = [12.14, 17.91, 78.07, 462.79, 0.092]
MALIGNANT_BASELINES = [17.46, 21.60, 115.36, 978.37, 0.102]

if os.path.exists(MODEL_PATH):
    try:
        model_pipeline = joblib.load(MODEL_PATH)
        print(f"Successfully mounted model pipeline from {MODEL_PATH}")
    except Exception as e:
        print(f"Fatal error reading model file: {str(e)}")
        sys.exit(1)
else:
    print(f"CRITICAL Error: {MODEL_PATH} not found. Execute 'train.py' first.")
    sys.exit(1)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()

        if not data:
            return jsonify({"error": "Empty payload received."}), 400

        processed_features = {}
        raw_values = []
        normalized_values = []

        for feature in REQUIRED_FEATURES:
            if feature not in data:
                return jsonify({"error": f"Missing metric: {feature}"}), 400
            
            try:
                val = float(data[feature])
                if val <= 0:
                    return jsonify({"error": f"Value for {feature} must be greater than zero."}), 400
                
                processed_features[FEATURE_MAPPING[feature]] = [val]
                raw_values.append(val)
                
                # Normalize values for the Radar Chart scaling
                norm = (val / DATASET_MAXIMUMS[feature]) * 100
                normalized_values.append(min(float(norm), 100.0))
                
            except ValueError:
                return jsonify({"error": f"Field '{feature}' must contain valid numbers."}), 400

        input_dataframe = pd.DataFrame(processed_features)

        prediction_class = int(model_pipeline.predict(input_dataframe)[0])
        probabilities = model_pipeline.predict_proba(input_dataframe)[0]
        confidence = float(probabilities[prediction_class] * 100)
        diagnosis_output = "Benign" if prediction_class == 1 else "Malignant"

        # Calculate normalized comparative arrays for UI graphing
        benign_norm = [(b / DATASET_MAXIMUMS[f]) * 100 for b, f in zip(BENIGN_BASLINES, REQUIRED_FEATURES)]
        malignant_norm = [(m / DATASET_MAXIMUMS[f]) * 100 for m, f in zip(MALIGNANT_BASELINES, REQUIRED_FEATURES)]

        return jsonify({
            "success": True,
            "prediction": diagnosis_output,
            "confidence": f"{confidence:.2f}%",
            "raw_class": prediction_class,
            "probability_malignant": float(probabilities[0] * 100),
            "chart_data": {
                "labels": ["Radius", "Texture", "Perimeter", "Area", "Smoothness"],
                "patient": normalized_values,
                "benign_baseline": benign_norm,
                "malignant_baseline": malignant_norm
            }
        }), 200

    except Exception as e:
        return jsonify({"error": "An unexpected error occurred.", "details": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)