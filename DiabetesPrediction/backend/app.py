from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import joblib
from flask_cors import CORS

# Load all models and scaler
model_dl = tf.keras.models.load_model("diabetes_model.h5")
model_knn = joblib.load("knn_model.pkl")
model_svm_linear = joblib.load("svm_linear_model.pkl")
model_svm_rbf = joblib.load("svm_rbf_model.pkl")
model_logreg = joblib.load("logreg_model.pkl")
scaler = joblib.load("scaler.pkl")

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "API is running", "endpoint": "/predict (POST only)"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        features = np.array(data["features"]).reshape(1, -1)
        scaled_features = scaler.transform(features)
        model_choice = data.get("model", "deep_learning")  # Default to deep learning

        if model_choice == "deep_learning":
            prediction = model_dl.predict(scaled_features)[0][0]
            result = "Diabetic" if prediction > 0.5 else "Not Diabetic"
            probability = float(prediction)
        elif model_choice == "knn":
            prediction = model_knn.predict(scaled_features)[0]
            result = "Diabetic" if prediction == 1 else "Not Diabetic"
            probability = float(model_knn.predict_proba(scaled_features)[0][1])
        elif model_choice == "svm_linear":
            prediction = model_svm_linear.predict(scaled_features)[0]
            result = "Diabetic" if prediction == 1 else "Not Diabetic"
            probability = float(model_svm_linear.predict_proba(scaled_features)[0][1])
        elif model_choice == "svm_rbf":
            prediction = model_svm_rbf.predict(scaled_features)[0]
            result = "Diabetic" if prediction == 1 else "Not Diabetic"
            probability = float(model_svm_rbf.predict_proba(scaled_features)[0][1])
        elif model_choice == "logistic_regression":
            prediction = model_logreg.predict(scaled_features)[0]
            result = "Diabetic" if prediction == 1 else "Not Diabetic"
            probability = float(model_logreg.predict_proba(scaled_features)[0][1])
        else:
            return jsonify({"error": "Invalid model choice"}), 400

        return jsonify({"prediction": result, "probability": probability, "model": model_choice})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)