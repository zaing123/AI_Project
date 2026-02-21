# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import joblib
import pandas as pd
import io
import os

app = Flask(__name__)
CORS(app)

# -------------------------------
# Load Model Safely (Cloud Ready)
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "sentiment_model.pkl")

model_data = joblib.load(model_path)
vectorizer = model_data["vectorizer"]
svm_model = model_data["model"]


# -------------------------------
# Utility Function
# -------------------------------
def predict_sentiment(reviews):
    reviews_tfidf = vectorizer.transform(reviews)
    sentiments = svm_model.predict(reviews_tfidf)

    results = []
    for r, s in zip(reviews, sentiments):

        if s == "Positive":
            emoji = "😊"
            stars = "⭐⭐⭐⭐⭐"
        elif s == "Neutral":
            emoji = "😐"
            stars = "⭐⭐⭐"
        else:
            emoji = "😞"
            stars = "⭐"

        results.append({
            "review": r,
            "sentiment": s,
            "emoji": emoji,
            "stars": stars
        })

    return results


# -------------------------------
# Health Check Route
# -------------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Sentiment API is running 🚀"})


# -------------------------------
# JSON Prediction Route
# -------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data or "reviews" not in data:
            return jsonify({"error": "No reviews provided"}), 400

        reviews = data["reviews"]

        if isinstance(reviews, str):
            reviews = [reviews]

        results = predict_sentiment(reviews)

        summary = pd.DataFrame(results)["sentiment"].value_counts().to_dict()

        return jsonify({
            "results": results,
            "summary": summary
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------------------
# File Upload Route
# -------------------------------
@app.route("/upload", methods=["POST"])
def upload_file():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        filename = file.filename.lower()

        if filename.endswith(".csv"):
            df = pd.read_csv(file)
            if "review" not in df.columns:
                return jsonify({"error": "CSV must contain 'review' column"}), 400
            reviews = df["review"].astype(str).tolist()

        elif filename.endswith(".txt"):
            reviews = [line.decode("utf-8").strip() for line in file if line.strip()]

        else:
            return jsonify({"error": "Unsupported file type"}), 400

        # Perform AI Prediction
        results = predict_sentiment(reviews)

        # Create the summary count for the Android PieChart
        summary = pd.DataFrame(results)["sentiment"].value_counts().to_dict()

        # IMPORTANT: Return JSON so the Android App can display the list and chart
        return jsonify({
            "results": results,
            "summary": summary
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------------------
# Run Locally
# -------------------------------
if __name__ == "__main__":
    app.run()
