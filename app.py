import joblib
from flask import Flask, request, jsonify, render_template
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load model and vectorizer
classmodel = joblib.load("model/best_spam_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

# Label mapping
label_map = {0: "Not Spam", 1: "Spam"}


@app.route("/")
def home():
    """Render home page."""
    return render_template("home.html")


@app.route("/predict_api", methods=["POST"])
def predict_api():
    """API endpoint for model prediction (JSON input)."""
    data = request.get_json()

    if not data or "data" not in data:
        return jsonify({"error": "No 'data' field found in JSON request"}), 400

    message = data["data"]

    try:
        new_data = vectorizer.transform([message])
        output = classmodel.predict(new_data)[0]
        prediction_label = label_map.get(output, str(output))

        # If model supports probability
        confidence = None
        if hasattr(classmodel, "predict_proba"):
            proba = classmodel.predict_proba(new_data)
            confidence = round(np.max(proba[0]) * 100, 2)

        response = {"prediction": prediction_label}
        if confidence is not None:
            response["confidence"] = f"{confidence}%"

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    """Form-based prediction for UI."""
    message = request.form.get("message", "")

    if not message.strip():
        return render_template("home.html", prediction_text="Please enter a message.")

    try:
        new_data = vectorizer.transform([message])
        output = classmodel.predict(new_data)[0]
        prediction_label = label_map.get(output, str(output))

        # Confidence score (optional)
        confidence = None
        if hasattr(classmodel, "predict_proba"):
            proba = classmodel.predict_proba(new_data)
            confidence = round(np.max(proba[0]) * 100, 2)
        if confidence:
            text = f"The message is {prediction_label} (Confidence: {confidence}%)"
        else:
            text = f"The message is {prediction_label}"
        return render_template("home.html", prediction_text=text)
    except Exception as e:
        return render_template("home.html", prediction_text=f"Error: {str(e)}")


@app.route("/ping", methods=["GET"])
def ping():
    """Health check route."""
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True)
