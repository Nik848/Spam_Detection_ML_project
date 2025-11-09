import joblib
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
## Load the model
classmodel = joblib.load("spam_classifier_model.pkl")
vectorizer=joblib.load(open('tfidf_vectorizer.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    message = request.json['data']
    print("Received message:", message)
    new_data = vectorizer.transform([message])
    output = classmodel.predict(new_data)
    return jsonify({'prediction': str(output[0])})


if __name__=="__main__":
    app.run(debug=True)