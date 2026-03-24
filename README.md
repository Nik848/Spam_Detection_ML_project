# 📧 Spam Message Detection (ML + Flask)

### 🔍 Overview  
An end-to-end **Spam Detection System** that uses **Machine Learning** and **NLP** to classify text messages as **Spam** or **Not Spam**.  
Built with **Flask** for deployment and **TF-IDF + LightGBM/XGBoost** for intelligent message prediction.

---

## ⚙️ Tech Stack
**Languages:** Python  
**Frameworks:** Flask  
**Libraries:** scikit-learn, XGBoost, LightGBM, NLTK, Optuna, joblib  
**Visualization:** matplotlib, seaborn  

---

## 🧠 Features
- Clean and preprocess text (stopwords, punctuation, normalization)  
- Train multiple models and select the best using F1-score  
- Confidence score for predictions  
- Interactive **web UI** and **REST API** for real-time classification  

---

## 📁 Project Structure

```text
Spam-Detection-Project/
│
├── cleaning.ipynb              # Text preprocessing & cleaning
├── model.ipynb                 # Model training & evaluation
├── app.py                      # Flask app for serving predictions
├── requirements.txt            # Dependencies list
│
├── model/
│   ├── best_spam_model.pkl     # Trained ML model
│   └── tfidf_vectorizer.pkl    # TF-IDF vectorizer
│
└── templates/
    └── home.html               # Web UI template
```
---

## 🚀 Setup & Run

### Local (development)

```bash
# Clone repository
git clone https://github.com/yourusername/spam-detection-ml.git
cd spam-detection-ml

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```
Then open http://127.0.0.1:5000 in your browser.

---

## 🐳 Docker Deployment

```bash
# Build and start with Docker Compose
docker compose up --build
```
Then open http://localhost:5000 in your browser.

Or build/run manually:
```bash
docker build -t spam-detection .
docker run -p 5000:5000 spam-detection
```

---

## ☁️ Heroku Deployment

```bash
# Login and create app
heroku login
heroku create your-app-name

# Deploy
git push heroku main
heroku open
```

The included `Procfile` tells Heroku to serve the app with **gunicorn**.

🌐 API Example
```bash
curl -X POST http://127.0.0.1:5000/predict_api \
     -H "Content-Type: application/json" \
     -d '{"data": "Congratulations! You won a prize!"}'
```
Response:
```
{"prediction": "Spam", "confidence": "96.4%"}
```
📊 Results
| Model                | F1-Score | Accuracy  |
| -------------------- | -------- | --------- |
| Logistic Regression  | 0.96     | 96.4%     |
| XGBoost              | 0.97     | 97.8%     |
| **LightGBM (Final)** | **0.98** | **98.1%** |

---
👨‍💻 Author: Nikhil Kumar
🔗 LinkedIn https://www.linkedin.com/in/ml-nikhil/
