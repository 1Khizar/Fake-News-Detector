from flask import Flask, request, render_template, jsonify
import joblib
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from datetime import datetime

# --- NLTK setup ---
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
punctuations = set(string.punctuation)
lemmatizer = WordNetLemmatizer()

# --- Load models ---
try:
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
except:
    model = None
    vectorizer = None

# --- Flask setup ---
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

daily_stats = {'total_checks': 0, 'fake_detected': 0, 'real_detected': 0}

# --- Preprocessing function ---
def preprocess(text):
    text = text.lower()
    text = re.sub('<.*?>', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and w not in punctuations]
    return ' '.join(tokens)

# --- Routes ---
@app.route('/')
def home():
    return render_template('index.html', daily_stats=daily_stats)

@app.route('/predict',methods=['POST'])
def predict():
    text = request.form.get("news_text","").strip()

    cleaned = preprocess(text)
    
    # Model prediction
    if model and vectorizer:
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0]
        if pred==0:
            result = "Fake News"
            confidence = round(prob[0]*100,2)
            color="danger"
            icon="⚠️"
        else:
            result="Real News"
            confidence=round(prob[1]*100,2)
            color="success"
            icon="✅"
        
    
    # Update stats
    daily_stats['total_checks']+=1
    if result=='Fake News': 
        daily_stats['fake_detected']+=1
    else:
        daily_stats['real_detected']+=1
    
    return jsonify({
    'success': True,
    'prediction': result,
    'color': color,
    'icon': icon,
    'confidence': confidence,
    'daily_stats': daily_stats
    })


if __name__=='__main__':
    print("🚀 Running Fake News Detection Server...")
    app.run(debug=True)
