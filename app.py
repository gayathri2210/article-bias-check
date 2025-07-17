import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from newspaper import Article
import re
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# --- DATA LOADING AND MODEL TRAINING (Happens only once when the server starts) ---

print("Loading data and training model...")

# Load and prepare data
df = pd.read_csv('scraped_articles.csv')
df = df[['meaningful_phrase', 'bias', 'outlet', 'url']].dropna()
df = df[df['meaningful_phrase'].str.strip().astype(bool)]
df['label'] = df['bias'].map({'left': 0, 'right': 1}).dropna() # Drop rows where bias is not 'left' or 'right'
df = df.dropna(subset=['label']) # Ensure no NaN labels
df['label'] = df['label'].astype(int)

# Use TF-IDF for better semantic similarity
vectorizer = TfidfVectorizer(
    ngram_range=(1, 3),
    stop_words='english',
    min_df=2,
    max_features=10000
)
all_phrases_vec = vectorizer.fit_transform(df['meaningful_phrase'])

# Train classifier
clf = LogisticRegression(max_iter=1000).fit(all_phrases_vec, df['label'])

print("Model ready.")

# --- UTILITY FUNCTIONS (Your original functions) ---

def predict_bias(text: str):
    vec = vectorizer.transform([text])
    prob_right = clf.predict_proba(vec)[0][1]
    label = 'right' if prob_right >= 0.5 else 'left'
    confidence = prob_right if label == 'right' else 1 - prob_right
    return label, confidence

def extract_key_terms(text):
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    common_words = {'this', 'that', 'with', 'have', 'will', 'from', 'they', 'been', 'said', 'each', 'which', 'their', 'time', 'would', 'there', 'could', 'other'}
    return [w for w in words if w not in common_words]

def get_most_relevant_articles(input_text, df, vectorizer, all_phrases_vec):
    input_vec = vectorizer.transform([input_text])
    base_similarities = cosine_similarity(input_vec, all_phrases_vec).flatten()
    
    key_terms = extract_key_terms(input_text)
    
    boosted_similarities = base_similarities.copy()
    for i, phrase in enumerate(df['meaningful_phrase']):
        term_matches = sum(1 for term in key_terms if term in phrase.lower())
        if term_matches > 0:
            boosted_similarities[i] += 0.3 * term_matches
    
    df['similarity'] = boosted_similarities
    
    results = {}
    for bias in ['left', 'right']:
        bias_df = df[df['bias'] == bias].copy()
        if not bias_df.empty:
            best_match = bias_df.nlargest(1, 'similarity').iloc[0]
            results[bias] = {
                'outlet': best_match['outlet'],
                'url': best_match['url'],
                'meaningful_phrase': best_match['meaningful_phrase'],
                'similarity': best_match['similarity']
            }
    return results

# --- FLASK APP SETUP ---

app = Flask(__name__)
CORS(app)  # This enables communication from the frontend

@app.route('/')
def home():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """The main API endpoint."""
    try:
        data = request.get_json()
        input_text = data['text']

        if not input_text or not input_text.strip():
            return jsonify({'error': 'Input text cannot be empty.'}), 400

        # Get bias prediction
        label, confidence = predict_bias(input_text)
        
        # Get contrasting articles
        articles = get_most_relevant_articles(input_text, df, vectorizer, all_phrases_vec)

        # Prepare the response
        response_data = {
            'prediction': {
                'label': label,
                'confidence': confidence
            },
            'articles': articles
        }
        
        return jsonify(response_data)

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({'error': 'An internal error occurred.'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
