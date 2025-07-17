import sys
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer  # Changed from CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from newspaper import Article
import re

# URL filtering function
def is_article_url(url):
    return bool(re.search(r'/\d{4}/|/article/|/politics/|/news/|/opinion/|/story/', url)) and not url.rstrip('/').endswith(('/news', '/politics', '/crime', '/opinion', '/story'))

# Paragraph cleaning function
def clean_paragraph(para):
    bad_phrases = [
        "sign up", "privacy policy", "affiliate", "advertising", "terms of service",
        "compensated", "click or buy", "newsletter", "subscribe", "cookies", 
        "By entering your email", "marketing messages", "advertising partners",
        "may be compensated", "receive an affiliate", "terms of service"
    ]
    para_lower = para.lower()
    return not any(bad in para_lower for bad in bad_phrases) and len(para.split()) > 20

# Load and prepare data
df = pd.read_csv('scraped_articles.csv')
df = df[['meaningful_phrase', 'bias', 'outlet', 'url']].dropna()
df = df[df['meaningful_phrase'].str.strip().astype(bool)]
df = df[df['url'].apply(is_article_url)]
df['label'] = df['bias'].map({'left': 0, 'right': 1})

print(f"Filtered to {len(df)} real article URLs")

# *KEY FIX: Use TF-IDF instead of CountVectorizer for better semantic similarity*
vectorizer = TfidfVectorizer(
    ngram_range=(1,3),  # Include unigrams, bigrams, trigrams
    stop_words='english', 
    min_df=2,
    max_features=10000  # Limit features for better performance
)

all_phrases_vec = vectorizer.fit_transform(df['meaningful_phrase'])

# Train classifier
X_train, X_test, y_train, y_test = train_test_split(
    df['meaningful_phrase'], df['label'],
    test_size=0.2, random_state=42, stratify=df['label']
)
clf = LogisticRegression(max_iter=1000).fit(vectorizer.transform(X_train), y_train)

def predict_bias(text: str):
    vec = vectorizer.transform([text])
    prob_right = clf.predict_proba(vec)[0][1]
    label = 'right' if prob_right >= 0.5 else 'left'
    confidence = prob_right if label=='right' else 1 - prob_right
    return label, confidence

def extract_key_terms(text):
    """Extract key terms from input text for better matching"""
    # Simple keyword extraction - get important words
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    # Remove common words
    common_words = {'this', 'that', 'with', 'have', 'will', 'from', 'they', 'been', 'said', 'each', 'which', 'their', 'time', 'would', 'there', 'could', 'other'}
    return [w for w in words if w not in common_words]

def get_most_relevant_articles(input_text, df, vectorizer, all_phrases_vec):
    """Enhanced relevance matching with keyword boosting"""
    
    # Get base similarity scores
    input_vec = vectorizer.transform([input_text])
    base_similarities = cosine_similarity(input_vec, all_phrases_vec).flatten()
    
    # Extract key terms from input
    key_terms = extract_key_terms(input_text)
    print(f"Key terms found: {key_terms}")
    
    # Boost scores for phrases containing key terms
    boosted_similarities = base_similarities.copy()
    for i, phrase in enumerate(df['meaningful_phrase']):
        phrase_lower = phrase.lower()
        # Count how many key terms appear in this phrase
        term_matches = sum(1 for term in key_terms if term in phrase_lower)
        if term_matches > 0:
            # Boost similarity score based on term matches
            boosted_similarities[i] += 0.3 * term_matches
    
    df['similarity'] = boosted_similarities
    
    results = {}
    for bias in ['left', 'right']:
        bias_df = df[df['bias'] == bias].copy()
        if not bias_df.empty:
            # Get top matches
            top_matches = bias_df.nlargest(5, 'similarity')
            print(f"\nTop {bias} matches:")
            for _, row in top_matches.head(3).iterrows():
                print(f"  {row['similarity']:.3f}: {row['meaningful_phrase']}")
            
            # Take the best match
            best_match = top_matches.iloc[0]
            results[bias] = {
                'outlet': best_match['outlet'],
                'url': best_match['url'],
                'meaningful_phrase': best_match['meaningful_phrase'],
                'similarity': best_match['similarity']
            }
    
    return results

def extract_paragraphs(text):
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    if not paragraphs:
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    return paragraphs

def get_best_paragraph(input_text, article_text):
    paragraphs = extract_paragraphs(article_text)
    paragraphs = [p for p in paragraphs if clean_paragraph(p)]
    if not paragraphs:
        return ""
    
    texts = [input_text] + paragraphs
    para_vecs = vectorizer.transform(texts)
    similarities = cosine_similarity(para_vecs[0:1], para_vecs[1:]).flatten()
    best_idx = similarities.argmax()
    return paragraphs[best_idx]

def fetch_article_paragraph(url, input_text):
    try:
        article = Article(url)
        article.download()
        article.parse()
        full_text = article.text
        best_para = get_best_paragraph(input_text, full_text)
        words = best_para.split()
        if len(words) > 100:
            best_para = ' '.join(words[:100]) + '...'
        return best_para if best_para else None
    except Exception as e:
        return None

def print_articles(articles, input_text):
    print("\nMost Relevant Contrasting Perspectives:")
    for bias in ['left', 'right']:
        if articles[bias]:
            url = articles[bias]['url']
            similarity = articles[bias]['similarity']
            print(f"\n{bias.upper()}-leaning (relevance: {similarity:.3f}):")
            print(f"Outlet: {articles[bias]['outlet']}")
            
            # Try to fetch full paragraph
            para = fetch_article_paragraph(url, input_text)
            if para and len(para) > 50:  # Only use if substantial content
                print(f"Excerpt: {para}")
            else:
                # Fallback to meaningful phrase
                print(f"Excerpt: {articles[bias]['meaningful_phrase']}")
            print(f"URL: {url}")

def main():
    print("Enter article text (or 'exit'):")
    while True:
        text = input('\n> ').strip()
        if text.lower() == 'exit':
            print("Exiting.")
            break
            
        # Get prediction
        label, conf = predict_bias(text)
        print(f"\nPredicted bias: {label.upper()} (confidence: {conf:.2%})")
        
        # Get most relevant articles
        articles = get_most_relevant_articles(text, df, vectorizer, all_phrases_vec)
        print_articles(articles, text)

if __name__ == '__main__':
    main()