import os
import pandas as pd    
import numpy as np          
import uuid
import pickle
import re
from dotenv import load_dotenv
from crewai.tools import tool
import google.generativeai as genai
import time
load_dotenv()

api_key = os.getenv('GEMINI_API_KEY')  

ml_model = None
tfidf_vectorizer = None

def load_ml_models():
    """Load the trained TF-IDF vectorizer and Naive Bayes model"""
    global ml_model, tfidf_vectorizer
    
    try:
        # Try to load saved models
        with open('D:\\Agentic_AI\\Capstone\\data\\feedback_classifier.pkl', 'rb') as f:
            ml_model = pickle.load(f)
        with open('D:\\Agentic_AI\\Capstone\\data\\tfidf_vectorizer.pkl', 'rb') as f:
            tfidf_vectorizer = pickle.load(f)
        print("Loaded saved ML models successfully!")
        return True
    except FileNotFoundError:
        print("Saved models not found.")
        return False
    
def preprocess_text(text):
    """Clean and preprocess text for model"""
    if pd.isna(text):
        return ""
    
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    text = ' '.join([word for word in text.split() if len(word) > 2])
    return text


def extract_technical_details(text):
    if not isinstance(text, str):
        return "No technical details"
    keywords = [
        ("android", "Android device"),
        ("iphone", "iOS device"),
        ("crash", "Crash issue"),
        ("login", "Login failure"),
        ("feature", "Feature issue"),
        ("sync", "Sync issue"),
        ("slow", "Performance issue")
    ]
    return ", ".join(label for key, label in keywords if key in text.lower()) or "General Bug"


def estimate_severity(details):
    """Severity estimation"""
    details = details.lower()
    
    if any(word in details for word in ["crash", "data loss", "won't start"]):
        return "Critical"
    elif any(word in details for word in ["freeze", "login", "sync", "error","problem"]):
        return "High"
    elif any(word in details for word in ["slow", "performance"]):
        return "Medium"
    else:
        return "Low"

def classify_feedback_ml(df, source_type):
    """Enhanced classification using ML model and rule-based fallback"""
    global ml_model, tfidf_vectorizer
    
    df = df.copy()
    
    # Set up source-specific columns
    if source_type == 'review':
        df['feedback_text'] = df['review_text']
        df['source_id'] = df['review_id']
        df['source_type'] = "app_store"
    else:
        df['feedback_text'] = df['body']
        df['source_id'] = df['email_id']
        df['source_type'] = "support_email"

    # Preprocess text 
    df['cleaned_text'] = df['feedback_text'].apply(preprocess_text)
    
    # Use model 
    if ml_model is not None and tfidf_vectorizer is not None:
        print(f"Using ML model for {source_type} classification...")
        
        # Transform text using TF-IDF trained model
        X_tfidf = tfidf_vectorizer.transform(df['cleaned_text'])
        
        # Get predictions and probabilities
        predictions = ml_model.predict(X_tfidf)
        probabilities = ml_model.predict_proba(X_tfidf)
        
        # Map model predictions to categories
        category_mapping = {
            'bug': 'Bug',
            'feature_request': 'Feature Request', 
            'praise': 'Praise',
            'complaint': 'Complaint',
            'spam': 'Spam'
        }
        
        df['category'] = [category_mapping.get(pred, 'Complaint') for pred in predictions]
        df['confidence'] = np.round(np.max(probabilities, axis=1),2)
    else:
        print(f"Using rule-based classification for {source_type}...")
        # Fallback method
        categories = {
            "Bug": ["crash", "not working", "freeze", "error", "fail", "can't", "cannot", "login", "sync", "slow", "unresponsive"],
            "Feature Request": ["please add", "feature request", "would love", "suggestion", "add", "improve", "dark mode", "fingerprint"],
            "Praise": ["love", "amazing", "great","loving", "awesome", "perfect", "thank you", "works perfectly","excellent","oustanding"],
            "Complaint": ["bad", "slow", "expensive", "terrible", "hate", "customer support", "issue", "problem", "poor"],
            "Spam": ["http", "buy now", "free", "subscribe", "click here", "visit our website", "check out", "asdjklqwe"]
        }

        def detect_category_rules(text):
            if not isinstance(text, str):
                return "Spam", 0.3
            text = text.lower()
            
            # spam (highest priority)
            for keyword in categories["Spam"]:
                if keyword in text:
                    return "Spam", 0.9
            
            for category, keywords in categories.items():
                if category != "Spam" and any(key in text for key in keywords):
                    return category, 0.8
            return "Complaint", 0.5

        results = df['cleaned_text'].apply(
            lambda text: pd.Series(detect_category_rules(text), index=['category', 'confidence'])
        )
        df = df.join(results)
    return df
    
    

def extract_feature_summary(text):
    if not isinstance(text, str):
        return "Unclear feature request"
    text = text.lower()
    keywords = ["please add", "feature request", "would love", "suggestion", "add", "improve"]
    sentences = text.split('.')
    for sentence in sentences:
        for keyword in keywords:
            if keyword in sentence:
                return sentence.strip().capitalize()
    return "Unclear feature request"

def generate_titles_batch(feedbacks, batch_size=10):
    """Process multiple feedbacks in one API call"""
    
    all_titles = []
    
    # Process feedbacks in batches
    for i in range(0, len(feedbacks), batch_size):
        batch = feedbacks[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (len(feedbacks) + batch_size - 1) // batch_size
        
        print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} feedbacks)...")
        
        # Create one big prompt for the entire batch
        prompt = """
        Create concise, actionable titles (3-10 words) for these feedbacks. 
        Start with action verbs when appropriate (Fix, Add, Resolve, etc.).
        """
        
        # Add each feedback to the prompt
        for idx, feedback in enumerate(batch, 1):
            prompt += f"{idx}. Feedback: \"{feedback[:200]}...\"\n"
        
        prompt += f"\nRespond with exactly {len(batch)} titles in this format:\n1. [Title for feedback 1]\n2. [Title for feedback 2]\n3. [Title for feedback 3]\n..."
        
        # Try to get titles for this batch
        batch_titles = []
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=300
                )
            )
            
            # Parse the response to extract titles
            lines = response.text.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line and ('.' in line):
                    # Extract title after "1. " or "2. " etc.
                    title = line.split('.', 1)[1].strip()
                    title = title.strip('"\'`')
                    batch_titles.append(title)
            
            # Make sure we have the right number of titles
            while len(batch_titles) < len(batch):
                batch_titles.append(f"Process feedback: {batch[len(batch_titles)][:30]}...")
            
            batch_titles = batch_titles[:len(batch)]  # Don't take more than we need
            
        except Exception as e:
            print(f"Error in batch {batch_num}: {e}")
            # Fallback: create simple titles
            batch_titles = [f"Process: {fb[:30]}..." for fb in batch]
        
        all_titles.extend(batch_titles)
        
        # Small delay between batches
        if i + batch_size < len(feedbacks):
            time.sleep(2)
    
    return all_titles