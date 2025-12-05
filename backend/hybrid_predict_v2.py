"""
Improved Hybrid Phishing Classifier with BiLSTM + Attention
Matches the methodology:
- Subject: TF-IDF + Logistic Regression
- Body: BiLSTM + Attention Mechanism
- Hybrid: Weighted combination
"""
import os
import pickle
import re
from typing import Dict
from bs4 import BeautifulSoup
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow.keras.layers import Layer

# Attention Layer (must be defined for loading the model)
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.W = self.add_weight(name="attention_weight", 
                                shape=(input_shape[-1], input_shape[-1]),
                                initializer="random_normal", 
                                trainable=True)
        self.b = self.add_weight(name="attention_bias", 
                                shape=(input_shape[-1],),
                                initializer="zeros", 
                                trainable=True)
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, x):
        e = tf.nn.tanh(tf.matmul(x, self.W) + self.b)
        e = tf.reduce_sum(e, axis=-1, keepdims=True)
        a = tf.nn.softmax(e, axis=1)
        output = x * a
        return tf.reduce_sum(output, axis=1)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])
    
    def get_config(self):
        return super(AttentionLayer, self).get_config()

BASE_DIR = os.path.dirname(__file__)

# Subject model paths
SUBJ_VECT_PATH = os.path.join(BASE_DIR, "subject_vectorizer.pkl")
SUBJ_CLF_PATH = os.path.join(BASE_DIR, "subject_logreg.pkl")

# Body model paths (with attention)
BODY_MODEL_PATH = os.path.join(BASE_DIR, "phishing_bilstm_attention_body.h5")
BODY_TOK_PATH = os.path.join(BASE_DIR, "tokenizer_body_attention.pkl")
BODY_TOK_META = os.path.join(BASE_DIR, "tokenizer_body_attention_meta.json")

# Fallback to old model if attention model not available
BODY_MODEL_PATH_OLD = os.path.join(BASE_DIR, "phishing_lstm_body.h5")
BODY_TOK_PATH_OLD = os.path.join(BASE_DIR, "tokenizer_body.pkl")
BODY_TOK_META_OLD = os.path.join(BASE_DIR, "tokenizer_body_meta.json")

# Weights for hybrid combination
THRESHOLD = float(os.environ.get("SPAM_THRESHOLD", 0.5))
ALPHA_SUBJ = 0.3  # 30% weight for subject
ALPHA_BODY = 0.7  # 70% weight for body (as per methodology)

# Global model cache
_subj_vect = None
_subj_clf = None
_body_model = None
_body_tok = None
_body_maxlen = 200

def _load_subject_models():
    """Load TF-IDF vectorizer and Logistic Regression for subject classification"""
    global _subj_vect, _subj_clf
    if _subj_vect is None:
        with open(SUBJ_VECT_PATH, "rb") as f:
            _subj_vect = pickle.load(f)
    if _subj_clf is None:
        with open(SUBJ_CLF_PATH, "rb") as f:
            _subj_clf = pickle.load(f)
    return _subj_vect, _subj_clf

def _load_body_models():
    """Load BiLSTM + Attention model and tokenizer for body classification"""
    global _body_model, _body_tok, _body_maxlen
    
    if _body_tok is None:
        # Try loading attention model tokenizer first
        if os.path.exists(BODY_TOK_PATH):
            with open(BODY_TOK_PATH, "rb") as f:
                _body_tok = pickle.load(f)
            try:
                import json
                with open(BODY_TOK_META, "r") as f:
                    meta = json.load(f)
                    _body_maxlen = int(meta.get("maxlen", 200))
            except Exception:
                _body_maxlen = 200
        else:
            # Fallback to old model
            with open(BODY_TOK_PATH_OLD, "rb") as f:
                _body_tok = pickle.load(f)
            try:
                import json
                with open(BODY_TOK_META_OLD, "r") as f:
                    meta = json.load(f)
                    _body_maxlen = int(meta.get("maxlen", 200))
            except Exception:
                _body_maxlen = 200
    
    if _body_model is None:
        # Try loading attention model first
        if os.path.exists(BODY_MODEL_PATH):
            _body_model = load_model(
                BODY_MODEL_PATH, 
                custom_objects={'AttentionLayer': AttentionLayer},
                compile=False
            )
            print("✓ Loaded BiLSTM + Attention model")
        else:
            # Fallback to old model
            _body_model = load_model(BODY_MODEL_PATH_OLD, compile=False)
            print("✓ Loaded BiLSTM model (no attention)")
    
    return _body_model, _body_tok, _body_maxlen

def clean_text_enhanced(text: str) -> str:
    """
    Enhanced text preprocessing:
    - Remove HTML tags
    - Lowercase
    - Remove special characters
    - Remove multiple spaces
    """
    if not text:
        return ""
    
    # Remove HTML
    try:
        text = BeautifulSoup(text, "html.parser").get_text()
    except:
        pass
    
    # Lowercase
    text = text.lower()
    
    # Remove special characters
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def classify_email(subject: str, body: str) -> Dict:
    """
    Hybrid phishing classifier combining:
    1. TF-IDF + Logistic Regression for subject (30% weight)
    2. BiLSTM + Attention for body (70% weight)
    
    Returns: {
        "score": float,           # Final hybrid score
        "label": str,            # "spam" or "legit"
        "p_subject": float,      # Subject probability
        "p_body": float,         # Body probability
        "confidence": float      # How confident the model is
    }
    """
    try:
        # Preprocess
        subj_clean = clean_text_enhanced(subject)
        body_clean = clean_text_enhanced(body)
        
        # === SUBJECT CLASSIFICATION ===
        subj_vect, subj_clf = _load_subject_models()
        X_subj = subj_vect.transform([subj_clean])
        p_subj = float(subj_clf.predict_proba(X_subj)[0, 1])
        
        # === BODY CLASSIFICATION ===
        body_model, body_tok, maxlen = _load_body_models()
        seq = body_tok.texts_to_sequences([body_clean])
        X_body = pad_sequences(seq, maxlen=maxlen, padding='post', truncating='post')
        p_body = float(body_model.predict(X_body, verbose=0)[0, 0])
        
        # === HYBRID COMBINATION ===
        # Weighted average: 30% subject + 70% body
        score = ALPHA_SUBJ * p_subj + ALPHA_BODY * p_body
        
        # Classification based on threshold
        label = "spam" if score >= THRESHOLD else "legit"
        
        # Confidence: how far from the threshold
        confidence = abs(score - THRESHOLD)
        
        return {
            "score": round(score, 4),
            "label": label,
            "p_subject": round(p_subj, 4),
            "p_body": round(p_body, 4),
            "confidence": round(confidence, 4)
        }
    
    except Exception as e:
        print(f"Error in classification: {e}")
        return {
            "score": 0.0,
            "label": "unknown",
            "error": str(e),
            "p_subject": None,
            "p_body": None,
            "confidence": 0.0
        }

# For testing
if __name__ == "__main__":
    # Test the classifier
    test_subject = "URGENT: Verify your account now!"
    test_body = "Dear user, click here to verify your account immediately or it will be suspended."
    
    result = classify_email(test_subject, test_body)
    print("\n=== Test Classification ===")
    print(f"Subject: {test_subject}")
    print(f"Body: {test_body}")
    print(f"\nResult: {result}")
