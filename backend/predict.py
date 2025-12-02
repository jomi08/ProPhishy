# predict.py
import os
import numpy as np

_model = None
_tokenizer = None

def load_tokenizer(path="tokenizer.pkl"):
    global _tokenizer
    if _tokenizer is None:
        import pickle
        if not os.path.exists(path):
            raise FileNotFoundError(f"Tokenizer file not found at {path}")
        with open(path, "rb") as f:
            _tokenizer = pickle.load(f)
    return _tokenizer

def load_model(path="phishing_lstm.h5"):
    global _model
    if _model is None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")
        from tensorflow.keras.models import load_model
        _model = load_model(path, compile=False)
    return _model

def preprocess_texts(texts, tokenizer, maxlen=200):
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    seqs = tokenizer.texts_to_sequences(texts)
    seqs = pad_sequences(seqs, maxlen=maxlen, padding="post", truncating="post")
    return seqs

def classify_email(text, tokenizer_path="tokenizer.pkl", model_path="phishing_lstm.h5", maxlen=200):
    try:
        tok = load_tokenizer(tokenizer_path)
        model = load_model(model_path)
        X = preprocess_texts([text], tok, maxlen=maxlen)
        pred = model.predict(X, verbose=0)
        # assume model outputs shape (1,1) or (1,) depending on final layer
        score = float(pred.reshape(-1)[0])
        label = "spam" if score >= 0.5 else "legit"
        return {"score": score, "label": label}
    except Exception as e:
        return {"score": 0.0, "label": "unknown", "error": str(e)}
