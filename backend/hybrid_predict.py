import os
import pickle
from typing import Dict

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

BASE_DIR = os.path.dirname(__file__)

SUBJ_VECT_PATH = os.path.join(BASE_DIR, "subject_vectorizer.pkl")
SUBJ_CLF_PATH = os.path.join(BASE_DIR, "subject_logreg.pkl")
BODY_MODEL_PATH = os.path.join(BASE_DIR, "phishing_lstm_body.h5")
BODY_TOK_PATH = os.path.join(BASE_DIR, "tokenizer_body.pkl")
BODY_TOK_META = os.path.join(BASE_DIR, "tokenizer_body_meta.json")

THRESHOLD = float(os.environ.get("SPAM_THRESHOLD", 0.4))
ALPHA_SUBJ = 0.4  # subject weight
ALPHA_BODY = 0.6  # body weight

_subj_vect = None
_subj_clf = None
_body_model = None
_body_tok = None
_body_maxlen = 200

def _load_subject_models():
    global _subj_vect, _subj_clf
    if _subj_vect is None:
        with open(SUBJ_VECT_PATH, "rb") as f:
            _subj_vect = pickle.load(f)
    if _subj_clf is None:
        with open(SUBJ_CLF_PATH, "rb") as f:
            _subj_clf = pickle.load(f)
    return _subj_vect, _subj_clf

def _load_body_models():
    global _body_model, _body_tok, _body_maxlen
    if _body_tok is None:
        with open(BODY_TOK_PATH, "rb") as f:
            _body_tok = pickle.load(f)
        # load meta
        try:
            import json
            with open(BODY_TOK_META, "r") as f:
                meta = json.load(f)
                _body_maxlen = int(meta.get("maxlen", 200))
        except Exception:
            _body_maxlen = 200
    if _body_model is None:
        _body_model = load_model(BODY_MODEL_PATH, compile=False)
    return _body_model, _body_tok, _body_maxlen

def _clean_text(s: str) -> str:
    import re
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def classify_email(subject: str, body: str) -> Dict:
    """
    Hybrid classifier:
      - Subject -> TF-IDF + LogisticRegression
      - Body    -> BiLSTM
      - Final   -> weighted combination of both probabilities
    Returns: {"score": float, "label": "spam"|"legit", "p_subject": float, "p_body": float}
    """
    try:
        subj = _clean_text(subject)
        bd = _clean_text(body)

        # Subject branch
        subj_vect, subj_clf = _load_subject_models()
        X_subj = subj_vect.transform([subj])
        p_subj = float(subj_clf.predict_proba(X_subj)[0, 1])

        # Body branch
        body_model, body_tok, maxlen = _load_body_models()
        seq = body_tok.texts_to_sequences([bd])
        X_body = pad_sequences(seq, maxlen=maxlen, padding="post", truncating="post")
        p_body = float(body_model.predict(X_body, verbose=0)[0, 0])

        # Combine
        score = ALPHA_SUBJ * p_subj + ALPHA_BODY * p_body
        label = "spam" if score >= THRESHOLD else "legit"
        return {
            "score": score,
            "label": label,
            "p_subject": p_subj,
            "p_body": p_body,
        }
    except Exception as e:
        return {"score": 0.0, "label": "unknown", "error": str(e)}
