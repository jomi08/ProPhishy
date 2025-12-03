# predict.py
import os
import math
import pickle
from typing import List, Dict

# ---------------- CONFIG ----------------
THRESHOLD = float(os.environ.get("SPAM_THRESHOLD", 0.5))

# Default paths (adjust if your files are named differently)
KERAS_MODEL_PATH = os.path.join(os.path.dirname(__file__), "phishing_lstm.h5")
TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), "tokenizer.pkl")
SKLEARN_MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), "vectorizer.pkl")
MAXLEN = int(os.environ.get("PREDICT_MAXLEN", 200))

# ---------------- globals ----------------
_model = None
_tokenizer = None
_sklearn_clf = None
_vect = None
_backend = None  # "keras" or "sklearn" or None


# ----------------- helpers -----------------
def _sigmoid(x):
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except Exception:
        return float(x)


def _detect_backend():
    global _backend
    if _backend is not None:
        return _backend
    # prefer sklearn if both exist and are valid
    if os.path.exists(SKLEARN_MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        _backend = "sklearn"
    elif os.path.exists(KERAS_MODEL_PATH) and os.path.exists(TOKENIZER_PATH):
        _backend = "keras"
    elif os.path.exists(KERAS_MODEL_PATH):
        # if only keras model exists but tokenizer missing, still try keras (may fail)
        _backend = "keras"
    else:
        _backend = None
    return _backend


def _load_sklearn():
    global _sklearn_clf, _vect
    if _sklearn_clf is None or _vect is None:
        with open(SKLEARN_MODEL_PATH, "rb") as f:
            _sklearn_clf = pickle.load(f)
        with open(VECTORIZER_PATH, "rb") as f:
            _vect = pickle.load(f)
    return _sklearn_clf, _vect


def _load_keras():
    global _model, _tokenizer
    if _tokenizer is None:
        with open(TOKENIZER_PATH, "rb") as f:
            _tokenizer = pickle.load(f)
    if _model is None:
        # lazy import to avoid requiring TF unless needed
        try:
            from tensorflow.keras.models import load_model
        except Exception as e:
            raise RuntimeError("Failed to import Keras / TensorFlow: " + str(e))
        _model = load_model(KERAS_MODEL_PATH, compile=False)
    return _model, _tokenizer


def _preprocess_texts_for_keras(texts: List[str], tokenizer, maxlen=MAXLEN):
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    seqs = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(seqs, maxlen=maxlen, padding="post", truncating="post")
    return X


# ---------------- core functions ----------------
def score_emails(texts: List[str]) -> List[float]:
    """
    Returns a list of probabilities (floats between 0 and 1) in same order as texts.
    """
    if not texts:
        return []

    backend = _detect_backend()
    if backend == "sklearn":
        clf, vect = _load_sklearn()
        X = vect.transform(texts)
        # sklearn classifiers commonly have predict_proba
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(X)[:, 1]
            return [float(p) for p in probs]
        # fallback to decision_function -> convert via sigmoid
        if hasattr(clf, "decision_function"):
            scores = clf.decision_function(X)
            # decision_function may return (N,) or (N,1)
            probs = [float(_sigmoid(s if not hasattr(s, "__len__") else s[0])) for s in scores]
            return probs
        # last fallback: predict (0/1)
        preds = clf.predict(X)
        return [1.0 if int(p) == 1 else 0.0 for p in preds]

    elif backend == "keras":
        model, tokenizer = _load_keras()
        X = _preprocess_texts_for_keras(texts, tokenizer, maxlen=MAXLEN)
        out = model.predict(X, verbose=0)
        # normalize output to probability in [0,1]
        probs = []
        # If output is shape (N,) or (N,1) or (N,2)
        import numpy as _np
        arr = _np.array(out)
        if arr.ndim == 1:
            # values are probably probabilities already
            probs = arr.tolist()
        elif arr.ndim == 2 and arr.shape[1] == 1:
            probs = arr[:, 0].tolist()
        elif arr.ndim == 2 and arr.shape[1] == 2:
            # common when final Dense(2, activation='softmax')
            probs = arr[:, 1].tolist()
        else:
            # unexpected shape: attempt to squeeze
            try:
                flat = arr.reshape(arr.shape[0], -1)
                # take first column and apply sigmoid if outside [0,1]
                col = flat[:, 0]
                # check range
                minv, maxv = float(col.min()), float(col.max())
                if minv < 0 or maxv > 1:
                    probs = [_sigmoid(float(v)) for v in col.tolist()]
                else:
                    probs = [float(v) for v in col.tolist()]
            except Exception:
                # safe fallback: return zeros
                probs = [0.0] * arr.shape[0]
        # ensure floats and clipped to [0,1]
        probs = [max(0.0, min(1.0, float(p))) for p in probs]
        return probs

    else:
        # no model found: return zeros (unknown)
        return [0.0 for _ in texts]


def classify_email(text: str) -> Dict:
    """
    Classify a single email string. Returns dict:
      {"score": float, "label": "spam"|"legit"|"unknown"}
    """
    try:
        probs = score_emails([text])
        score = float(probs[0]) if probs else 0.0
        if score is None:
            return {"score": 0.0, "label": "unknown"}
        label = "spam" if score >= THRESHOLD else "legit"
        return {"score": score, "label": label}
    except Exception as e:
        return {"score": 0.0, "label": "unknown", "error": str(e)}
