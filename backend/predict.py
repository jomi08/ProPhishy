import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load model
model = load_model("phishing_lstm.h5")

# Load tokenizer (save it after training)
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

def predict_email(text):
    """Return a dict with numeric score and label.

    score: float between 0 and 1 where higher means more likely phishing.
    label: 'Phishing' if score > 0.5 else 'Safe'
    """
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=200)
    pred = float(model.predict(padded)[0][0])
    label = "Phishing" if pred > 0.5 else "Safe"
    return {"score": pred, "label": label}
