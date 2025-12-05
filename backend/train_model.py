# train_model_improved.py
"""
Improved training script for phishing LSTM.

Expect CSV with columns:
  text_combined,label
where label is 0 or 1 (0=legit, 1=spam).

Saves:
  - phishing_lstm.h5
  - tokenizer.pkl
  - tokenizer_meta.json  (stores num_words and maxlen)

Run examples:
  # simplest: put CSV at backend/data/phishing_email.csv and just run:
  #   python train_model_improved.py
  #
  # or specify a different dataset / output dir:
  #   python train_model_improved.py --data path/to/your.csv --out_dir .
"""

import os
import json
import argparse
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# ---------- paths / args ----------
BASE_DIR = os.path.dirname(__file__)  # backend/ directory

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data",
    default=os.path.join(BASE_DIR, "data", "phishing_email.csv"),
    help="CSV file with text_combined,label (default: backend/data/phishing_email.csv)",
)
parser.add_argument(
    "--out_dir",
    default=BASE_DIR,
    help="Directory to save model/tokenizer (default: backend/ directory)",
)
parser.add_argument("--num_words", type=int, default=50000, help="Top vocabulary size (num_words)")
parser.add_argument("--maxlen", type=int, default=200, help="Sequence maxlen")
parser.add_argument("--embed_dim", type=int, default=128, help="Embedding size")
parser.add_argument("--batch", type=int, default=64, help="Batch size")
parser.add_argument("--epochs", type=int, default=6, help="Number of training epochs")
parser.add_argument("--val_size", type=float, default=0.1, help="Validation split fraction")
parser.add_argument("--random_state", type=int, default=42)
args = parser.parse_args()

OUT_DIR = args.out_dir
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- load ----------
print("Loading dataset:", args.data)
df = pd.read_csv(args.data)
if "text_combined" not in df.columns or "label" not in df.columns:
    raise SystemExit("CSV must have columns: text_combined,label")

texts = df["text_combined"].fillna("").astype(str).tolist()
labels = df["label"].fillna(0).astype(int).tolist()
labels = np.array(labels)

# ---------- basic cleaning (aligns with tokenizer expectations) ----------
import re
def clean_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

texts = [clean_text(t) for t in texts]

# ---------- split ----------
train_texts, val_texts, y_train, y_val = train_test_split(
    texts, labels, test_size=args.val_size, random_state=args.random_state, stratify=labels
)
print("Train/Val sizes:", len(train_texts), len(val_texts))
print("Train label distribution:", np.bincount(y_train), "Val label distribution:", np.bincount(y_val))

# ---------- tokenizer ----------
NUM_WORDS = args.num_words
MAXLEN = args.maxlen
print("Fitting Tokenizer(num_words=%d, oov_token='<OOV>')..." % NUM_WORDS)
tokenizer = Tokenizer(num_words=NUM_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(train_texts)

# sequences
X_train = tokenizer.texts_to_sequences(train_texts)
X_val = tokenizer.texts_to_sequences(val_texts)

X_train = pad_sequences(X_train, maxlen=MAXLEN, padding="post", truncating="post")
X_val = pad_sequences(X_val, maxlen=MAXLEN, padding="post", truncating="post")
print("X_train shape:", X_train.shape, "X_val shape:", X_val.shape)

# ---------- class weights ----------
classes = np.unique(y_train)
class_weight_dict = None
if len(classes) > 1:
    cw = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight_dict = {int(classes[i]): float(cw[i]) for i in range(len(classes))}
print("Class weights:", class_weight_dict)

# ---------- model ----------
print("Building model (Embedding -> BiLSTM -> Dense)...")
model = Sequential([
    Embedding(input_dim=NUM_WORDS, output_dim=args.embed_dim, input_length=MAXLEN),
    Bidirectional(LSTM(64, return_sequences=False)),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

MODEL_PATH = os.path.join(OUT_DIR, "phishing_lstm.h5")
TOK_PATH = os.path.join(OUT_DIR, "tokenizer.pkl")
TOK_META = os.path.join(OUT_DIR, "tokenizer_meta.json")

callbacks = [
    ModelCheckpoint(MODEL_PATH, monitor="val_loss", save_best_only=True, verbose=1),
    EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1)
]

# ---------- train ----------
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=args.epochs,
    batch_size=args.batch,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)

# ---------- save tokenizer & meta ----------
with open(TOK_PATH, "wb") as f:
    pickle.dump(tokenizer, f)
with open(TOK_META, "w") as f:
    json.dump({"num_words": NUM_WORDS, "maxlen": MAXLEN}, f)
print("Saved model to:", MODEL_PATH)
print("Saved tokenizer to:", TOK_PATH, "and meta to:", TOK_META)
print("Done.")
