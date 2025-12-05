import os
import json
import argparse
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight

import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

BASE_DIR = os.path.dirname(__file__)

def clean_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=os.path.join(BASE_DIR, "data", "phishing_email.csv"),
        help="CSV with subject/body/label or text_combined/label",
    )
    parser.add_argument("--subject_col", default="subject")
    parser.add_argument("--body_col", default="body")
    parser.add_argument("--label_col", default="label")
    parser.add_argument("--num_words", type=int, default=50000)
    parser.add_argument("--maxlen", type=int, default=200)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--tfidf_max_features", type=int, default=20000)
    args = parser.parse_args()

    out_dir = BASE_DIR
    os.makedirs(out_dir, exist_ok=True)

    print("Loading dataset:", args.data)
    df = pd.read_csv(args.data)

    # You only have text_combined + label -> reuse text_combined for both
    if args.subject_col not in df.columns or args.body_col not in df.columns:
        if "text_combined" not in df.columns:
            raise SystemExit(
                f"CSV must have columns subject/body or text_combined + {args.label_col}"
            )
        print("subject/body not found – using text_combined for both subject and body")
        df[args.subject_col] = df["text_combined"]
        df[args.body_col] = df["text_combined"]

    if args.label_col not in df.columns:
        raise SystemExit(f"CSV must have label column: {args.label_col}")

    subjects = df[args.subject_col].fillna("").astype(str).tolist()
    bodies = df[args.body_col].fillna("").astype(str).tolist()
    labels = df[args.label_col].fillna(0).astype(int).to_numpy()

    subjects_clean = [clean_text(t) for t in subjects]
    bodies_clean = [clean_text(t) for t in bodies]

    X_subj_train, X_subj_val, X_body_train, X_body_val, y_train, y_val = train_test_split(
        subjects_clean,
        bodies_clean,
        labels,
        test_size=args.val_size,
        random_state=args.random_state,
        stratify=labels,
    )

    print("Train/Val sizes:", len(y_train), len(y_val))
    print("Train label distrib:", np.bincount(y_train), "Val:", np.bincount(y_val))

    # ---------- SUBJECT: TF-IDF + LogisticRegression ----------
    print("\n[1] Training subject TF-IDF + LogisticRegression...")
    subj_vect = TfidfVectorizer(
        max_features=args.tfidf_max_features,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    X_subj_train_vec = subj_vect.fit_transform(X_subj_train)
    X_subj_val_vec = subj_vect.transform(X_subj_val)

    subj_clf = LogisticRegression(max_iter=1000, n_jobs=-1)
    subj_clf.fit(X_subj_train_vec, y_train)

    # ---------- BODY: BiLSTM ----------
    print("\n[2] Training body BiLSTM...")
    NUM_WORDS = args.num_words
    MAXLEN = args.maxlen

    tokenizer = Tokenizer(num_words=NUM_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_body_train)

    seq_train = tokenizer.texts_to_sequences(X_body_train)
    seq_val = tokenizer.texts_to_sequences(X_body_val)

    X_body_train_pad = pad_sequences(seq_train, maxlen=MAXLEN, padding="post", truncating="post")
    X_body_val_pad = pad_sequences(seq_val, maxlen=MAXLEN, padding="post", truncating="post")

    classes = np.unique(y_train)
    class_weight_dict = None
    if len(classes) > 1:
        cw = compute_class_weight("balanced", classes=classes, y=y_train)
        class_weight_dict = {int(classes[i]): float(cw[i]) for i in range(len(classes))}
    print("Class weights:", class_weight_dict)

    model = Sequential([
        Embedding(input_dim=NUM_WORDS, output_dim=args.embed_dim, input_length=MAXLEN),
        Bidirectional(LSTM(64, return_sequences=False)),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()

    MODEL_PATH = os.path.join(out_dir, "phishing_lstm_body.h5")
    TOK_PATH = os.path.join(out_dir, "tokenizer_body.pkl")
    TOK_META = os.path.join(out_dir, "tokenizer_body_meta.json")

    callbacks = [
        ModelCheckpoint(MODEL_PATH, monitor="val_loss", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
    ]

    history = model.fit(
        X_body_train_pad,
        y_train,
        validation_data=(X_body_val_pad, y_val),
        epochs=args.epochs,
        batch_size=args.batch,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1,
    )

    # Save body tokenizer + meta
    with open(TOK_PATH, "wb") as f:
        pickle.dump(tokenizer, f)
    with open(TOK_META, "w") as f:
        json.dump({"num_words": NUM_WORDS, "maxlen": MAXLEN}, f)

    print("Saved body BiLSTM model to:", MODEL_PATH)
    print("Saved body tokenizer to:", TOK_PATH)

    # ---------- Save subject model ----------
    SUBJ_VECT_PATH = os.path.join(out_dir, "subject_vectorizer.pkl")
    SUBJ_CLF_PATH = os.path.join(out_dir, "subject_logreg.pkl")
    with open(SUBJ_VECT_PATH, "wb") as f:
        pickle.dump(subj_vect, f)
    with open(SUBJ_CLF_PATH, "wb") as f:
        pickle.dump(subj_clf, f)
    print("Saved subject TF-IDF vectorizer to:", SUBJ_VECT_PATH)
    print("Saved subject LogisticRegression to:", SUBJ_CLF_PATH)

    print("Done. Hybrid (subject TF-IDF + body BiLSTM) trained.")

if __name__ == "__main__":
    main()
