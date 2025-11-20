import numpy as np
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# =========================
# 1. Load dataset
# =========================
# Make sure your CSV has columns: "text_combined" and "label"
df = pd.read_csv("phishing_email.csv")

texts = df['text_combined'].astype(str).values
labels = df['label'].values

# =========================
# 2. Preprocessing
# =========================
max_words = 5000   # only keep top 5000 words
max_len = 200      # max length of each input sequence

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)

X = tokenizer.texts_to_sequences(texts)
X = pad_sequences(X, maxlen=max_len)

# train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42
)

# =========================
# 3. Build Model
# =========================
model = Sequential([
    Embedding(max_words, 128, input_length=max_len),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# =========================
# 4. Train Model
# =========================
history = model.fit(
    X_train, y_train,
    batch_size=64,
    epochs=5,
    validation_data=(X_test, y_test),
    verbose=1
)

# =========================
# 5. Save Model & Tokenizer
# =========================
model.save("phishing_lstm.h5")

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("✅ Model and tokenizer saved successfully!")
