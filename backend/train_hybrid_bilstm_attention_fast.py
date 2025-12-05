# Fast training script with optimizations for CPU
import pandas as pd
import numpy as np
import pickle
import json
from bs4 import BeautifulSoup
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Bidirectional, Dense, Dropout, Layer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Optimize TensorFlow for CPU
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)

class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', 
                                shape=(input_shape[-1], input_shape[-1]),
                                initializer='glorot_uniform', trainable=True)
        self.b = self.add_weight(name='attention_bias', 
                                shape=(input_shape[-1],),
                                initializer='zeros', trainable=True)
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, x):
        e = tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        a = tf.nn.softmax(e, axis=1)
        output = x * a
        return tf.reduce_sum(output, axis=1)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

def clean_text(text):
    """Fast text cleaning"""
    if not isinstance(text, str):
        return ""
    # Remove HTML
    text = BeautifulSoup(text, "html.parser").get_text()
    # Lowercase and remove extra spaces
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def build_bilstm_attention_model(vocab_size, embedding_dim=50, max_length=150, lstm_units=64):
    """Smaller, faster model"""
    input_layer = Input(shape=(max_length,))
    embedding = Embedding(vocab_size, embedding_dim, input_length=max_length)(input_layer)
    
    # Smaller BiLSTM
    bilstm = Bidirectional(LSTM(lstm_units, return_sequences=True))(embedding)
    
    # Attention
    attention = AttentionLayer()(bilstm)
    dropout = Dropout(0.3)(attention)
    
    # Output
    output = Dense(1, activation='sigmoid')(dropout)
    
    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def train_body_classifier(csv_path='data/phishing_email.csv', 
                         max_words=8000,  # Good vocabulary size
                         max_length=180):  # Balanced sequence length
    
    print("Loading dataset...")
    df = pd.read_csv(csv_path)
    
    # Get text column
    if 'body' in df.columns:
        texts = df['body'].fillna('').astype(str)
    elif 'text' in df.columns:
        texts = df['text'].fillna('').astype(str)
    elif 'text_combined' in df.columns:
        texts = df['text_combined'].fillna('').astype(str)
    else:
        raise ValueError("No text column found")
    
    labels = df['label'].values
    
    print(f"Dataset size: {len(texts)}")
    print(f"Spam: {sum(labels)}, Legitimate: {len(labels) - sum(labels)}")
    
    # Use FULL dataset - no sampling!
    print("\n✓ Using full dataset for maximum accuracy")
    
    # Clean texts (pre-compute once)
    print("Cleaning text (this may take a few minutes)...")
    cleaned_texts = [clean_text(text) for text in texts]
    
    # Tokenization
    print("Tokenizing...")
    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(cleaned_texts)
    sequences = tokenizer.texts_to_sequences(cleaned_texts)
    X = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
    y = labels
    
    vocab_size = min(len(tokenizer.word_index) + 1, max_words)
    print(f"Vocabulary size: {vocab_size}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Training: {len(X_train)}, Test: {len(X_test)}")
    
    # Build smaller model
    print("Building optimized BiLSTM + Attention model...")
    model = build_bilstm_attention_model(vocab_size, embedding_dim=50, max_length=max_length, lstm_units=64)
    model.summary()
    
    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    checkpoint = ModelCheckpoint('phishing_bilstm_attention_body.h5', monitor='val_accuracy', save_best_only=True)
    
    # Train with larger batch size for speed
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        epochs=10,  # Fewer epochs
        batch_size=256,  # MUCH larger batch for speed
        validation_split=0.15,
        callbacks=[early_stop, checkpoint],
        verbose=1
    )
    
    # Evaluate
    print("\nEvaluating...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Loss: {loss:.4f}")
    
    # Save
    print("Saving...")
    with open('tokenizer_body_attention.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    
    with open('tokenizer_body_attention_meta.json', 'w') as f:
        json.dump({'maxlen': max_length, 'vocab_size': vocab_size}, f)
    
    print("\n✓ Training complete!")
    print("Files saved:")
    print("  - phishing_bilstm_attention_body.h5")
    print("  - tokenizer_body_attention.pkl")
    print("  - tokenizer_body_attention_meta.json")
    
    return model, tokenizer, history

if __name__ == "__main__":
    print("=" * 60)
    print("FAST TRAINING MODE - Optimized for CPU")
    print("=" * 60)
    model, tokenizer, history = train_body_classifier(
        csv_path='data/phishing_email.csv',
        max_words=8000,
        max_length=180
    )
