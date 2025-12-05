"""
Train an improved BiLSTM model with Attention mechanism for email body classification.
This matches the methodology: BiLSTM + Attention for body text analysis.
"""
import os
import json
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Dropout, Layer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Attention Layer
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
        # x shape: (batch_size, time_steps, features)
        e = tf.nn.tanh(tf.matmul(x, self.W) + self.b)
        e = tf.reduce_sum(e, axis=-1, keepdims=True)
        a = tf.nn.softmax(e, axis=1)
        output = x * a
        return tf.reduce_sum(output, axis=1)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

def clean_text(text):
    """Enhanced text cleaning"""
    import re
    from bs4 import BeautifulSoup
    
    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()
    # Lowercase
    text = text.lower()
    # Remove special characters but keep spaces
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def build_bilstm_attention_model(vocab_size, embedding_dim=100, max_length=200, lstm_units=128):
    """
    Build BiLSTM model with Attention mechanism
    """
    input_layer = Input(shape=(max_length,))
    
    # Embedding layer (can use pre-trained GloVe if available)
    embedding = Embedding(vocab_size, embedding_dim, input_length=max_length)(input_layer)
    
    # Bidirectional LSTM
    lstm = Bidirectional(LSTM(lstm_units, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(embedding)
    
    # Attention mechanism
    attention = AttentionLayer()(lstm)
    
    # Dropout for regularization
    dropout = Dropout(0.3)(attention)
    
    # Output layer
    output = Dense(1, activation='sigmoid')(dropout)
    
    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def train_body_classifier(csv_path='data/phishing_email.csv', max_words=10000, max_length=200):
    """
    Train BiLSTM + Attention model for email body classification
    """
    print("Loading dataset...")
    df = pd.read_csv(csv_path)
    
    # Assuming columns: 'body' or 'text', and 'label' (0=legit, 1=spam)
    if 'body' in df.columns:
        texts = df['body'].fillna('').astype(str)
    elif 'text' in df.columns:
        texts = df['text'].fillna('').astype(str)
    elif 'text_combined' in df.columns:
        texts = df['text_combined'].fillna('').astype(str)
    else:
        raise ValueError("No 'body', 'text', or 'text_combined' column found")
    
    labels = df['label'].values
    
    print(f"Dataset size: {len(texts)}")
    print(f"Spam emails: {sum(labels)}, Legitimate: {len(labels) - sum(labels)}")
    
    # Clean texts
    print("Cleaning texts...")
    texts = [clean_text(t) for t in texts]
    
    # Tokenization
    print("Tokenizing...")
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
    y = labels
    
    vocab_size = min(len(tokenizer.word_index) + 1, max_words)
    print(f"Vocabulary size: {vocab_size}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Build model
    print("Building BiLSTM + Attention model...")
    model = build_bilstm_attention_model(vocab_size, embedding_dim=100, max_length=max_length, lstm_units=128)
    model.summary()
    
    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    checkpoint = ModelCheckpoint('phishing_bilstm_attention_body.h5', monitor='val_accuracy', save_best_only=True)
    
    # Train
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=64,
        validation_split=0.2,
        callbacks=[early_stop, checkpoint],
        verbose=1
    )
    
    # Evaluate
    print("\nEvaluating on test set...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Loss: {loss:.4f}")
    
    # Save tokenizer and metadata
    print("Saving tokenizer and metadata...")
    with open('tokenizer_body_attention.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    
    with open('tokenizer_body_attention_meta.json', 'w') as f:
        json.dump({'maxlen': max_length, 'vocab_size': vocab_size}, f)
    
    print("\n✓ Training complete!")
    print("Saved files:")
    print("  - phishing_bilstm_attention_body.h5")
    print("  - tokenizer_body_attention.pkl")
    print("  - tokenizer_body_attention_meta.json")
    
    return model, tokenizer, history

if __name__ == "__main__":
    # Train the model
    model, tokenizer, history = train_body_classifier(
        csv_path='data/phishing_email.csv',
        max_words=10000,
        max_length=200
    )
