# Improving Your ProPhishy Model - Implementation Guide

## Current Implementation ✅

You already have a hybrid approach working:
- **Subject**: TF-IDF + Logistic Regression (40% weight)
- **Body**: BiLSTM (60% weight)
- **Hybrid**: Weighted combination

## Improvements to Match Your Methodology

### 1. Add Attention Mechanism to BiLSTM ⭐

**Why?** Attention helps the model focus on important words like "urgent", "click here", "verify account".

**Steps:**

```bash
# 1. Install required packages
cd backend
pip install beautifulsoup4 lxml

# 2. Train the new model with attention
python train_hybrid_bilstm_attention.py

# 3. This will create:
#    - phishing_bilstm_attention_body.h5
#    - tokenizer_body_attention.pkl
#    - tokenizer_body_attention_meta.json
```

### 2. Use the Improved Predictor

**Option A: Replace current predictor**
```bash
# Backup current predictor
cp hybrid_predict.py hybrid_predict_old.py

# Use new predictor
cp hybrid_predict_v2.py hybrid_predict.py
```

**Option B: Update app.py to import v2**
```python
# In app.py, change:
from hybrid_predict import classify_email

# To:
from hybrid_predict_v2 import classify_email
```

### 3. Adjust Weights (Already in v2)

Changed to match your methodology:
- Subject: **30% weight** (was 40%)
- Body: **70% weight** (was 60%)

This gives more importance to the body text analysis.

### 4. Enhanced Preprocessing

The new version includes:
- ✅ HTML tag removal (using BeautifulSoup)
- ✅ Better text cleaning
- ✅ Consistent preprocessing for training and prediction

## Model Architecture Comparison

### Current (Basic BiLSTM):
```
Input → Embedding → BiLSTM → Dense → Output
```

### Improved (BiLSTM + Attention):
```
Input → Embedding → BiLSTM → Attention → Dropout → Dense → Output
                                  ↑
                      (Focuses on key phrases)
```

## Performance Improvements You'll See

1. **Better Accuracy**: Attention mechanism focuses on important words
2. **More Interpretable**: Can visualize which words influenced the decision
3. **Robust to Noise**: Better at ignoring irrelevant parts of emails
4. **Aligned with Research**: Follows best practices in NLP

## Configuration Options

You can tune these in `hybrid_predict_v2.py`:

```python
# Threshold for spam classification
THRESHOLD = 0.5  # Lower = more aggressive, Higher = more lenient

# Weights for hybrid combination
ALPHA_SUBJ = 0.3  # Subject weight (30%)
ALPHA_BODY = 0.7  # Body weight (70%)
```

## Training Tips

### For Better Results:

1. **More Data**: Use larger datasets (Enron, Nazario, Kaggle)
2. **Balance Classes**: Ensure equal spam/legit examples
3. **Hyperparameter Tuning**:
   - `embedding_dim`: 100-300
   - `lstm_units`: 64-256
   - `max_length`: 150-300
   - `dropout`: 0.2-0.5

### Optional: Use Pre-trained Embeddings (GloVe)

```python
# Download GloVe embeddings
# wget http://nlp.stanford.edu/data/glove.6B.zip
# unzip glove.6B.zip

# In train_hybrid_bilstm_attention.py, replace Embedding layer:
embedding_matrix = load_glove_embeddings('glove.6B.100d.txt', tokenizer)
embedding = Embedding(vocab_size, 100, weights=[embedding_matrix], 
                     trainable=False)(input_layer)
```

## Evaluation Metrics

After training, you'll see:
- **Accuracy**: Overall correctness
- **Precision**: Of emails marked as spam, how many are actually spam?
- **Recall**: Of all spam emails, how many did we catch?
- **F1-Score**: Balance between precision and recall

## Next Steps

1. ✅ Train the attention model (run `train_hybrid_bilstm_attention.py`)
2. ✅ Test it with sample emails
3. ✅ Compare performance with old model
4. ✅ Deploy the better one

## Quick Start

```bash
# Complete workflow
cd /Users/annie/prophishing/ProPhishy/backend

# Make sure you have the dataset
ls data/phishing_email.csv

# Train improved model
python train_hybrid_bilstm_attention.py

# Test it
python hybrid_predict_v2.py

# If satisfied, update app.py to use it
```

## Current vs Improved Performance

| Aspect | Current | Improved |
|--------|---------|----------|
| Body Model | BiLSTM | BiLSTM + Attention |
| Preprocessing | Basic | Enhanced (HTML removal) |
| Subject Weight | 40% | 30% |
| Body Weight | 60% | 70% |
| Interpretability | Low | High (attention weights) |
| Efficiency | Good | Better (focused learning) |

## Troubleshooting

**If training fails:**
- Check dataset format (needs 'body'/'text' and 'label' columns)
- Ensure sufficient RAM (model needs ~2GB)
- Reduce batch_size or max_length if memory issues

**If accuracy is low:**
- Check data quality and balance
- Try different hyperparameters
- Increase training epochs
- Use more training data

## Summary

You're implementing a **research-grade phishing detector**! The improvements align perfectly with your methodology and will give you:

✅ Better accuracy
✅ More robust classification  
✅ Explainable predictions (via attention)
✅ Production-ready system

Your methodology is solid - the attention mechanism is the key enhancement that will push your model to the next level!
