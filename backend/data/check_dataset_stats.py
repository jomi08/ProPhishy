# check_dataset_stats.py
import pandas as pd

df = pd.read_csv("data/phishing_email.csv")

print("\n--- Column Names ---")
print(df.columns)

print("\n--- Label Distribution ---")
print(df['label'].value_counts())

print("\n--- Avg text length ---")
print(df['text_combined'].fillna('').astype(str).apply(len).mean())

print("\n--- Example rows ---")
print(df['text_combined'].head(10))

