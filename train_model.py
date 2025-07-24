import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score

# 🚀 Load and clean dataset
df = pd.read_csv("fake_or_real_news.csv")  # Ensure this file exists locally
df = df[['title', 'text', 'label']].dropna()

# 🧠 Combine title and text
df['content'] = df['title'] + " " + df['text']
X = df['content']
y = df['label'].map({'REAL': 1, 'FAKE': 0})  # Convert labels to 1 (REAL), 0 (FAKE)

# 🧪 Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 🔍 TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 🧨 Train Classifier
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train_vec, y_train)

# ✅ Evaluate
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model trained. Accuracy on test set: {accuracy:.2%}")

# 💾 Save model & vectorizer
os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/news_classifier.pkl')
joblib.dump(vectorizer, 'model/tfidf_vectorizer.pkl')
print("📦 Model and vectorizer saved to /model/")
