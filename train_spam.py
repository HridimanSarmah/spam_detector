
import os
import pandas as pd
import requests
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import joblib


DATA_URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"


os.makedirs("models", exist_ok=True)

print("📥 Downloading dataset...")
r = requests.get(DATA_URL, timeout=30)
r.raise_for_status()
df = pd.read_csv(StringIO(r.text), sep='\t', header=None, names=['label','message'])
df['label_num'] = df.label.map({'ham':0, 'spam':1}

X = df['message']
y = df['label_num']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === TF-IDF Vectorizer ===
vect = TfidfVectorizer(min_df=2, ngram_range=(1,2), stop_words='english')
X_train_tfidf = vect.fit_transform(X_train)
X_test_tfidf = vect.transform(X_test)

# === Train Model ===
model = MultinomialNB(alpha=0.1)
model.fit(X_train_tfidf, y_train)

# === Evaluate ===
preds = model.predict(X_test_tfidf)
print("✅ Accuracy:", accuracy_score(y_test, preds))
print("\nClassification Report:\n", classification_report(y_test, preds))

joblib.dump(vect, "models/tfidf_vectorizer.joblib")
joblib.dump(model, "models/spam_model.joblib")

print("💾 Model and vectorizer saved in /models folder")
