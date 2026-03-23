import pandas as pd
import pickle
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ✅ Get base path (NO PATH ERROR)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_path = os.path.join(BASE_DIR, "dataset", "text", "data.csv")
model_path = os.path.join(BASE_DIR, "models", "text_model.pkl")

# ✅ Load dataset
df = pd.read_csv(data_path)

texts = df["text"]
labels = df["label"].map({"real": 0, "fake": 1})

# ✅ Convert text → numbers
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# ✅ Train model
model = LogisticRegression()
model.fit(X, labels)

# ✅ Save model + vectorizer
with open(model_path, "wb") as f:
    pickle.dump((model, vectorizer), f)

print("✅ Text model trained and saved successfully")