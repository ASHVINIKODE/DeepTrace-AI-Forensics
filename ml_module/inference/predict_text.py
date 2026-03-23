import pickle
import os

# ✅ Fix path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "models", "text_model.pkl")

# ✅ Load model
with open(model_path, "rb") as f:
    model, vectorizer = pickle.load(f)

def predict_text(text):

    X = vectorizer.transform([text])
    pred = model.predict(X)

    if pred[0] == 0:
        return "Real Text"
    else:
        return "Fake Text"


# ✅ Test
if __name__ == "__main__":
    user_input = input("Enter text: ")
    result = predict_text(user_input)
    print("Result:", result)