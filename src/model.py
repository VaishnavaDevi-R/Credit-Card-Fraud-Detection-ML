import joblib
import os
from sklearn.ensemble import RandomForestClassifier

def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=30,
        n_jobs=-1,
        random_state=42
    )

    model.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)

    joblib.dump({
        "model": model,
        "features": X_train.columns.tolist()
    }, "models/fraud_model.pkl")

    print("✅ Model saved correctly")

    return model


def load_model():
    bundle = joblib.load("models/fraud_model.pkl")
    return bundle["model"], bundle["features"]