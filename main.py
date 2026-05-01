import os
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from src.data_loader import load_data
from src.model import train_model, load_model
from src.evaluate import evaluate_model

print("📥 Loading data...")
df = load_data("data/creditcard.csv")

# Sample for speed
df = df.sample(n=50000, random_state=42)

X = df.drop("Class", axis=1)
y = df["Class"]

print("⚖️ Applying SMOTE...")
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42
)

model_path = "models/fraud_model.pkl"

if os.path.exists(model_path):
    print("📦 Loading model...")
    model, features = load_model()
else:
    print("🤖 Training model...")
    model = train_model(X_train, y_train)

y_pred = model.predict(X_test)

evaluate_model(y_test, y_pred, y)