from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

def evaluate_model(y_test, y_pred, y_original):

    print("\n📋 Classification Report:\n")
    print(classification_report(y_test, y_pred))

    os.makedirs("outputs", exist_ok=True)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Confusion Matrix")

    plt.savefig("outputs/confusion_matrix.png")
    plt.close()

    # Fraud distribution
    plt.figure(figsize=(5, 3))
    sns.countplot(x=y_original)
    plt.title("Fraud Distribution")

    plt.savefig("outputs/fraud_distribution.png")
    plt.close()

    print("✅ Outputs saved")