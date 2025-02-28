import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from credit_preprocess import preprocess_data  # Ensure `credit_preprocess.py` exists

def train_model():
    # Preprocess data first
    preprocess_data(r"D:\dev\Dataset\creditcard.csv")  # Ensure correct path

    # Load preprocessed dataset
    data = pd.read_csv("preprocessed_fraud.csv")

    # Split into features and target
    X = data.drop(columns=['Class'])
    y = data['Class']

    # Train-test split (80-20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Choose model (Uncomment to switch)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    # model = LogisticRegression(max_iter=1000)

    # Train model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Model evaluation
    print("🔹 Model Performance:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    # Save trained model
    joblib.dump(model, "fraud_model.pkl")
    print("Model trained and saved as 'fraud_model.pkl'!")

if __name__ == "__main__":
    train_model()
