import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from data_preprocess import preprocess_data  # Ensure this file is named `data_preprocess.py`

def train_model():
    # First, preprocess the data and save it
    preprocess_data(r"D:\dev\Dataset\advertising.csv")

    # Load preprocessed dataset
    data = pd.read_csv(r"D:\dev\preprocessed_advertising.csv")

    # Split features and target
    X = data[['TV', 'Radio', 'Newspaper']]
    y = data['Sales']

    # Split dataset (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save trained model
    joblib.dump(model, "sales_model.pkl")

    print("✅ Model trained successfully and saved as 'sales_model.pkl'!")

if __name__ == "__main__":
    train_model()
