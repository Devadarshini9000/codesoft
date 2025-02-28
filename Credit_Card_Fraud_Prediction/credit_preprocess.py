import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

def preprocess_data(file_path):
    # Load dataset (Ensure correct file format)
    data = pd.read_csv(file_path)

    # Drop missing values (if any)
    data = data.dropna()

    # Normalize 'Amount' column
    scaler = StandardScaler()
    data['Amount'] = scaler.fit_transform(data[['Amount']])

    # Save the scaler for future use
    joblib.dump(scaler, "scaler.pkl")

    # Separate features and target
    X = data.drop(columns=['Class'])
    y = data['Class']

    # Handle class imbalance using SMOTE (Oversampling fraud cases)
    smote = SMOTE(sampling_strategy=0.2, random_state=42)  # 20% fraud cases
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Save preprocessed dataset
    preprocessed_df = pd.DataFrame(X_resampled, columns=X.columns)
    preprocessed_df['Class'] = y_resampled
    preprocessed_df.to_csv("preprocessed_fraud.csv", index=False)

    print("✅ Preprocessed dataset saved as 'preprocessed_fraud.csv'.")

if __name__ == "__main__":
    preprocess_data(r"D:\dev\Dataset\creditcard.csv")  # Use raw string or double backslashes
