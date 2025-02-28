import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

def preprocess_data(file_path):
    # Load dataset
    data = pd.read_csv(file_path)
    
    # Drop missing values (if any)
    data = data.dropna()
    
    # Select features and target variable
    X = data[['TV', 'Radio', 'Newspaper']]
    y = data['Sales']

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save the scaler for later use in Streamlit
    joblib.dump(scaler, "scaler.pkl")

    # Convert back to DataFrame for saving
    X_scaled_df = pd.DataFrame(X_scaled, columns=['TV', 'Radio', 'Newspaper'])
    X_scaled_df['Sales'] = y  # Add target variable back

    # Save preprocessed dataset
    X_scaled_df.to_csv("preprocessed_advertising.csv", index=False)

    print("✅ Preprocessed dataset saved as 'preprocessed_advertising.csv'.")

    return X_scaled, y

# Run preprocessing when script is executed
if __name__ == "__main__":
    preprocess_data(r"D:\dev\Dataset\advertising.csv")
