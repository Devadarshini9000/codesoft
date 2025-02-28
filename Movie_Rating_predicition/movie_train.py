import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import pickle

def prepare_data(df):
    # Drop rows with missing values
    df = df.dropna()
    
    # Convert Year and Duration to numeric
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce')
    
    # Fill missing values with median
    df['Year'] = df['Year'].fillna(df['Year'].median())
    df['Duration'] = df['Duration'].fillna(df['Duration'].median())
    
    # Create label encoders for categorical variables
    encoders = {}
    categorical_columns = ['Director', 'Actor 1', 'Actor 2', 'Actor 3']
    
    for column in categorical_columns:
        encoders[column] = LabelEncoder()
        df[f'{column}_encoded'] = encoders[column].fit_transform(df[column].fillna('Unknown'))
    
    # Handle genres
    df['Genre'] = df['Genre'].fillna('Unknown')
    # Split genres and get unique genres
    all_genres = set()
    for genres in df['Genre'].str.split(','):
        if isinstance(genres, list):
            all_genres.update([g.strip() for g in genres])
    
    # Create binary columns for each genre
    for genre in all_genres:
        df[f'Genre_{genre}'] = df['Genre'].str.contains(genre, case=False, regex=False).astype(int)
    
    # Select features
    feature_columns = ['Year', 'Duration']
    feature_columns.extend([f'{col}_encoded' for col in categorical_columns])
    feature_columns.extend([col for col in df.columns if col.startswith('Genre_')])
    
    X = df[feature_columns]
    y = df['Rating']
    
    return X, y, encoders, list(all_genres)

def train_model():
    # Read the dataset
    df = pd.read_csv('Cleaned_IMDb_Movies_India.csv')
    
    # Prepare data
    X, y, encoders, genres = prepare_data(df)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save the model and encoders
    with open('movie_model.pkl', 'wb') as f:
        pickle.dump({
            'model': model,
            'encoders': encoders,
            'genres': genres,
            'feature_columns': X.columns.tolist()
        }, f)
    
    # Print model performance
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"Train R2 Score: {train_score:.4f}")
    print(f"Test R2 Score: {test_score:.4f}")

if __name__ == "__main__":
    train_model()