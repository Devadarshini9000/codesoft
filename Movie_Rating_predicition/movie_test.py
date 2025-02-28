import streamlit as st
import pandas as pd
import numpy as np
import pickle

def load_model():
    with open('movie_model.pkl', 'rb') as f:
        return pickle.load(f)

def safe_encode(encoder, value):
    """Safely encode a value, returning the code for 'Unknown' if value is unseen"""
    try:
        return encoder.transform([value])[0]
    except:
        try:
            return encoder.transform(['Unknown'])[0]
        except:
            return 0

def predict_rating(model_data, year, duration, genre, director, actor1, actor2, actor3):
    try:
        # Create feature dictionary
        features = {
            'Year': year,
            'Duration': duration,
            'Director_encoded': safe_encode(model_data['encoders']['Director'], director),
            'Actor 1_encoded': safe_encode(model_data['encoders']['Actor 1'], actor1),
            'Actor 2_encoded': safe_encode(model_data['encoders']['Actor 2'], actor2),
            'Actor 3_encoded': safe_encode(model_data['encoders']['Actor 3'], actor3)
        }
        
        # Add genre features
        for g in model_data['genres']:
            features[f'Genre_{g}'] = 1 if g in genre else 0
        
        # Create DataFrame with correct column order
        df = pd.DataFrame([features])
        df = df.reindex(columns=model_data['feature_columns'], fill_value=0)
        
        # Make prediction
        prediction = model_data['model'].predict(df)
        return prediction[0]
    
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None

def main():
    st.title('Movie Rating Predictor')
    
    try:
        model_data = load_model()
        
        # Input fields
        year = st.number_input('Year', min_value=1900, max_value=2023, value=2012)
        duration = st.number_input('Duration (minutes)', min_value=30, max_value=300, value=82)
        
        # Create genre selection
        genres = sorted(list(set(g.replace('Genre_', '') for g in model_data['feature_columns'] if g.startswith('Genre_'))))
        genre = st.selectbox('Genre', genres)
        
        director = st.text_input('Director Name', 'Allyson Patel')
        actor1 = st.text_input('Actor 1 Name', 'Yash Dave')
        actor2 = st.text_input('Actor 2 Name', 'Muntazir Ahmad')
        actor3 = st.text_input('Actor 3 Name', 'Kiran Bhatia')
        
        if st.button('Predict Rating'):
            if all([director, actor1, actor2, actor3]):
                rating = predict_rating(model_data, year, duration, genre, director, actor1, actor2, actor3)
                if rating is not None:
                    # Display prediction
                    st.success(f'Predicted Rating: {rating:.1f}/10')
                    
                    # Visual representation
                    st.subheader('Rating Scale')
                    st.progress(min(rating/10, 1.0))
                    
                    # Feedback based on rating
                    if rating >= 7.5:
                        st.balloons()
                        st.markdown('🌟 **Potential Blockbuster!**')
                    elif rating >= 6.0:
                        st.markdown('👍 **Looks Promising!**')
                    else:
                        st.markdown('🤔 **Might Need Some Improvements**')
            else:
                st.warning('Please fill in all fields')
                
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")

if __name__ == '__main__':
    main()