import streamlit as st
import joblib
import pandas as pd
import nltk
import string
import logging
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)

def clean_text(text):
    """Clean and preprocess text for genre prediction.
    
    This function performs several text preprocessing steps:
    1. Converts text to lowercase
    2. Removes punctuation
    3. Tokenizes the text
    4. Removes stopwords
    5. Applies lemmatization
    
    Args:
        text (str): Raw text to be preprocessed
        
    Returns:
        str: Cleaned and preprocessed text
    """
    if not isinstance(text, str):
        logging.warning(f"Received non-string input: {type(text)}. Converting to string.")
        text = str(text)
        
    lemmatizer = nltk.stem.WordNetLemmatizer()
    stop_words = set(nltk.corpus.stopwords.words('english'))
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = ''.join([c for c in text if c not in string.punctuation])
    
    # Tokenize and remove stopwords
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    return ' '.join(tokens)

# Load model and related components
try:
    model = joblib.load('../model/lr_model.pkl')
    tfidf = joblib.load('../model/tfidf_vectorizer.pkl')
    le = joblib.load('../model/label_encoder.pkl')
    misclassified = pd.read_csv('../model/misclassified_examples.csv')
    logging.info("Successfully loaded model and related components")
except Exception as e:
    logging.error(f"Error loading model components: {str(e)}")
    st.error("Failed to load model components. Please check the model files.")
    st.stop()

st.title('Movie Genre Classification')

description = st.text_area('Movie Description', height=150)
if st.button('Predict Genre'):
    if description.strip() == '':
        st.error('Please enter a movie description.')
    else:
        cleaned = clean_text(description)
        X = tfidf.transform([cleaned])
        # Get probability scores for all classes
        pred_probs = model.predict_proba(X)[0]
        # Get top 3 predictions
        top_3_indices = pred_probs.argsort()[-3:][::-1]
        top_3_genres = le.inverse_transform(top_3_indices)
        
        # Display top 3 predictions
        st.success("Top 3 Predicted Genres:")
        for i, genre in enumerate(top_3_genres, 1):
            st.write(f"**{i}. {genre}**")

st.header('Feature Importance by Genre')
st.write('Top words contributing to each genre:')

# Calculate feature importance using the neural network weights
feature_names = tfidf.get_feature_names_out()
# Get the weights from the first layer
weights = np.abs(model.coefs_[0])
# Average the weights across neurons for each feature
importance = np.mean(weights, axis=1)
# Get top features
top_indices = importance.argsort()[-20:][::-1]
top_features = [feature_names[i] for i in top_indices]
top_importance = importance[top_indices]

# Display overall most important features
st.subheader('Most Important Features Overall')

# Create a DataFrame for plotting
feature_importance_df = pd.DataFrame({
    'Feature': top_features[:10],
    'Importance (%)': top_importance[:10] * 100  # Convert to percentage
})

# Create a bar chart using Streamlit
st.bar_chart(
    feature_importance_df.set_index('Feature')['Importance (%)'],
    use_container_width=True
)

# Display the numerical values
for feature, imp in zip(top_features[:10], top_importance[:10]):
    st.write(f"**{feature}**: {imp*100:.2f}%")

st.header('Example Misclassifications')
for _, row in misclassified.iterrows():
    with st.expander(f"True: {row['genre']}, Predicted: {row['predicted']}"):
        st.write(f"**Description**: {row['description']}")
        st.write(f"**True Genre**: {row['genre']}")
        st.write(f"**Predicted Genre**: {row['predicted']}")