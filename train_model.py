import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import os
import logging
import re
from scipy.sparse import hstack
from sklearn.utils.class_weight import compute_class_weight
import shap

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('train_model.log')
    ]
)

class GenreAwareTfidf(TfidfVectorizer):
    """Custom TF-IDF vectorizer that boosts genre-specific keywords"""
    def __init__(self, genre_keywords=None, **kwargs):
        super().__init__(**kwargs)
        self.genre_keywords = genre_keywords or {}

    def transform(self, X):
        X_tfidf = super().transform(X)
        for genre, words in self.genre_keywords.items():
            for word in words:
                if word in self.vocabulary_:
                    X_tfidf[:, self.vocabulary_[word]] *= 5
        return X_tfidf

def add_keyword_features(df):
    """Add binary features based on presence of genre-specific keywords"""
    genre_keywords = {
        'action': ['fight', 'explosion', 'battle'],
        'comedy': ['funny', 'laugh', 'humor'],
        'drama': ['life', 'relationship', 'emotional'],
        'horror': ['scary', 'ghost', 'monster'],
        'scifi': ['space', 'alien', 'robot'],
        'romance': ['love', 'romantic', 'heart']
    }
    
    if 'description_clean' not in df.columns:
        raise KeyError("'description_clean' column missing - run clean_text() first")
    
    for genre, keywords in genre_keywords.items():
        pattern = '|'.join(keywords)
        df[f'has_{genre}'] = df['description_clean'].str.contains(
            pattern, case=False, regex=True
        ).astype(int)

def clean_text(text):
    """Clean and preprocess text"""
    if not isinstance(text, str):
        return ""
    
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    return ' '.join(tokens)

def download_nltk_resources():
    """Download required NLTK resources"""
    resources = ['stopwords', 'punkt', 'wordnet']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}')
            logging.info(f'Resource {resource} already available')
        except LookupError:
            logging.info(f'Downloading {resource}...')
            nltk.download(resource)
            logging.info(f'Successfully downloaded {resource}')

def load_data(file_path, file_type='train'):
    """Load data file with robust error handling"""
    try:
        abs_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), file_path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Data file not found: {abs_path}")
            
        data = []
        with open(abs_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    parts = [part.strip() for part in line.strip().split(':::') if part.strip()]
                    
                    if file_type == 'train' and len(parts) >= 4:
                        data.append({
                            'id': parts[0],
                            'title': parts[1],
                            'genre': parts[2],
                            'description': parts[3]
                        })
                    elif file_type == 'test' and len(parts) >= 3:
                        data.append({
                            'id': parts[0],
                            'title': parts[1],
                            'description': parts[2]
                        })
                    elif file_type == 'solution' and len(parts) >= 3:
                        # For solution files, ensure we're getting the correct genre field (index 2)
                        # This fixes the issue with "Edgar's Lunch (1998)" being treated as a genre
                        data.append({
                            'id': parts[0],
                            'genre': parts[2]
                        })
                except Exception as e:
                    logging.warning(f"Error parsing line {line_num}: {str(e)}")
                    continue
        
        if not data:
            raise ValueError("No valid data records found")
            
        df = pd.DataFrame(data)
        
        # Verify required columns
        if file_type == 'train' and not all(col in df.columns for col in ['id', 'title', 'genre', 'description']):
            raise ValueError("Train data missing required columns")
        elif file_type == 'test' and not all(col in df.columns for col in ['id', 'title', 'description']):
            raise ValueError("Test data missing required columns")
        elif file_type == 'solution' and not all(col in df.columns for col in ['id', 'genre']):
            raise ValueError("Solution data missing required columns")
            
        return df
        
    except Exception as e:
        logging.error(f"Error loading {file_path}: {str(e)}")
        raise

def evaluate_model(model, X, y, le, df, set_name="Test"):
    """Evaluate model performance with error handling"""
    try:
        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        logging.info(f'\n{set_name} Set Evaluation:')
        logging.info(f'Accuracy: {accuracy:.4f}')
        
        report = classification_report(y, y_pred, target_names=le.classes_)
        logging.info(f'Classification Report:\n{report}')
        
        df = df.copy()
        df['predicted'] = le.inverse_transform(y_pred)
        misclassified = df[df['genre'] != df['predicted']]
        logging.info(f'Misclassified samples: {len(misclassified)}')
        
        return misclassified.sample(min(5, len(misclassified)))
    
    except Exception as e:
        logging.error(f"Evaluation failed: {str(e)}")
        return pd.DataFrame()

def main():
    try:
        # Initialize resources
        download_nltk_resources()
        
        # Load data with validation
        logging.info("Loading and validating data...")
        train_df = load_data('data/train_data.txt', 'train')
        test_features = load_data('data/test_data.txt', 'test')
        test_solution = load_data('data/test_data_solution.txt', 'solution')
        
        # Create test_df after validation
        test_df = test_features.merge(test_solution, on='id', how='inner')
        if test_df.empty:
            raise ValueError("No valid test data after merging features and solutions")
        
        # Data cleaning with validation
        logging.info("Cleaning text data...")
        for df in [train_df, test_df]:
            df['description_clean'] = df['description'].apply(clean_text)
            add_keyword_features(df)
        
        # Split data with stratification
        logging.info("Splitting data...")
        train_df, val_df = train_test_split(
            train_df,
            test_size=0.2,
            stratify=train_df['genre'],
            random_state=42
        )
        
        # Vectorization
        logging.info("Vectorizing text...")
        tfidf = GenreAwareTfidf(
            genre_keywords={
                "horror": ["ghost", "haunted"],
                "scifi": ["alien", "spaceship"],
                "romance": ["love", "wedding"]
            },
            max_features=5000,
            ngram_range=(1,2)
        )
        
        X_train = hstack([
            tfidf.fit_transform(train_df['description_clean']),
            train_df.filter(regex='has_').values
        ])
        X_val = hstack([
            tfidf.transform(val_df['description_clean']),
            val_df.filter(regex='has_').values
        ])
        X_test = hstack([
            tfidf.transform(test_df['description_clean']),
            test_df.filter(regex='has_').values
        ])
        
        # Label encoding
        le = LabelEncoder()
        y_train = le.fit_transform(train_df['genre'])
        y_val = le.transform(val_df['genre'])
        y_test = le.transform(test_df['genre'])
        
        # Handle class imbalance
        logging.info("Balancing classes...")
        try:
            smote = SMOTE(random_state=42)
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
            logging.info("SMOTE applied successfully")
        except Exception as e:
            logging.warning(f"SMOTE failed: {str(e)}. Using class weights instead.")
            class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
            sample_weights = np.array([class_weights[y] for y in y_train])
            X_train_res, y_train_res = X_train, y_train
        
        # Model training
        logging.info("Training model...")
        model = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=42,
            verbose=True
        )
        
        if 'sample_weights' in locals():
            model.fit(X_train_res, y_train_res, sample_weight=sample_weights)
        else:
            model.fit(X_train_res, y_train_res)
        
        # Evaluation
        evaluate_model(model, X_val, y_val, le, val_df, "Validation")
        test_misclassified = evaluate_model(model, X_test, y_test, le, test_df)
        
        # Explainability (optional)
        try:
            logging.info("Generating SHAP explanations...")
            X_sample = X_train[:100].toarray()
            explainer = shap.KernelExplainer(model.predict_proba, X_sample)
            shap_values = explainer.shap_values(X_sample)
            shap.summary_plot(shap_values[0], X_sample, 
                            feature_names=list(tfidf.get_feature_names_out()) + list(train_df.filter(regex='has_').columns))
        except Exception as e:
            logging.warning(f"SHAP explanation failed: {str(e)}")
        
        # Save artifacts
        logging.info("Saving artifacts...")
        os.makedirs('model', exist_ok=True)
        joblib.dump(model, 'model/mlp_model.pkl')
        joblib.dump(tfidf, 'model/tfidf_vectorizer.pkl')
        joblib.dump(le, 'model/label_encoder.pkl')
        test_misclassified.to_csv('model/misclassified_examples.csv', index=False)
        
        logging.info("Training completed successfully!")
        
    except Exception as e:
        logging.error(f"Fatal error in main: {str(e)}")
        raise

if __name__ == '__main__':
    main()