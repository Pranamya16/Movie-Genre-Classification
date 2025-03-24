import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import logging

def evaluate_model(model, X, y, label_encoder, df, dataset_name="Test"):
    """Evaluate model performance and return misclassified examples.
    
    Args:
        model: Trained model instance
        X: Feature matrix
        y: True labels (encoded)
        label_encoder: LabelEncoder instance
        df: Original DataFrame with text data
        dataset_name: Name of the dataset being evaluated (default: "Test")
        
    Returns:
        DataFrame containing misclassified examples
    """
    # Make predictions
    y_pred = model.predict(X)
    
    # Calculate accuracy
    accuracy = accuracy_score(y, y_pred)
    logging.info(f"\n{dataset_name} Accuracy: {accuracy:.4f}")
    
    # Generate classification report
    class_names = label_encoder.classes_
    report = classification_report(y, y_pred, target_names=class_names)
    logging.info(f"\n{dataset_name} Classification Report:\n{report}")
    
    # Find misclassified examples
    misclassified_idx = y != y_pred
    misclassified_df = df.iloc[misclassified_idx].copy()
    misclassified_df['predicted'] = label_encoder.inverse_transform(y_pred[misclassified_idx])
    
    # Log some misclassified examples
    logging.info(f"\nSample of misclassified examples from {dataset_name} set:")
    for _, row in misclassified_df.head().iterrows():
        logging.info(f"True: {row['genre']}, Predicted: {row['predicted']}")
        logging.info(f"Description: {row['description']}\n")
    
    return misclassified_df