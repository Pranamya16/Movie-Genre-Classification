# Movie Genre Classification

## Objective
Develop a machine learning model to classify movies into genres based on plot descriptions, using three dataset files (train, test, and test solutions). The Streamlit app provides predictions, feature insights, and misclassification examples.

## Model Architecture
- Multi-Layer Perceptron (MLP) neural network
- Architecture: Input Layer → Hidden Layer (100 neurons) → Hidden Layer (50 neurons) → Output Layer
- Features: TF-IDF vectorization with up to 10,000 features
- Early stopping enabled to prevent overfitting

## Dataset Structure
- `train_data.txt`: ID ::: TITLE ::: GENRE ::: DESCRIPTION
- `test_data.txt`: ID ::: TITLE ::: DESCRIPTION
- `test_data_solution.txt`: ID ::: GENRE

## Steps to Run
1. **Clone the Repository**
   ```bash
   git clone https://github.com/Pranamya16/movie-genre-classification.git
   cd movie-genre-classification
