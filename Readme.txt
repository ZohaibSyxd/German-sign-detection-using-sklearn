# Machine Learning Traffic Sign Classification

This project performs classification of traffic sign images using engineered features and ensemble machine learning models. The workflow includes feature engineering, model selection, hyperparameter tuning, and submission to a Kaggle in-class competition.

## Project Structure

- **main.py**: Main script for data loading, feature engineering, model training, evaluation, and Kaggle submission file creation.
- **A2/2025_A2/train/**: Directory containing training metadata and features.
- **A2/2025_A2/test/**: Directory containing test metadata and features.

## Features

- **Feature Engineering**: Extracts statistical and interaction features from HOG, color histograms, and additional image features.
- **Models**:
  - Random Forest (default and tuned)
  - Support Vector Machine (SVM, default and tuned)
  - Stacking Ensemble (Random Forest + SVM)
- **Evaluation**: Prints cross-validation accuracy, validation accuracy, class-wise performance, and confusion matrix for each model.
- **Visualization**: t-SNE plot of the feature space.
- **Kaggle Submission**: Generates a `submission.csv` file for the test set.

## How to Run

1. Place all required data files in the specified directories.
2. Run `main.py` in your Python environment.
3. After training and evaluation, a `submission.csv` file will be created in the project directory.
4. Upload `submission.csv` to the Kaggle competition page.

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib