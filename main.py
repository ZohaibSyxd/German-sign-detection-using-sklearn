import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Base directory for the dataset
train_dir = os.path.join("A2", "2025_A2", "train")
test_dir = os.path.join("A2", "2025_A2", "test")

# Load training metadata and features
train_metadata = pd.read_csv(os.path.join(train_dir, "train_metadata.csv"))
train_features = pd.read_csv(os.path.join(train_dir, "Features", "color_histogram.csv"))
train_hog_pca = pd.read_csv(os.path.join(train_dir, "Features", "hog_pca.csv"))
train_additional = pd.read_csv(os.path.join(train_dir, "Features", "additional_features.csv"))

# Load test metadata and features
test_metadata = pd.read_csv(os.path.join(test_dir, "test_metadata.csv"))
test_features = pd.read_csv(os.path.join(test_dir, "Features", "color_histogram.csv"))
test_hog_pca = pd.read_csv(os.path.join(test_dir, "Features", "hog_pca.csv"))
test_additional = pd.read_csv(os.path.join(test_dir, "Features", "additional_features.csv"))

# Merge train and test features
train_df = train_metadata.merge(train_features, on='image_path') \
                         .merge(train_hog_pca, on='image_path') \
                         .merge(train_additional, on='image_path')
test_df = test_metadata.merge(test_features, on='image_path') \
                       .merge(test_hog_pca, on='image_path') \
                       .merge(test_additional, on='image_path')

def engineer_features(df):
    hog_cols = [col for col in df.columns if 'hog_pca' in col]
    if hog_cols:
        df['hog_mean'] = df[hog_cols].mean(axis=1)
        df['hog_std'] = df[hog_cols].std(axis=1)
        df['hog_max'] = df[hog_cols].max(axis=1)
        df['hog_min'] = df[hog_cols].min(axis=1)
        df['hog_range'] = df['hog_max'] - df['hog_min']
        df['hog_skew'] = df[hog_cols].skew(axis=1)
        df['hog_kurtosis'] = df[hog_cols].kurtosis(axis=1)
    ch_cols = [col for col in df.columns if 'ch_' in col]
    if ch_cols:
        df['ch_mean'] = df[ch_cols].mean(axis=1)
        df['ch_std'] = df[ch_cols].std(axis=1)
        df['ch_max'] = df[ch_cols].max(axis=1)
        df['ch_min'] = df[ch_cols].min(axis=1)
        df['ch_range'] = df['ch_max'] - df['ch_min']
        df['ch_peaks'] = (df[ch_cols] > 0.2).sum(axis=1)
    if all(col in df.columns for col in ['mean_r', 'mean_g', 'mean_b']):
        df['red_blue_ratio'] = df['mean_r'] / (df['mean_b'] + 1e-5)
        df['red_green_ratio'] = df['mean_r'] / (df['mean_g'] + 1e-5)
        df['blue_green_ratio'] = df['mean_b'] / (df['mean_g'] + 1e-5)
        df['rgb_max'] = df[['mean_r', 'mean_g', 'mean_b']].max(axis=1)
        df['rgb_min'] = df[['mean_r', 'mean_g', 'mean_b']].min(axis=1)
        df['rgb_contrast'] = df['rgb_max'] - df['rgb_min']
        df['redness'] = df['mean_r'] / (df['mean_g'] + df['mean_b'] + 1e-5)
        df['color_std'] = df[['mean_r', 'mean_g', 'mean_b']].std(axis=1)
        df['is_bw'] = (df['color_std'] < 10).astype(int)
        df['dominant_color'] = df[['mean_r', 'mean_g', 'mean_b']].idxmax(axis=1).map({'mean_r': 0, 'mean_g': 1, 'mean_b': 2})
    if 'edge_density' in df.columns:
        df['edge_density_bin'] = pd.qcut(df['edge_density'], 3, labels=False, duplicates='drop')
        df['high_edge_density'] = (df['edge_density'] > df['edge_density'].median()).astype(int)
    if 'texture_variance' in df.columns and 'edge_density' in df.columns:
        df['edge_texture_product'] = df['edge_density'] * df['texture_variance']
        df['edge_texture_ratio'] = df['edge_density'] / (df['texture_variance'] + 1e-5)
    return df

def evaluate_model(X, y, model, model_name="Model", cv=5):
    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns if hasattr(X, 'columns') else None)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    cv_scores = cross_val_score(model, X_scaled, y, cv=cv, n_jobs=-1)
    print(f"\n{model_name} - Cross-validation accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    print(f"{model_name} - Validation accuracy: {accuracy_score(y_val, y_pred):.4f}")
    print(f"{model_name} - Classification Report:\n{classification_report(y_val, y_pred)}")
    cm = confusion_matrix(y_val, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"{model_name} Confusion Matrix")
    plt.show()

# Prepare data
X_basic = train_df.drop(['id', 'image_path', 'ClassId'], axis=1)
y = train_df['ClassId']
X_eng = engineer_features(X_basic.copy())

# t-SNE visualization
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_basic)
plt.figure(figsize=(10, 8))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab20', s=10)
plt.title('t-SNE of Feature Space')
plt.colorbar(label='ClassId')
plt.show()

# 1. Random Forest, default params, no engineered features
rf_default = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
evaluate_model(X_basic, y, rf_default, model_name="Random Forest (default, no engineered features)")

# 2. Random Forest, default params, with engineered features
rf_default_eng = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
evaluate_model(X_eng, y, rf_default_eng, model_name="Random Forest (default, engineered features)")

# 3. Random Forest, best params, with engineered features
rf_best = RandomForestClassifier(
    n_estimators=600,
    max_depth=40,
    criterion='entropy',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
evaluate_model(X_eng, y, rf_best, model_name="Random Forest (best params, engineered features)")

# --- SVM Grid Search for best params (on engineered features) ---
svm_param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 0.01, 0.1, 1]
}
svm_grid = GridSearchCV(
    SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42),
    svm_param_grid,
    cv=3,
    n_jobs=-1,
    verbose=1
)
svm_grid.fit(StandardScaler().fit_transform(SimpleImputer(strategy='mean').fit_transform(X_eng)), y)
print("Best SVM params:", svm_grid.best_params_)

# 4. SVM, best params, with engineered features
svm_best = SVC(
    kernel='rbf',
    C=svm_grid.best_params_['C'],
    gamma=svm_grid.best_params_['gamma'],
    class_weight='balanced',
    probability=True,
    random_state=42
)
evaluate_model(X_eng, y, svm_best, model_name="SVM (best params, engineered features)")

# Create stacking ensemble
stacking_clf = StackingClassifier(
    estimators=[
        ('rf', rf_best),
        ('svm', svm_best)
    ],
    final_estimator=LogisticRegression(max_iter=1000, class_weight='balanced'),
    cv=5,
    n_jobs=-1,
    passthrough=False
)
evaluate_model(X_eng, y, stacking_clf, model_name="Stacking Ensemble (RF + SVM)")

# --- Test set evaluation ---
# Drop unnecessary columns from test set
X_test_basic = test_df.drop(['id', 'image_path'], axis=1)
X_test_eng = engineer_features(X_test_basic.copy())
if 'ClassId' in X_test_eng.columns:
    X_test_eng = X_test_eng.drop('ClassId', axis=1)

# Impute and scale using training set stats
imputer = SimpleImputer(strategy='mean')
scaler = StandardScaler()
X_train_imputed = imputer.fit_transform(X_eng)
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_imputed = imputer.transform(X_test_eng)
X_test_scaled = scaler.transform(X_test_imputed)

# Fit stacking classifier on all training data
stacking_clf.fit(X_train_scaled, y)
test_preds = stacking_clf.predict(X_test_scaled)

# Prepare submission
submission = pd.DataFrame({
    'id': test_df['id'],
    'ClassId': test_preds
})
submission.to_csv('submission.csv', index=False)
print("Submission file saved as submission.csv")