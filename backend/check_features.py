import joblib
import pandas as pd

# Load the feature columns
feature_columns = joblib.load('saved_models/feature_columns.pkl')
print("=" * 50)
print("MODEL FEATURES")
print("=" * 50)
print(f"Number of features: {len(feature_columns)}")
print("\nFeatures list:")
for i, col in enumerate(feature_columns):
    print(f"{i+1}. {col}")

# Load sample of training data to see what values look like
try:
    from ml_models.data_preparation import DataPreprocessor
    preprocessor = DataPreprocessor()
    X, y = preprocessor.load_and_preprocess()
    print("\n" + "=" * 50)
    print("SAMPLE TRAINING DATA (first 5 rows)")
    print("=" * 50)
    print(X.head())
    print("\nData types:")
    print(X.dtypes)
except Exception as e:
    print(f"\nCould not load training data: {e}")

print("\n" + "=" * 50)
print("SOLUTION")
print("=" * 50)
print("The features your model expects are listed above.")
print("When sending data to the API, use EXACTLY these feature names.")
print("\nFor example, if your model expects 'age_at_diagnosis',")
print("use that instead of just 'age' in the API request.")