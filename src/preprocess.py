"""
Data Preprocessing Module: Football Player Performance Rating System
Academic Mini Project - Experiment 12

This module handles:
1. Loading the dataset
2. Data cleaning (missing values check, duplicate removal)
3. Feature selection (dropping non-predictive columns like Player_ID, Player_Name)
4. Categorical feature encoding (One-Hot Encoding for 'Position')
5. Feature scaling (StandardScaler for numerical features)
6. Train-Test Split (80% train, 20% test)
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Features list
NUMERICAL_FEATURES = [
    "Age",
    "Matches_Played",
    "Minutes_Played",
    "Goals",
    "Assists",
    "Pass_Accuracy",
    "Shots",
    "Key_Passes",
    "Tackles",
    "Interceptions",
    "Distance_Covered",
    "Sprint_Speed"
]

CATEGORICAL_FEATURES = ["Position"]

TARGET_FEATURE = "Performance_Rating"

def get_preprocessor():
    """
    Creates a scikit-learn ColumnTransformer that:
    - Scales numerical features to mean 0, variance 1 using StandardScaler
    - One-hot encodes the categorical 'Position' feature
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            ("cat", OneHotEncoder(drop=None, sparse_output=False, handle_unknown="ignore"), CATEGORICAL_FEATURES)
        ]
    )
    return preprocessor

def load_and_preprocess_data(dataset_path="dataset/football_players.csv", test_size=0.2, random_state=42):
    """
    Loads dataset, validates integrity, applies train-test split, and fits the preprocessor.
    
    Returns:
        X_train, X_test, y_train, y_test, preprocessor, df_raw
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")

    # Step 1: Load dataset
    df = pd.read_csv(dataset_path)
    
    # Step 2: Data Cleaning
    # Check and handle missing values
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        # Fill numerical with median and categorical with mode
        for col in NUMERICAL_FEATURES:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
        for col in CATEGORICAL_FEATURES:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].mode()[0])
                
    # Check and remove duplicates
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        df = df.drop_duplicates()

    # Step 3: Feature & Target separation
    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_FEATURE]
    
    # Step 4: Train-Test Split (80% training, 20% testing)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Step 5: Feature Scaling & Encoding via ColumnTransformer
    preprocessor = get_preprocessor()
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)
    
    return X_train, X_test, y_train, y_test, preprocessor, df

if __name__ == "__main__":
    print("=" * 60)
    print(" STEP 3: RUNNING DATA PREPROCESSING CHECKS")
    print("=" * 60)
    
    csv_path = "dataset/football_players.csv"
    raw_df = pd.read_csv(csv_path)
    print(f"\n1. Loaded Dataset Successfully: {csv_path}")
    print(f"   Total Records: {raw_df.shape[0]}, Total Columns: {raw_df.shape[1]}")
    
    print("\n2. Checking Missing Values per Column:")
    missing = raw_df.isnull().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "   No missing values found (0 across all columns).")
    
    print("\n3. Checking Duplicate Records:")
    dups = raw_df.duplicated().sum()
    print(f"   Duplicate rows found: {dups}")
    
    print("\n4. Feature Separation:")
    print("   Dropped Non-Predictive Columns: ['Player_ID', 'Player_Name']")
    print(f"   Numerical Features ({len(NUMERICAL_FEATURES)}): {NUMERICAL_FEATURES}")
    print(f"   Categorical Features ({len(CATEGORICAL_FEATURES)}): {CATEGORICAL_FEATURES}")
    print(f"   Target Variable: {TARGET_FEATURE}")
    
    X_train, X_test, y_train, y_test, preprocessor, df = load_and_preprocess_data(csv_path)
    
    # Extract feature names after One-Hot Encoding
    encoded_cat_names = list(preprocessor.named_transformers_['cat'].get_feature_names_out(CATEGORICAL_FEATURES))
    all_feature_names = NUMERICAL_FEATURES + encoded_cat_names
    
    print("\n5. Train-Test Split (80% Train, 20% Test):")
    print(f"   Training Set X shape: {X_train.shape} | y shape: {y_train.shape}")
    print(f"   Testing Set  X shape: {X_test.shape}  | y shape: {y_test.shape}")
    
    print("\n6. Features after One-Hot Encoding & Scaling:")
    print(f"   Total features after encoding: {len(all_feature_names)}")
    for idx, f in enumerate(all_feature_names, 1):
        print(f"    {idx:2d}. {f}")
        
    print("\n7. Sample Scaled Feature Vector (First training row):")
    print(np.round(X_train[0], 3))
    print("\nPreprocessing completed and verified successfully!")
    print("=" * 60)
