"""
Job Application Analysis Tool
Analyzes job application data to identify rejection patterns and predict outcomes.
Updated: January 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    'data_path': Path(r'C:\Users\johnw\OneDrive\Desktop\Career\JobAppDataCURRENT.xlsx'),
    'output_dir': Path(r'C:\Users\johnw\OneDrive\Desktop\Career'),
    'sheets_to_analyze': ['2025', '2024'],
    'target_column': 'Rejected',
    'feature_columns': [
        'Company', 'Job Title', 'Application Source', 'Resume Version',
        'Industry', 'Seniority Level', 'Referral', 'Interval', 'Apply vs Post'
    ],
    'stage_columns': ['Screening', 'Interview', 'Final Round'],
    'random_state': 42,
    'test_size': 0.2
}


def load_and_combine_data(config: dict) -> pd.DataFrame:
    """Load and combine data from multiple sheets."""
    print(f"Loading data from: {config['data_path']}")
    
    if not config['data_path'].exists():
        raise FileNotFoundError(f"Data file not found: {config['data_path']}")
    
    dfs = []
    for sheet in config['sheets_to_analyze']:
        try:
            df = pd.read_excel(config['data_path'], sheet_name=sheet)
            df['Year'] = sheet
            dfs.append(df)
            print(f"  Loaded {len(df)} rows from sheet '{sheet}'")
        except Exception as e:
            print(f"  Warning: Could not load sheet '{sheet}': {e}")
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"Total records: {len(combined)}")
    return combined


def engineer_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Create derived features for better prediction."""
    df = df.copy()
    
    stage_cols = config['stage_columns']
    for col in stage_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: 1 if pd.notna(x) and x not in ['', 'No', 0] else 0)
    
    df['Pipeline_Score'] = df[stage_cols].sum(axis=1) if all(c in df.columns for c in stage_cols) else 0
    
    if 'Referral' in df.columns:
        df['Had_Referral'] = df['Referral'].apply(lambda x: 1 if pd.notna(x) and x not in ['', 'No', 0] else 0)
    
    if 'Cover Letter' in df.columns:
        df['Had_Cover_Letter'] = df['Cover Letter'].apply(lambda x: 1 if pd.notna(x) and x not in ['', 'No', 0] else 0)
    
    if 'Application Date' in df.columns:
        df['Application Date'] = pd.to_datetime(df['Application Date'], errors='coerce')
        df['Day_of_Week'] = df['Application Date'].dt.dayofweek
        df['Month'] = df['Application Date'].dt.month
    
    return df


def prepare_features(df: pd.DataFrame, config: dict) -> tuple:
    """Prepare feature matrix and target variable."""
    df = df.copy()
    
    target_col = config['target_column']
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found")
    
    df['Target'] = df[target_col].apply(lambda x: 1 if pd.notna(x) and x not in ['', 'No', 0] else 0)
    
    feature_cols = []
    encoders = {}
    
    categorical = ['Company', 'Job Title', 'Application Source', 'Resume Version', 
                   'Industry', 'Seniority Level', 'Year']
    
    for col in categorical:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown').astype(str)
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col])
            feature_cols.append(f'{col}_encoded')
            encoders[col] = le
    
    numeric = ['Interval', 'Apply vs Post', 'Pipeline_Score', 'Had_Referral', 
               'Had_Cover_Letter', 'Day_of_Week', 'Month']
    
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median())
            feature_cols.append(col)
    
    X = df[feature_cols]
    y = df['Target']
    
    print(f"Features prepared: {len(feature_cols)} features, {len(y)} samples")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    return X, y, feature_cols, encoders, df


def train_models(X: pd.DataFrame, y: pd.Series, config: dict) -> dict:
    """Train and evaluate multiple models."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config['test_size'], random_state=config['random_state']
    )
    
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=config['random_state']),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=config['random_state'])
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Training: {name}")
        print('='*50)
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(model, X, y, cv=5)
        
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Cross-validation: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'y_test': y_test,
            'y_pred': y_pred
        }
    
    return results
