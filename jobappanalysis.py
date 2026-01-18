"""
Job Application Analysis Tool
Analyzes job application data to identify patterns in rejections, 
detect anomalies, and provide actionable insights.

Author: John Williams
GitHub: https://github.com/thejohndwilliams/jobappanalysis
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_FILE_PATH = Path.home() / "OneDrive/Desktop/Career/JobAppDataCURRENT.xlsx"
DATA_SHEETS = ["2025", "2024"]  # Sheets containing application data

FEATURE_COLUMNS = [
    "Company",
    "Job Title", 
    "Application Source",
    "Resume Version",
    "Industry",
    "Seniority Level",
    "Interval",
    "Apply vs Post",
    "Has_Referral",
    "Has_Cover_Letter",
    "Made_Screening",
    "Made_Interview",
    "Made_Final_Round",
]

CATEGORICAL_COLUMNS = [
    "Company",
    "Job Title",
    "Application Source", 
    "Resume Version",
    "Industry",
    "Seniority Level",
]


# =============================================================================
# DATA LOADING & PREPROCESSING
# =============================================================================

def load_data(file_path: Path) -> pd.DataFrame:
    """Load and combine data from multiple sheets."""
    print(f"Loading data from: {file_path}")
    
    if not file_path.exists():
        raise FileNotFoundError(f"Excel file not found: {file_path}")
    
    dfs = []
    xl = pd.ExcelFile(file_path)
    
    for sheet in DATA_SHEETS:
        if sheet in xl.sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet)
            df["Source_Year"] = sheet
            dfs.append(df)
            print(f"  Loaded {len(df)} records from sheet '{sheet}'")
        else:
            print(f"  Warning: Sheet '{sheet}' not found, skipping")
    
    if not dfs:
        raise ValueError("No data sheets found")
    
    combined = pd.concat(dfs, ignore_index=True, sort=False)
    print(f"Total records loaded: {len(combined)}")
    return combined


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and engineer features from raw data."""
    print("
Preprocessing data...")
    
    df = df.copy()
    
    df["Was_Rejected"] = df["Rejected"].notna().astype(int)
    
    df["Has_Referral"] = df["Referral"].notna().astype(int)
    df["Has_Cover_Letter"] = df["Cover Letter"].notna().astype(int)
    df["Made_Screening"] = df["Screening"].notna().astype(int)
    df["Made_Interview"] = df["Interview"].notna().astype(int)
    df["Made_Final_Round"] = df["Final Round"].notna().astype(int)
    df["Got_Offer"] = df["Offer"].notna().astype(int)
    
    df["Interval"] = df["Interval"].fillna(df["Interval"].median())
    df["Apply vs Post"] = df["Apply vs Post"].fillna(df["Apply vs Post"].median())
    
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")
    
    print(f"  Target distribution: {df['Was_Rejected'].value_counts().to_dict()}")
    print(f"  Referral rate: {df['Has_Referral'].mean():.1%}")
    print(f"  Interview rate: {df['Made_Interview'].mean():.1%}")
    
    return df


def encode_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Encode categorical features for modeling."""
    print("
Encoding features...")
    
    encoders = {}
    df_encoded = df.copy()
    
    for col in CATEGORICAL_COLUMNS:
        if col in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            encoders[col] = le
            print(f"  {col}: {len(le.classes_)} unique values")
    
    return df_encoded, encoders


# =============================================================================
# MODELING
# =============================================================================

def train_rejection_model(X: pd.DataFrame, y: pd.Series) -> tuple:
    """Train Random Forest to predict rejections."""
    print("
Training rejection prediction model...")
    
    if len(X) < 10:
        raise ValueError("Insufficient data for model training")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"
  Accuracy: {accuracy:.2%}")
    print("
  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Not Rejected", "Rejected"]))
    
    return model, X_test, y_test, y_pred


def detect_anomalies(X: pd.DataFrame, contamination: float = 0.1) -> pd.Series:
    """Detect anomalous applications using Isolation Forest."""
    print(f"
Detecting anomalies (contamination={contamination})...")
    
    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42
    )
    
    predictions = iso_forest.fit_predict(X)
    anomaly_mask = predictions == -1
    
    print(f"  Found {anomaly_mask.sum()} anomalous applications ({anomaly_mask.mean():.1%})")
    return pd.Series(anomaly_mask, index=X.index, name="Is_Anomaly")


# =============================================================================
# ANALYSIS & VISUALIZATION
# =============================================================================

def analyze_feature_importance(model, feature_names: list) -> pd.Series:
    """Extract and rank feature importances."""
    importances = pd.Series(
        model.feature_importances_,
        index=feature_names
    ).sort_values(ascending=False)
    return importances


def generate_insights(df: pd.DataFrame, importances: pd.Series) -> list[str]:
    """Generate actionable insights from the analysis."""
    insights = []
    
    top_features = importances.head(3).index.tolist()
    insights.append(f"Top factors influencing rejection: {', '.join(top_features)}")
    
    if "Has_Referral" in df.columns:
        referral_rejection_rate = df[df["Has_Referral"] == 1]["Was_Rejected"].mean()
        no_referral_rejection_rate = df[df["Has_Referral"] == 0]["Was_Rejected"].mean()
        if referral_rejection_rate < no_referral_rejection_rate:
            diff = no_referral_rejection_rate - referral_rejection_rate
            insights.append(f"Referrals reduce rejection rate by {diff:.1%}")
    
    if "Has_Cover_Letter" in df.columns:
        cl_rejection_rate = df[df["Has_Cover_Letter"] == 1]["Was_Rejected"].mean()
        no_cl_rejection_rate = df[df["Has_Cover_Letter"] == 0]["Was_Rejected"].mean()
        if cl_rejection_rate < no_cl_rejection_rate:
            diff = no_cl_rejection_rate - cl_rejection_rate
            insights.append(f"Cover letters reduce rejection rate by {diff:.1%}")
    
    if "Interval" in df.columns:
        avg_interval = df[df["Was_Rejected"] == 1]["Interval"].mean()
        insights.append(f"Average time to rejection: {avg_interval:.1f} days")
    
    if "Application Source" in df.columns:
        source_stats = df.groupby("Application Source").agg({
            "Was_Rejected": "mean",
            "Job Title": "count"
        }).rename(columns={"Job Title": "Count"})
        source_stats = source_stats[source_stats["Count"] >= 5]
        if len(source_stats) > 0:
            best_source = source_stats["Was_Rejected"].idxmin()
            best_rate = source_stats.loc[best_source, "Was_Rejected"]
            insights.append(f"Best application source: {best_source} ({best_rate:.1%} rejection rate)")
    
    return insights


def create_visualizations(importances: pd.Series, df: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Generate and save visualization plots."""
    print("
Generating visualizations...")
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_files = []
    
    plt.style.use("seaborn-v0_8-whitegrid")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    importances.head(10).plot(kind="barh", ax=ax, color="#2E86AB")
    ax.set_xlabel("Importance")
    ax.set_title("Top Features Impacting Rejection")
    ax.invert_yaxis()
    plt.tight_layout()
    path = output_dir / "feature_importance.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved_files.append(path)
    print(f"  Saved: {path}")
    
    return saved_files


def save_results(df, y_test, y_pred, importances, insights, anomalies, output_path):
    """Save analysis results to Excel."""
    print(f"
Saving results to: {output_path}")
    
    df_with_anomalies = df.copy()
    df_with_anomalies["Is_Anomaly"] = anomalies
    
    anomaly_records = df_with_anomalies[df_with_anomalies["Is_Anomaly"] == True][[
        "Company", "Job Title", "Application Date", "Application Source",
        "Industry", "Seniority Level", "Was_Rejected", "Interval"
    ]]
    
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary_data = {
            "Metric": ["Total Applications", "Rejection Rate", "Referral Rate", 
                      "Interview Rate", "Offer Rate", "Avg Days to Rejection", "Anomalies Detected"],
            "Value": [len(df), f"{df['Was_Rejected'].mean():.1%}", f"{df['Has_Referral'].mean():.1%}",
                     f"{df['Made_Interview'].mean():.1%}", f"{df['Got_Offer'].mean():.1%}",
                     f"{df[df['Was_Rejected']==1]['Interval'].mean():.1f}", anomalies.sum()]
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name="Summary", index=False)
        pd.DataFrame({"Insights": insights}).to_excel(writer, sheet_name="Insights", index=False)
        importances.reset_index().rename(columns={"index": "Feature", 0: "Importance"}).to_excel(
            writer, sheet_name="Feature Importance", index=False)
        pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred}).to_excel(
            writer, sheet_name="Predictions", index=False)
        anomaly_records.to_excel(writer, sheet_name="Anomalies", index=False)
        df_with_anomalies.to_excel(writer, sheet_name="Full Data", index=False)
    
    print(f"  Saved {len(writer.sheets)} sheets")


def main():
    parser = argparse.ArgumentParser(description="Analyze job application data")
    parser.add_argument("-i", "--input", type=Path, default=DEFAULT_FILE_PATH)
    parser.add_argument("-o", "--output", type=Path, default=Path("JobAppResults.xlsx"))
    parser.add_argument("--plots-dir", type=Path, default=Path("plots"))
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--contamination", type=float, default=0.1)
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("JOB APPLICATION REJECTION ANALYSIS")
    print("=" * 60)
    
    try:
        df_raw = load_data(args.input)
        df = preprocess_data(df_raw)
        df_encoded, encoders = encode_features(df)
        
        available_features = [f for f in FEATURE_COLUMNS if f in df_encoded.columns]
        X = df_encoded[available_features]
        y = df_encoded["Was_Rejected"]
        
        model, X_test, y_test, y_pred = train_rejection_model(X, y)
        importances = analyze_feature_importance(model, available_features)
        anomalies = detect_anomalies(X, contamination=args.contamination)
        insights = generate_insights(df, importances)
        
        print("
" + "=" * 60)
        print("KEY INSIGHTS")
        print("=" * 60)
        for i, insight in enumerate(insights, 1):
            print(f"  {i}. {insight}")
        
        if not args.no_plots:
            create_visualizations(importances, df, args.plots_dir)
        
        save_results(df, y_test, y_pred, importances, insights, anomalies, args.output)
        
        print("
Analysis complete!")
        
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
