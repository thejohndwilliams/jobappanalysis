"""
Job Application Analysis Tool
Analyzes job application data to identify patterns in rejections and predict outcomes.

Author: John Williams
Repository: github.com/thejohndwilliams/jobappanalysis
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


# Default configuration
DEFAULT_CONFIG = {
    "input_file": Path.home() / "OneDrive/Desktop/Career/JobAppDataCURRENT.xlsx",
    "output_dir": Path.home() / "OneDrive/Desktop/Career",
    "sheet_name": None,  # Auto-detect
    "test_size": 0.2,
    "random_state": 42,
    "n_estimators": 100,
    "contamination": 0.1,
}

# Required columns for analysis
REQUIRED_COLUMNS = ["Rejected"]
OPTIONAL_COLUMNS = ["Company", "Job Title", "Phase 2", "Interval", "Application Source", "Resume Version"]


def load_data(file_path: Path, sheet_name: str = None) -> pd.DataFrame:
    """Load Excel data with automatic sheet detection."""
    print(f"Loading data from: {file_path}")
    
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")
    
    xl = pd.ExcelFile(file_path)
    available_sheets = xl.sheet_names
    print(f"Available sheets: {available_sheets}")
    
    # Auto-detect sheet if not specified
    if sheet_name is None:
        if "ALL" in available_sheets:
            sheet_name = "ALL"
        else:
            sheet_name = available_sheets[0]
        print(f"Auto-selected sheet: {sheet_name}")
    
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def validate_columns(df: pd.DataFrame) -> list:
    """Validate and return available columns for analysis."""
    available = [col for col in df.columns if col in REQUIRED_COLUMNS + OPTIONAL_COLUMNS]
    missing_required = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")
    
    print(f"Using columns: {available}")
    return available


def preprocess_data(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Clean and preprocess data for modeling."""
    df = df[columns].copy()
    
    initial_rows = len(df)
    df = df.dropna(subset=["Rejected"])
    print(f"Dropped {initial_rows - len(df)} rows with missing 'Rejected' values")
    
    if "Interval" in df.columns:
        median_interval = df["Interval"].median()
        df["Interval"] = df["Interval"].fillna(median_interval)
        print(f"Filled missing Interval values with median: {median_interval}")
    
    categorical_cols = ["Company", "Job Title", "Application Source", "Resume Version"]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype("category").cat.codes
    
    if "Phase 2" in df.columns:
        df["Phase 2"] = df["Phase 2"].apply(lambda x: 1 if pd.notnull(x) else 0)
    
    print(f"Preprocessed data shape: {df.shape}")
    return df


def train_model(X: pd.DataFrame, y: pd.Series, config: dict) -> tuple:
    """Train Random Forest classifier and return model with predictions."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config["test_size"], random_state=config["random_state"]
    )
    
    model = RandomForestClassifier(
        n_estimators=config["n_estimators"],
        random_state=config["random_state"]
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print("\n=== Model Performance ===")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    
    return model, X_test, y_test, y_pred


def detect_anomalies(X: pd.DataFrame, config: dict) -> pd.Series:
    """Detect anomalous applications using Isolation Forest."""
    iso_forest = IsolationForest(
        n_estimators=config["n_estimators"],
        contamination=config["contamination"],
        random_state=config["random_state"]
    )
    iso_forest.fit(X)
    return iso_forest.predict(X)


def save_visualizations(model, X: pd.DataFrame, output_dir: Path):
    """Generate and save feature importance visualization."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    feature_importances = pd.Series(model.feature_importances_, index=X.columns)
    feature_importances.nlargest(10).plot(kind="barh", ax=ax)
    ax.set_xlabel("Importance")
    ax.set_title("Top Features Impacting Rejection")
    plt.tight_layout()
    
    plot_path = output_dir / f"feature_importance_{timestamp}.png"
    fig.savefig(plot_path, dpi=150)
    print(f"Saved feature importance plot: {plot_path}")
    plt.close(fig)


def save_results(df: pd.DataFrame, y_test, y_pred, output_dir: Path):
    """Save analysis results to Excel."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"JobAppResults_{timestamp}.xlsx"
    
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Data with Anomalies", index=False)
        pd.DataFrame({"Actual": y_test, "Predicted": y_pred}).to_excel(
            writer, sheet_name="Predictions", index=False
        )
    
    print(f"Results saved: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Analyze job application data for rejection patterns"
    )
    parser.add_argument("-i", "--input", type=Path, default=DEFAULT_CONFIG["input_file"],
                        help="Path to input Excel file")
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_CONFIG["output_dir"],
                        help="Output directory for results")
    parser.add_argument("-s", "--sheet", type=str, default=None,
                        help="Sheet name to analyze (auto-detect if not specified)")
    args = parser.parse_args()
    
    config = DEFAULT_CONFIG.copy()
    config["input_file"] = args.input
    config["output_dir"] = args.output
    config["sheet_name"] = args.sheet
    
    try:
        df = load_data(config["input_file"], config["sheet_name"])
        columns = validate_columns(df)
        df = preprocess_data(df, columns)
        
        if len(df) < 10:
            print("ERROR: Not enough data for analysis (minimum 10 rows required)")
            sys.exit(1)
        
        X = df.drop(["Rejected"], axis=1)
        y = df["Rejected"]
        
        model, X_test, y_test, y_pred = train_model(X, y, config)
        
        df["Anomaly"] = detect_anomalies(X, config)
        anomalies = df[df["Anomaly"] == -1]
        print(f"\nDetected {len(anomalies)} anomalous applications")
        
        config["output_dir"].mkdir(parents=True, exist_ok=True)
        save_visualizations(model, X, config["output_dir"])
        save_results(df, y_test, y_pred, config["output_dir"])
        
        print("\n=== Analysis Complete ===")
        
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Unexpected error - {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
