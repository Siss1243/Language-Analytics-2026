"""
Test model for Assignment 2
"""

# IMPORTS
from pathlib import Path
import pandas as pd
import csv
import matplotlib.pyplot as plt
from joblib import load
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
from utils import preprocess_text, basic_inspection


# PATHS
OUTPUT_DIR = Path(__file__).parent.parent / "output"
DATA_DIR = Path(__file__).parent.parent / "data"

# MAIN
def test_model():

    #Load test data
    data = pd.read_csv(DATA_DIR / "news_test_data.csv")
    #basic_inspection(data)

    X_test = preprocess_text(data["text"])
    y_test = data["label"]

    print("Data loaded")

    #Load pipeline
    pipeline_file = sorted(OUTPUT_DIR.glob("pipeline_*.joblib"))[-1] #Latest
    pipeline = load(pipeline_file)
    print(f"Loaded pipeline: {pipeline_file}")

    #Predict
    print("Predicting with pipeline")
    y_pred = pipeline.predict(X_test)

    #Evaluate
    results = evaluate_model(y_test, y_pred, pipeline, pipeline_file)
    #analyze_errors(X_test, y_test, y_pred, n_examples=5)
    #show_feature_importance(pipeline, top_n=20)

    #save results
    save_results_to_csv(results)
    plot_and_save_all(pipeline, X_test, y_test, OUTPUT_DIR, y_pred)


#FUNCTIONS

#model evaluation
def evaluate_model(y_test, y_pred, pipeline, pipeline_file):
    """Compute all evaluation metrics and extract parameters"""

    acc = round(accuracy_score(y_test, y_pred), 2)
    precision = round(precision_score(y_test, y_pred, average='macro', zero_division=0), 2)
    recall = round(recall_score(y_test, y_pred, average='macro', zero_division=0), 2)
    f1 = round(f1_score(y_test, y_pred, average='macro', zero_division=0), 2)
    print(f"\nAccuracy: {acc:.4f}\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    params = pipeline.get_params()

    # Extract key parameters (add more if needed)
    results = {
        "model": pipeline_file.name,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,

        # TF-IDF params
        "tfidf_max_features": params.get("tfidf__max_features"),
        "tfidf_ngram_range": params.get("tfidf__ngram_range"),
        "tfidf_min_df": params.get("tfidf__min_df"),
        "tfidf_max_df": params.get("tfidf__max_df"),

        # RF params
        "rf_n_estimators": params.get("rf__n_estimators"),
        "rf_max_depth": params.get("rf__max_depth"),
        "rf_min_samples_split": params.get("rf__min_samples_split"),
        "rf_min_samples_leaf": params.get("rf__min_samples_leaf"),
    }

    return results


#Continous store to CSV to manage results
def save_results_to_csv(results):
    """Append model results to CSV file"""

    output_file = OUTPUT_DIR / "model_results.csv"
    file_exists = output_file.exists()

    # Fixed column order (schema)
    fieldnames = [
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "tfidf_max_features",
        "tfidf_ngram_range",
        "tfidf_min_df",
        "tfidf_max_df",
        "rf_n_estimators",
        "rf_max_depth",
        "rf_min_samples_split",
        "rf_min_samples_leaf",
    ]

    with open(output_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        # Write header only once
        if not file_exists:
            writer.writeheader()

        # Ensure all keys exist (avoid missing columns crash)
        row = {key: results.get(key, None) for key in fieldnames}

        writer.writerow(row)

    print(f"\nResults saved to: {output_file}")

#Superficial analysis of errors
def analyze_errors(X_test, y_test, y_pred, n_examples=10):
    """
    Analyserer de fejl, modellen laver og viser nogle eksempler.

    X_test: pd.Series med teksten
    y_test: rigtige labels
    y_pred: forudsagte labels
    n_examples: antal eksempler at printe pr. fejl-type
    """
    
    # Build DF with results
    df = pd.DataFrame({
        "text": X_test,
        "true": y_test,
        "pred": y_pred
    })
    
    # Exstract errors
    errors = df[df["true"] != df["pred"]]
    
    print(f"Total fejl: {len(errors)} / {len(y_test)}")
    print("Fejl fordelt på type:")
    print(errors.groupby(["true", "pred"]).size())
    
    # Average length of error texts
    errors["length"] = errors["text"].str.len()
    avg_length = errors["length"].mean()
    print(f"\nGennemsnitslængde af fejltekster: {avg_length:.1f} tegn")
    
    # Print example
    for (true_label, pred_label), group in errors.groupby(["true", "pred"]):
        print(f"\nEksempler: {true_label} → {pred_label}")
        for i, row in enumerate(group["text"].head(n_examples)):
            print(f"{i+1}: {row[:200]}...")  # print først 200 tegn
        group_avg_len = group["length"].mean()
        print(f"Gennemsnitslængde for denne fejltype: {group_avg_len:.1f}  tegn")


#Feature importance
def show_feature_importance(pipeline, top_n=20):
    """
    Exstract top-n important features from current pipine
    """

    # Trained steps
    tfidf = pipeline.named_steps['tfidf']
    select = pipeline.named_steps.get('select', None)
    rf = pipeline.named_steps['rf']

    # feature names
    feature_names = tfidf.get_feature_names_out()

    # from select best if used
    if select is not None:
        # select.get_support() giver bool array af valgte features
        feature_names = feature_names[select.get_support()]

    # Exstract feature importance
    importances = rf.feature_importances_

    # range and sort
    feat_importances = pd.Series(importances, index=feature_names)
    top_features = feat_importances.nlargest(top_n)

    # Print top features
    print(f"\nTop {top_n} features:")
    for i, (feat, val) in enumerate(top_features.items(), start=1):
        print(f"{i}. {feat} - {val:.6f}")

    # Plot
    top_features.plot(kind='bar', figsize=(12,6))
    plt.title(f"Top {top_n} vigtige features (TF-IDF + RF)")
    plt.ylabel("Feature Importance")
    plt.show()


#Store visualizations 
def plot_and_save_all(pipeline, X_test, y_test, output_dir, y_pred):
    
    # Create figures folder
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Convert labels to binary (FAKE = 1)
    y_binary = (y_test == "FAKE").astype(int)

    # Get correct probability scores
    fake_index = list(pipeline.classes_).index("FAKE")
    y_scores = pipeline.predict_proba(X_test)[:, fake_index]
    
    # Common style
    plt.style.use("seaborn-v0_8-whitegrid")
    color_main = "#2a6f97"
    color_secondary = "#d62828"
    
    #Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=["REAL", "FAKE"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    
    ax.set_title("Confusion Matrix")
    
    plt.tight_layout()
    plt.savefig(figures_dir / "confusion_matrix.png", dpi=300)
    plt.close()
    
    
    #ROC curve
    fpr, tpr, _ = roc_curve(y_binary, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color=color_main, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(figures_dir / "roc_curve.png", dpi=300)
    plt.close()
    
    #Precision-recall curve
    precision, recall, _ = precision_recall_curve(y_binary, y_scores)
    ap_score = average_precision_score(y_binary, y_scores)
    
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color=color_secondary, label=f"AP = {ap_score:.2f}")
    
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(figures_dir / "precision_recall_curve.png", dpi=300)
    plt.close()
    
    print(f"Figures saved to: {figures_dir}")



# Run main
if __name__ == "__main__":
    test_model()