"""
Train model for assignment 2
"""

#IMPORTS
from pathlib import Path
import pandas as pd
import numpy as np
from utils import preprocess_text, basic_inspection
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline      
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import SelectKBest, chi2
from joblib import dump
import datetime

#PATHS
OUTPUT_DIR = Path(__file__).parent.parent / "output"
DATA_DIR = Path(__file__).parent.parent / "data"

#RANDOM STATE
random_state = 42

#TIME
timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

#PARAMETER SPACE 
param_dist = {
    # TF-IDF
    #'tfidf__max_features': [800, 10000, 12000],
    #'tfidf__ngram_range': [(1,3)],
    #'tfidf__min_df': [2,3,5],
    #'tfidf__max_df': [0.8, 0.9, 1.0],
    #'tfidf__sublinear_tf': [True, False],
    #'tfidf__binary': [True, False],

    # Random Forest
    #'rf__n_estimators': [5, 50, 100, 200],
    #'rf__max_depth': [None, 2, 10, 20],
    #'rf__min_samples_split': [1,2,3],
    #'rf__min_samples_leaf': [1,2,3],
    #'rf__max_features': ['sqrt', 'log2']
}

#MAIN
def train_model():
    """
    Main function to train and tune model

    opt in and out functionality for grid or random search

    Parameter space above is for manual paramter search choices
    """

    #Load the data
    data = pd.read_csv(DATA_DIR / "news_train_data.csv")
    #basic_inspection(data) #data stats before preprocessing
    print("Data loaded")

    #Preprocess the data
    data["text"] = preprocess_text(data["text"])
    print("data preprocessed")
    #basic_inspection(data) #data stats after preprocessing

    #Split the data 
    X_train = data["text"]
    y_train = data["label"]
    
    #Build pipeline
    pipeline = Pipeline([
        ('tfidf', build_tfidf()),
        ('select', SelectKBest(chi2, k=5000)),
        ('rf', build_RF())
    ])
    
    #Optimization/model fitting
    print("starting training")
    
    use_random_search = False
    use_grid_search = False

    #Run selected model
    if use_random_search:
        best_pipeline = run_random_search(pipeline, X_train, y_train, param_dist)
    
    elif use_grid_search:
        best_pipeline = run_grid_search(pipeline, X_train, y_train, param_dist)
    
    else: 
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
        print("Fold scores:", cv_scores)
        print(f"CV accuracy: {cv_scores.mean():.4f}")
        print(f"CV std (SD): {cv_scores.std():.4f}")
        
        pipeline.fit(X_train, y_train)
        best_pipeline = pipeline
        

    #Store pipeline
    pipeline_file = OUTPUT_DIR / f"pipeline_{timestamp}.joblib"
    dump(best_pipeline, pipeline_file)
    print(f"Pipeline stored: {pipeline_file}")




# FUNCTIONS

# TF-IDF vectorizer
def build_tfidf(
                max_features = 10000,     # Max number of features (vocabulary size)
                ngram_range = (1,3),      # Range of n-grams (unigrams, bigrams, etc.)
                min_df = 3, #5            # Ignore terms that appear in fewer documents
                max_df = 0.8, #0.9        # Ignore terms that appear in too many documents
                sublinear_tf = False, #F  # Use log-scaled term frequency instead of raw counts
                binary = False,           # Use binary term occurrence (0/1) instead of counts  
                use_idf = True,           # Apply inverse document frequency weighting
                norm = 'l2',              # Normalization method ('l1', 'l2', or None)
                stop_words = None #"english" #None         # Remove stopwords (e.g. 'english') or keep all
                ):
    """
    Create a TF-IDF vectorizer
    """

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
        binary=binary,
        use_idf=use_idf,
        norm=norm,
        lowercase=False,   # Already handled in preprocessing
        stop_words=stop_words
    )

    return vectorizer


# Random Forest
def build_RF(
            n_estimators=200,        # Number of trees in the forest
            max_depth=None, #None    # Maximum depth of each tree
            max_features="sqrt",     # Number of features considered at each split
            min_samples_split=2,     # Minimum samples required to split a node
            min_samples_leaf=1,      # Minimum samples required at a leaf node
            bootstrap=True,          # Whether to use bootstrap sampling
            class_weight=None,       # Handle class imbalance ('balanced' or None)
            verbose=False,           # Print training progress
            n_jobs=-1,
            random_state=random_state
            ):
    """
    Create RandomForestClassifier
    """

    RF = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        bootstrap=bootstrap,
        class_weight=class_weight,
        verbose=verbose,
        n_jobs=n_jobs,
        random_state=random_state
    )

    return RF


#Random search
def run_random_search(pipeline, X_train, y_train, param_dist):

    print("\nStarting Randomized Search...\n")
    print("\nActive parameters in search:")
    for k in param_dist:
        print("-", k)

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=20,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=2,
        random_state=random_state
    )

    search.fit(X_train, y_train)

    print("\n=== RANDOM SEARCH DONE ===")
    print(f"Best score: {search.best_score_:.4f}")

    print("\nBest parameters:")
    for k, v in search.best_params_.items():
        print(f"{k}: {v}")

    return search.best_estimator_

#Grid search 
def run_grid_search(pipeline, X_train, y_train, param_dist):

    print("\nStarting Grid Search...\n")
    print("\nActive parameters in search:")
    for k in param_dist:
        print("-", k)

    search = GridSearchCV(
        pipeline,
        param_grid=param_dist,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=2
    )

    search.fit(X_train, y_train)

    print("\n=== GRID SEARCH DONE ===")
    print(f"Best score: {search.best_score_:.4f}")

    print("\nBest parameters:")
    for k, v in search.best_params_.items():
        print(f"{k}: {v}")

    return search.best_estimator_


#run main
if __name__ == "__main__":
    train_model()