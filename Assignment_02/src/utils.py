"""
Utility functions for Assignment 2
"""

#IMPORTS
import pandas as pd
import re

#FUNCTIONS

#Preprocess data
def preprocess_text(data):
    """
    Preprocess text data and optionally filter by length.
    
    """

    # Lowercase 
    clean_text = data.str.lower() 
    
    # Replace newlines and tabs with space 
    clean_text = clean_text.str.replace(r'\s+', ' ', regex=True) 
    
    # Remove special characters (keep letters and numbers) 
    clean_text = clean_text.str.replace(r'[^a-z0-9 ]', '', regex=True) 
    
    # Strip leading/trailing whitespace 
    clean_text = clean_text.str.strip()

    """
    # Length filtering (ONLY if specified)
    if min_len is not None or max_len is not None:
        lengths = clean_text.str.len()

        mask = pd.Series(True, index=clean_text.index)

        if min_len is not None:
            mask &= lengths >= min_len

        if max_len is not None:
            mask &= lengths <= max_len

        clean_text = clean_text[mask]

        if y is not None:
            y = y[mask]

    if y is not None:
        return clean_text, y
    """

    return clean_text


#Basic data inspection
def basic_inspection(data):
    
    print("\n=== HEAD ===")
    print(data.head())

    print("\n=== INFO ===")
    print(data.info())

    print("\n=== MISSING VALUES ===")
    print(data.isnull().sum())

    print("\n=== CLASS DISTRIBUTION ===")
    print(data["label"].value_counts())

    print("\n=== TEXT LENGTH STATS (ALL) ===")
    print(data["text"].str.len().describe())

    # Additional stats per class
    print("\n=== TEXT LENGTH STATS BY LABEL ===")
    for label in data["label"].unique():
        lengths = data.loc[data["label"] == label, "text"].str.len()
        print(f"\nLabel: {label}")
        print(f"Count: {len(lengths)}")
        print(f"Mean length: {lengths.mean():.2f}")
        print(f"Std deviation: {lengths.std():.2f}")
        print(f"Min length: {lengths.min()}")
        print(f"Max length: {lengths.max()}")


#Print pipline used
def print_pipeline_params(pipeline):
    """
    Pretty-print parameters for each step in a sklearn pipeline
    """

    for step_name, step in pipeline.named_steps.items():
        print(f"\n=== {step_name.upper()} PARAMETERS ===")
        
        params = step.get_params()
        
        for key, value in params.items():
            print(f"{key}: {value}")



