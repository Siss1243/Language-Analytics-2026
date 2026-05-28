"""
Create a  train/test split from the original dataset.
"""

#IMPORTS
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


#PATHS
DATA_DIR = Path(__file__).parent.parent / "data"

#FUNCTION
def create_data_split(test_size=0.2, random_state=42):

    """
    Load dataset, split into test/train, and store as two datasets
    """

    #Load data
    data = pd.read_csv((DATA_DIR / "fake_real_news_train_data.csv"))

    #Split features and labels
    X = data["text"]
    y = data["label"]

    #Perform stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    #Build dataframes
    train_df = pd.DataFrame({"text": X_train, "label": y_train})
    test_df = pd.DataFrame({"text": X_test, "label": y_test})

    # Save to disk
    train_path = DATA_DIR / "train_split.csv"
    test_path = DATA_DIR / "test_split.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)



# MAIN
if __name__ == "__main__":
    create_data_split()
