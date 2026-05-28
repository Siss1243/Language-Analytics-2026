

#----------Imports----------#
import pandas as pd
import numpy as np
import os
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import nltk
import matplotlib.pyplot as plt

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

#-----------Main-----------#

def main():

    # Paths
    data_path = os.path.join("data", "narradetect.csv")
    output_path = os.path.join("output")
    
    # Load data and subset data
    df = load_data(data_path)
    
    print("[INFO] main() finished loading data")

    df_aph = subset_genre(df, "APHORISM")
    df_lit = subset_genre(df, "LITSTUDY")

    print("[INFO] Finished subsetting two target genres.")

    # Preprocess texts for each group
    df_aph_clean = text_preprocessing(df_aph)
    df_lit_clean = text_preprocessing(df_lit)

    print("[INFO] Preprocessing complete for both groups.")

    #print_text_example(df_aph_clean, "APHORISM", 5)
    #print_text_example(df_lit_clean, "LITSTUDY", 5)

    # Compute overall dataa description
    df_aph_clean, aph_stats = data_descriptive(df_aph_clean)
    df_lit_clean, lit_stats = data_descriptive(df_lit_clean)

    print("[INFO] Descriptive data computed")

    # Compute Lexical variation
    aph_sentence_lengths, aph_lex = lexical_variation(df_aph_clean)
    lit_sentence_lengths, lit_lex = lexical_variation(df_lit_clean)  

    print("[INFO] Lexical variation computed")

    #Print and save results: 
    results = {
    "APHORISM": {**aph_stats, **aph_lex},
    "LITSTUDY": {**lit_stats, **lit_lex}}

    results_df = (pd.DataFrame(results).T).round(2)

    results_df.to_csv(os.path.join(output_path, "stats_summary.csv"))

    print("\n===== FINAL RESULTS TABLE =====")
    print(results_df)

    #Compute and save figures
    plot_sentence_length_boxplot(aph_sentence_lengths, lit_sentence_lengths, output_path)

    print("[INFO] Analysis complete")

#----------Functions----------#


#Function: load data
def load_data(data_path):
    """
    Loads the CSV dataset
    """

    df = pd.read_csv(data_path, index_col=0)

    return df

#Function: Subset data
def subset_genre(df, genre_col):
    """
    Subset the dataframe to a specific genre
    """
    
    if genre_col not in df["genre"].unique():
        print(f"Genre '{genre_col}' not found in dataset")
    
    subset = df[df["genre"] == genre_col].copy()

    return subset

# Function: Preprocess data
def text_preprocessing(df_subset):
    """
    Preprocesses the texts in a subset dataframe:
    - Keep original texts
    - Lowercases text
    - Tokenizes using NLTK (both word and sentence)
    - Removes punctuation tokens
    """

    df_subset_preprocessed = df_subset[["text"]].copy()

    #Preprocessing loop
    processed_tokens = []

    for text in df_subset_preprocessed["text"]:

        #Preprocess
        text = text.lower() #Lowercasing
        tokens = word_tokenize(text) #Tokanize
        
        
        extra_punct = {"’", "‘", "“", "”", "–", "—", "…", "``", "''", "´"}
        all_punct = set(string.punctuation).union(extra_punct)
        tokens = [t for t in tokens if t not in all_punct]

        tokens = [t for t in tokens if t not in string.punctuation] #Remove puncturation

        processed_tokens.append(tokens)

    #Add cleaned tokens to dataset
    df_subset_preprocessed["tokens"] = processed_tokens
    df_subset_preprocessed["sentences"] = df_subset_preprocessed["text"].apply(sent_tokenize)
    
    return df_subset_preprocessed


#Function for overall data description
def data_descriptive(df_subset_preprocessed):
    """
    Computes the overall descriptive metrics for one subgroup
    - Number of texts
    - Average length of text's (in tokens)
    - Total number of tokens
    - Total number of types
    """
    
    #Number of texts
    n_texts = len(df_subset_preprocessed)
    
    #Length of each text
    df_subset_preprocessed["n_tokens"] = df_subset_preprocessed["tokens"].apply(len)
    
    #Number of tokens
    total_tokens = df_subset_preprocessed["n_tokens"].sum()    
    
    #Avg. length of texts
    avg_length = df_subset_preprocessed["n_tokens"].mean()
    std_length = df_subset_preprocessed["n_tokens"].std()

    #Number of types
    all_tokens = [t for tokens in df_subset_preprocessed["tokens"] 
                 for t in tokens]
    total_types = len(set(all_tokens))

    return df_subset_preprocessed, {
        "n_texts": n_texts,
        "avg_length": avg_length,
        "std_length": std_length,
        "total_tokens": total_tokens,
        "total_types": total_types}



#Function for lexical variation

def lexical_variation(df_subset_preprocessed):
    """
    Computes lexical variation metrics for one subgroup:
    - Mean and std of TTR (per text)
    - Mean and std of function word proportion (per text)
    - Average sentence length
    - Std. sentence length
    """

    # Lists for per-text metrics
    ttr_list = []
    funcword_list = []

    stop_words = set(stopwords.words("english"))

    # ---------- PER-TEXT METRICS ----------
    for tokens in df_subset_preprocessed["tokens"]:

        if len(tokens) == 0:
            continue

        # TTR per text
        types = set(tokens)
        ttr = len(types) / len(tokens)
        ttr_list.append(ttr)

        # Function word proportion per text
        func_words = [t for t in tokens if t in stop_words]
        func_ratio = (len(func_words) / len(tokens)) * 100
        funcword_list.append(func_ratio)

    # Aggregate lexical metrics
    mean_ttr = np.mean(ttr_list)
    std_ttr = np.std(ttr_list, ddof=1)

    mean_func = np.mean(funcword_list)
    std_func = np.std(funcword_list, ddof=1)


    # ---------- SENTENCE LENGTH ----------
    sentence_lengths = []

    extra_punct = {"’", "‘", "“", "”", "–", "—", "…", "``", "''", "´"}
    all_punct = set(string.punctuation).union(extra_punct)

    for text in df_subset_preprocessed["text"]:
        sentences = sent_tokenize(text)

        for sentence in sentences:
            tokens = word_tokenize(sentence.lower())
            tokens = [t for t in tokens if t not in all_punct]
            sentence_lengths.append(len(tokens))

    avg_sentence_length = np.mean(sentence_lengths)
    std_sentence_length = np.std(sentence_lengths, ddof=1)


    # ---------- RETURN ----------
    return sentence_lengths, {
        "mean_TTR": mean_ttr,
        "std_TTR": std_ttr,
        "mean_func_words": mean_func,
        "std_func_words": std_func,
        "avg_sentence_length": avg_sentence_length,
        "std_sentence_length": std_sentence_length}



#Function: Plotting figures
def plot_sentence_length_boxplot(aph_lengths, lit_lengths, output_path):
    
    plt.figure(figsize=(7,5), dpi=300)

    data = [aph_lengths, lit_lengths]

    box = plt.boxplot(
        data,
        labels=["APHORISM", "LITSTUDY"],
        patch_artist=True,
        showmeans=True,
        meanline=True,
        widths=0.6)

    # Clean style
    colors = ["#4C72B0", "#55A868"]
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    plt.ylabel("Sentence length (tokens)", fontsize=11)
    plt.title("Distribution of Sentence Length Across Genres", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # Remove top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "sentence_length_boxplot.png"), dpi=300)
    plt.close()

#Function: Example to understand data
def print_text_example(df, label, idx):
    """
    Prints one example text in:
    - Original form
    - Word-tokenized form
    - Sentence-tokenized form
    """
    print(f"\n===== EXAMPLE FROM {label} (index {idx}) =====")

    original = df["text"].iloc[idx]
    tokens = df["tokens"].iloc[idx]
    sentences = df["sentences"].iloc[idx]

    print("\n--- Original Text ---")
    print(original)

    print("\n--- Word Tokens ---")
    print(tokens)

    print("\n--- Sentence Tokens ---")
    for i, s in enumerate(sentences):
        print(f"[Sentence {i+1}]: {s}")

if __name__ == "__main__":
    main()