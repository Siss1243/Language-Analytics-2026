# Assignment 03 — N-gram Language Model (Taylor Swift Lyrics)

This project builds a classic **count-based n-gram language model** in Python for text generation.

The model is trained on a filtered dataset of Taylor Swift lyrics and can generate new lyric-like text based on learned token patterns.

---

# Project Goal

The purpose of the project is to:

- preprocess song lyrics
- train an n-gram language model
- generate new text from a seed word or phrase
- inspect generation decisions using an insight/debug mode
- save trained models and generated outputs

The model is **not** intended to copy songs, but to generate new text inspired by corpus patterns.

---

# Project Structure

```text
assignment_03/
├── in/
│   └── TaylorSwift.csv
│
├── out/
│   ├── models/
│   │   └── saved .pkl models
│   │
│   └── generated/
│       └── generated_texts.csv
│
├── src/
│   ├── train.py
│   ├── generate.py
│   └── ngrammodel.py
│
└── README.md
```


# Method Summary

The project uses a traditional statistical language modelling approach.

Main steps:

Load and clean lyric data
Tokenize text into words and punctuation
Count n-grams from order 1 to n
Estimate conditional probabilities
Generate text by probabilistic next-token sampling
Use stupid backoff when context is unseen
Save models and generated outputs

# Command Line Usage

This project uses `argparse` in both `train.py` and `generate.py`.

---

## train.py Arguments

| Argument       | Type | Default              | Required | Description                                               |
| -------------- | ---- | -------------------- | -------- | --------------------------------------------------------- |
| `--input`      | Path | `in/TaylorSwift.csv` | No       | Path to input CSV dataset                                 |
| `--output`     | Path | Auto timestamp save  | No       | Optional custom output path for saved model               |
| `--model-name` | str  | `taylor`             | No       | Name used when saving model                               |
| `--ngram-size` | int  | `3`                  | No       | Size of n-grams (2 = bigram, 3 = trigram, etc.)           |
| `--min-words`  | int  | `20`                 | No       | Minimum number of words required for lyric inclusion      |
| `--min-chars`  | int  | `500`                | No       | Minimum number of characters required for lyric inclusion |
| `--max-chars`  | int  | `2825`               | No       | Maximum number of characters allowed for lyric inclusion  |

## generate Arguments 

| Argument          | Type | Default    | Required | Description                                             |
| ----------------- | ---- | ----------------------------------- | -------- | ------------------------------------------------------- |
| `--model`         | str  | `latest`                            | No       | Path to saved model file or newest available model      |
| `--seed`          | str  | None                                | Yes      | Seed word or phrase used to begin generation            |
| `--max-tokens`    | int  | `30`                                | No       | Number of new tokens generated after the seed           |
| `--top-k`         | int  | `None`                              | No       | Restrict sampling to the top-k most probable candidates |
| `--output-csv`    | Path | `out/generated/generated_texts.csv` | No       | CSV file where generated outputs are appended           |
| `--insight`       | flag | `False`                             | No       | Show step-by-step model decisions during generation     |
| `--insight-top-n` | int  | `5`                                 | No       | Number of candidate words shown in insight mode         |


## Example commands

| Purpose                      | Command                                                           |
| ---------------------------- | ----------------------------------------------------------------- |
| Train trigram model          | `python src/train.py --ngram-size 3`                              |
| Train bigram model           | `python src/train.py --ngram-size 2`                              |
| Generate text                | `python src/generate.py --seed love`                              |
| Generate longer text         | `python src/generate.py --seed love --max-tokens 50`              |
| Generate with top-k          | `python src/generate.py --seed love --top-k 5`                    |
| Generate with insight        | `python src/generate.py --seed love --insight`                    |
| Insight with more candidates | `python src/generate.py --seed love --insight --insight-top-n 10` |
| Use specific model file      | `python src/generate.py --model out/models/model.pkl --seed love` |
