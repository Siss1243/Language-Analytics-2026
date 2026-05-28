"""
ngrammodel.py

Defines the NgramModel class used for preprocessing, training,
text generation, and saving/loading an n-gram language model.
"""

# IMPORTS
import glob
import os
import random
import re
from collections import Counter, defaultdict
from datetime import datetime

import joblib

class NgramModel:
    """
    A simple n-gram language model for text generation.
    """

    def __init__(self, name, ngram_size):
        """
        Initialize the model.
        """

        if ngram_size < 2:
            raise ValueError("ngram_size must be at least 2.")

        self.name = name
        self.n_gram_size = ngram_size

        # Vocabulary learned during training
        self.vocab = set()

        # Count storage for all orders from 1 to n
        self.ngram_counts_by_order = {
            order: Counter() for order in range(1, self.n_gram_size + 1)}

        self.context_counts_by_order = {
            order: Counter() for order in range(2, self.n_gram_size + 1)}

        self.context_to_next_by_order = {
            order: defaultdict(Counter) for order in range(2, self.n_gram_size + 1)}

        # Unigram counts for fallback
        self.unigram_counts = Counter()
        self.total_unigrams = 0

        # Status
        self.trained = False
        self.example_shown = False

    def preprocess_text(self, text):
        """
        Clean and tokenize a single text string.
        """

        original_text = text

        #Lowercase
        text = text.lower()

        #Normalize unusual whitespaces
        text = text.replace("\n", " ")

        #Collapse repeated whitespace
        text = re.sub(r"\s+", " ", text).strip()

        #Tokenize (keep letters, digits, apostrophes, periods)
        tokens = re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?|\.", text)
        #Without periods: tokens = re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text)

        # Show only one example
        if not self.example_shown:
            print("\n--- PREPROCESSING EXAMPLE ---")
            print("\nBefore preprocessing:")
            print(original_text[:300])

            print("\nAfter preprocessing:")
            print(tokens[:40])

            self.example_shown = True

        return tokens

    def train(self, lyrics):
        """
        Train the n-gram model on a list of lyrics.
        Builds counts for all orders from 1 to n for stupid backoff.
        """

        for text in lyrics:
            tokens = self.preprocess_text(text)

            if not tokens:
                continue

            # Add tokens to vocabulary
            self.vocab.update(tokens)

            # Unigram counts
            for token in tokens:
                self.ngram_counts_by_order[1][(token,)] += 1
                self.unigram_counts[token] += 1
                self.total_unigrams += 1

            # Higher-order n-grams
            max_order = min(self.n_gram_size, len(tokens))

            for order in range(2, max_order + 1):
                for i in range(len(tokens) - order + 1):
                    ngram = tuple(tokens[i:i + order])

                    history = ngram[:-1]
                    continuation = ngram[-1]

                    self.ngram_counts_by_order[order][ngram] += 1
                    self.context_counts_by_order[order][history] += 1
                    self.context_to_next_by_order[order][history][continuation] += 1

        self.trained = True

        print("\nTraining complete.")
        print(f"Vocabulary size: {len(self.vocab)}")
        print(f"Total unigrams: {self.total_unigrams}")

        for order in range(1, self.n_gram_size + 1):
            print(
                f"Distinct {order}-grams: {len(self.ngram_counts_by_order[order])}")

    
    def conditional_probability_distribution(self, history):
        """
        Return the conditional probability distribution
        for the given history.
        """

        if not self.trained:
            raise ValueError("Model has not been trained yet.")

        history = tuple(history)

        order = len(history) + 1

        if order < 2 or order > self.n_gram_size:
            return {}

        if history not in self.context_to_next_by_order[order]:
            return {}

        continuation_counts = self.context_to_next_by_order[order][history]

        total = sum(continuation_counts.values())

        probabilities = {}

        for word, count in continuation_counts.items():
            probabilities[word] = count / total

        return probabilities
    
    def backoff_probability_distribution(self, history):
        """
        Return a probability distribution using stupid backoff.

        The model first tries the full history. If that history is unseen,
        it backs off to progressively shorter suffix histories. If no
        history is found, it falls back to the unigram distribution.
        """

        if not self.trained:
            raise ValueError("Model has not been trained yet.")

        history = tuple(history)

        # Try full history, then shorter suffixes
        for start_index in range(len(history)):
            shorter_history = history[start_index:]
            distribution = self.conditional_probability_distribution(shorter_history)

            if distribution:
                return distribution

        # Final fallback: unigram distribution
        if self.total_unigrams == 0:
            return {}

        unigram_probabilities = {}

        for word, count in self.unigram_counts.items():
            unigram_probabilities[word] = count / self.total_unigrams

        return unigram_probabilities
    
    def backoff_distribution_with_insight(self, history):
        """
        Return a probability distribution and insight information.

        Returns a dictionary with:
        - distribution: probability distribution for next word
        - original_history: full history provided
        - used_history: history that was actually used
        - used_order: n-gram order used
        - backoff_used: whether backoff was needed
        """

        if not self.trained:
            raise ValueError("Model has not been trained yet.")

        history = tuple(history)
        original_history = history

        # Try full history, then shorter suffixes
        for start_index in range(len(history)):
            shorter_history = history[start_index:]
            distribution = self.conditional_probability_distribution(shorter_history)

            if distribution:
                return {
                    "distribution": distribution,
                    "original_history": original_history,
                    "used_history": shorter_history,
                    "used_order": len(shorter_history) + 1,
                    "backoff_used": shorter_history != original_history,
                }

        # Final fallback: unigram distribution
        if self.total_unigrams == 0:
            return {
                "distribution": {},
                "original_history": original_history,
                "used_history": (),
                "used_order": 1,
                "backoff_used": True,
            }

        unigram_probabilities = {}

        for word, count in self.unigram_counts.items():
            unigram_probabilities[word] = count / self.total_unigrams

        return {
            "distribution": unigram_probabilities,
            "original_history": original_history,
            "used_history": (),
            "used_order": 1,
            "backoff_used": True,
        }


    def sample_next_word(self, history, top_k=None):
        """
        Sample one next word using backoff and optional top-k sampling.
        """

        distribution = self.backoff_probability_distribution(history)

        if not distribution:
            return None

        ranked_items = sorted(
            distribution.items(),
            key=lambda item: item[1],
            reverse=True
        )

        if top_k is not None:
            if top_k < 1:
                raise ValueError("top_k must be at least 1.")
            ranked_items = ranked_items[:top_k]

        words = [word for word, _ in ranked_items]
        probabilities = [prob for _, prob in ranked_items]

        return random.choices(words, weights=probabilities, k=1)[0]

    def sample_next_word_with_insight(self, history, top_k=None):
        """
        Sample one next word and return insight information.
        """

        info = self.backoff_distribution_with_insight(history)
        distribution = info["distribution"]

        if not distribution:
            return None

        ranked_items = sorted(
            distribution.items(),
            key=lambda item: item[1],
            reverse=True
        )

        if top_k is not None:
            if top_k < 1:
                raise ValueError("top_k must be at least 1.")
            ranked_items = ranked_items[:top_k]

        words = [word for word, _ in ranked_items]
        probabilities = [prob for _, prob in ranked_items]

        chosen_word = random.choices(words, weights=probabilities, k=1)[0]

        chosen_probability = None
        chosen_rank = None

        for rank, (word, prob) in enumerate(ranked_items, start=1):
            if word == chosen_word:
                chosen_probability = prob
                chosen_rank = rank
                break

        return {
            "chosen_word": chosen_word,
            "chosen_probability": chosen_probability,
            "chosen_rank": chosen_rank,
            "ranked_candidates": ranked_items,
            "original_history": info["original_history"],
            "used_history": info["used_history"],
            "used_order": info["used_order"],
            "backoff_used": info["backoff_used"],
        }

    def generate(self, seed, max_tokens, top_k=None):
        """
        Generate text from the trained model using optional top-k sampling.
        Supports short seeds through stupid backoff.
        """

        if not self.trained:
            raise ValueError("Model has not been trained yet.")

        seed_tokens = self.preprocess_text(seed)

        if len(seed_tokens) == 0:
            raise ValueError("Seed must contain at least one valid token.")

        history_size = self.n_gram_size - 1

        generated_tokens = seed_tokens.copy()

        while len(generated_tokens) < len(seed_tokens) + max_tokens:

            # Use as much recent context as available
            current_history = tuple(generated_tokens[-history_size:])

            next_word = self.sample_next_word(
                current_history,
                top_k=top_k
            )

            if next_word is None:
                break

            generated_tokens.append(next_word)

        generated_text = " ".join(generated_tokens)

        # Remove spaces before punctuation
        generated_text = re.sub(r"\s+([.])", r"\1", generated_text)

        return generated_text
    
    def generate_with_insight(self, seed, max_tokens, top_k=None, insight_top_n=5):
        """
        Generate text and collect insight for each generation step.
        """

        if not self.trained:
            raise ValueError("Model has not been trained yet.")

        seed_tokens = self.preprocess_text(seed)

        if len(seed_tokens) == 0:
            raise ValueError("Seed must contain at least one valid token.")

        history_size = self.n_gram_size - 1
        generated_tokens = seed_tokens.copy()
        insights = []

        while len(generated_tokens) < len(seed_tokens) + max_tokens:

            current_history = tuple(generated_tokens[-history_size:])

            step_info = self.sample_next_word_with_insight(
                current_history,
                top_k=top_k
            )

            if step_info is None:
                break

            next_word = step_info["chosen_word"]
            generated_tokens.append(next_word)

            insights.append({
                "step": len(insights) + 1,
                "history_given": step_info["original_history"],
                "history_used": step_info["used_history"],
                "used_order": step_info["used_order"],
                "backoff_used": step_info["backoff_used"],
                "chosen_word": step_info["chosen_word"],
                "chosen_probability": step_info["chosen_probability"],
                "chosen_rank": step_info["chosen_rank"],
                "top_candidates": step_info["ranked_candidates"][:insight_top_n],
            })

        generated_text = " ".join(generated_tokens)

        # Remove spaces before punctuation
        generated_text = re.sub(r"\s+([.])", r"\1", generated_text)

        return generated_text, insights


    def save(self, model_path=None):
        """
        Save trained model to disk.
        """

        if not self.trained:
            raise ValueError("Cannot save an untrained model.")

        if model_path is None:
            os.makedirs("out/models", exist_ok=True)

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
            filename = f"{self.name}_n{self.n_gram_size}_{timestamp}.pkl"
            model_path = os.path.join("out/models", filename)

        else:
            directory = os.path.dirname(model_path)
            if directory:
                os.makedirs(directory, exist_ok=True)

        joblib.dump(self, model_path)

        print(f"Model saved to: {model_path}")


    @classmethod
    def load(cls, model_path="latest"):
        """
        Load a saved model from disk.
        Default loads the newest saved model.
        """

        if model_path == "latest":

            model_files = glob.glob("out/models/*.pkl")

            if not model_files:
                raise FileNotFoundError("No saved models found in out/models/")

            newest_file = max(model_files, key=os.path.getmtime)

            print(f"Loading latest model: {newest_file}")

            return joblib.load(newest_file)

        else:
            print(f"Loading model: {model_path}")
            return joblib.load(model_path)