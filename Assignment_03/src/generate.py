"""
generate.py

Loads a trained n-gram language model, generates text
from a seed string, and appends the result to a CSV file.
"""

# IMPORTS
import argparse
import csv
from datetime import datetime
from pathlib import Path

from ngrammodel import NgramModel


# PATHS
PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATED_DIR = PROJECT_ROOT / "out" / "generated"
DEFAULT_OUTPUT_CSV = GENERATED_DIR / "generated_texts.csv"

#FUNCTIONS: MAIN

def generate_text(args):
    """
    Run the text generation pipeline.
    """

    model = NgramModel.load(args.model)

    if args.insight:
        generated_text, insights = model.generate_with_insight(
            seed=args.seed,
            max_tokens=args.max_tokens,
            top_k=args.top_k,
            insight_top_n=args.insight_top_n,
        )
    else:
        generated_text = model.generate(
            seed=args.seed,
            max_tokens=args.max_tokens,
            top_k=args.top_k,
        )
        insights = None

    print("\n--- GENERATED TEXT ---\n")
    print(generated_text)

    if insights is not None:
        print("\n--- INSIGHT ---\n")

        for step in insights:
            print(f"Step {step['step']}")
            print(f"History given: {step['history_given']}")
            print(f"History used: {step['history_used']}")
            print(f"Used order: {step['used_order']}")
            print(f"Backoff used: {step['backoff_used']}")
            print(f"Chosen word: {step['chosen_word']}")
            print(f"Chosen probability: {step['chosen_probability']:.4f}")
            print(f"Chosen rank: {step['chosen_rank']}")

            print("Top candidates:")
            for rank, (word, prob) in enumerate(step["top_candidates"], start=1):
                print(f"  {rank}. {word} ({prob:.4f})")

            print()

    save_generation_to_csv(
        csv_path=args.output_csv,
        model_used=args.model,
        seed=args.seed,
        generated_text=generated_text,
    )

    print(f"\nGeneration saved to: {args.output_csv}")

# FUNCTIONS: SUPPORT
def save_generation_to_csv(csv_path, model_used, seed, generated_text):
    """
    Append one generated text example to a CSV file.

    Columns:
    timestamp, model_used, seed, generated_text
    """

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = csv_path.exists()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(csv_path, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(["timestamp", "model_used", "seed", "generated_text"])

        writer.writerow([timestamp, model_used, seed, generated_text])



# FUNCTIONS: PARSER
def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for text generation.
    """

    parser = argparse.ArgumentParser(
        description="Generate text using a trained n-gram language model."
    )

    parser.add_argument(
        "--model",
        type=str,
        default="latest",
        help="Path to a saved model file, or 'latest' to load the newest model.",
    )

    parser.add_argument(
        "--seed",
        type=str,
        required=True,
        help="Seed text used to start generation.",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=30,
        help="Number of new tokens to generate after the seed.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Optional top-k sampling. Limits sampling to the k most probable next words.",
    )

    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Path to CSV file where generated texts will be appended.",
    )

    parser.add_argument(
        "--insight",
        action="store_true",
        help="Print step-by-step insight into the model's word choices.",
    )

    parser.add_argument(
        "--insight-top-n",
        type=int,
        default=5,
        help="Number of top candidate words to show for each generation step.",
    )

    args = parser.parse_args()

    if args.max_tokens < 1:
        raise ValueError("--max-tokens must be at least 1.")

    if args.top_k is not None and args.top_k < 1:
        raise ValueError("--top-k must be at least 1.")

    return args


#CALL FOR MAIN
if __name__ == "__main__":
    args = parse_args()
    generate_text(args)