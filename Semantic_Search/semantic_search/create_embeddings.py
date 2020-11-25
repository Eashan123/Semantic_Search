from sentence_transformers import SentenceTransformer
from torch.cuda import is_available

import time
import argparse
import joblib
import pandas as pd
import re


def create_embeddings(
    corpus_file_path: str,
    sbert_name: str = "distilbert-base-nli-mean-tokens",
    batch_size: int = 64,
    show_progress_bar: bool = True,
    num_workers: int = 2,
):
    model = SentenceTransformer(sbert_name)

    device = "cuda" if is_available() else "cpu"
    print(f"Using: {device}")

    df = pd.read_csv(corpus_file_path)

    df["descriptions"] = df["descriptions"].apply(
        lambda x: re.sub(r"[^a-zA-Z]", " ", str(x))
    )

    corpus = df["descriptions"].values.tolist()

    corpus = [i for i in corpus if len(i.split()) > 3]

    new_df = pd.DataFrame({"descriptions": corpus})
    new_df.to_csv("data/processed_data.csv", index=False)

    corpus_embeddings = model.encode(
        sentences=corpus,
        batch_size=batch_size,
        show_progress_bar=True,
        device=device,
        num_workers=num_workers,
    )

    return corpus_embeddings


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        "Script for crating semantic search engine embeddings\n"
    )

    arg_parser.add_argument(
        "--corpus_path",
        help="String, Path to the txt file containing corpus",
        required=True,
    )

    arg_parser.add_argument(
        "--out_path",
        help="String, Output path where the embeddings "
        "will be stored as a pickle file",
        required=True,
    )

    arg_parser.add_argument(
        "--batch_size", help="Integer, Batch size for embeddings", required=True
    )

    arg_parser.add_argument(
        "--parallel_jobs",
        help="Integer, Number of parallel workers for DataLoader",
        required=True,
    )

    args = arg_parser.parse_args()

    embeddings = create_embeddings(
        corpus_file_path=args.corpus_path,
        batch_size=int(args.batch_size),
        show_progress_bar=True,
        num_workers=int(args.parallel_jobs),
    )
    joblib.dump(value=embeddings, filename=args.out_path, compress=True)
