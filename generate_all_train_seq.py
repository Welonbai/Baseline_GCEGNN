import argparse
import os
import pickle
from typing import List


def load_train(train_path: str):
    with open(train_path, "rb") as f:
        sequences, labels = pickle.load(f)
    if len(sequences) != len(labels):
        raise ValueError("Mismatch between number of sequences and labels")
    return sequences, labels


def rebuild_sessions(sequences: List[List[int]], labels: List[int]):
    restored = []
    for seq, label in zip(sequences, labels):
        if not seq:
            continue
        restored.append(seq + [label])
    return restored


def save_sessions(sessions, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(sessions, f, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    parser = argparse.ArgumentParser(description="Rebuild all_train_seq.txt from train.txt")
    parser.add_argument("--dataset", required=True, help="Dataset name under datasets/")
    parser.add_argument("--train", default="train.txt", help="Train pickle filename")
    parser.add_argument("--output", default="all_train_seq.txt", help="Output filename")
    args = parser.parse_args()

    dataset_dir = os.path.join("datasets", args.dataset)
    train_path = os.path.join(dataset_dir, args.train)
    output_path = os.path.join(dataset_dir, args.output)

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train pickle not found: {train_path}")

    sequences, labels = load_train(train_path)
    sessions = rebuild_sessions(sequences, labels)
    if not sessions:
        raise ValueError("No sessions were rebuilt; check the train file format")

    save_sessions(sessions, output_path)
    print(f"Saved {len(sessions)} sessions to {output_path}")


if __name__ == "__main__":
    main()
