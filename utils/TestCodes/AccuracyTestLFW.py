import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from ..ModelLoader.model_pytorch import load_model


def find_lfw_root(data_dir: Path) -> Path:
    root = data_dir / "lfw-deepfunneled"
    if not root.exists():
        raise FileNotFoundError(f"Could not find lfw-deepfunneled under {data_dir}")

    nested = root / "lfw-deepfunneled"
    if nested.exists() and any(p.is_dir() for p in nested.iterdir()):
        return nested
    return root


def parse_pairs_csv(pairs_path: Path):
    pairs = []
    with pairs_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row or all(not (col or "").strip() for col in row):
                continue
            row = [col.strip() for col in row if col is not None]
            row = [col for col in row if col != ""]
            if len(row) == 3:
                name, num1, num2 = row
                pairs.append((name, int(num1), name, int(num2), 1))
            elif len(row) == 4:
                name1, num1, name2, num2 = row
                pairs.append((name1, int(num1), name2, int(num2), 0))
            else:
                print(f"Skipping malformed row in {pairs_path}: {row}")
    return pairs


def image_file_for(name: str, number: int, lfw_root: Path) -> Path:
    filename = f"{name}_{int(number):04d}.jpg"
    return lfw_root / name / filename


def load_image(path: Path, transform):
    if not path.exists():
        raise FileNotFoundError(f"Missing image: {path}")
    image = Image.open(path).convert("RGB")
    return transform(image)


def build_embeddings(model, image_paths, device, transform, batch_size=16):
    embeddings = {}
    unique_paths = list(dict.fromkeys(image_paths))
    for start in range(0, len(unique_paths), batch_size):
        batch_paths = unique_paths[start : start + batch_size]
        tensors = [load_image(path, transform) for path in batch_paths]
        batch = torch.stack(tensors, dim=0).to(device)
        with torch.no_grad():
            output = model(batch)
            output = F.normalize(output, p=2, dim=1).cpu()
        for path, emb in zip(batch_paths, output):
            embeddings[path] = emb
    return embeddings


def evaluate_pairs(similarities, labels, threshold):
    predictions = [sim >= threshold for sim in similarities]
    correct = sum(int(pred == label) for pred, label in zip(predictions, labels))
    accuracy = correct / len(labels) if labels else 0.0

    positive_total = sum(int(label == 1) for label in labels)
    negative_total = sum(int(label == 0) for label in labels)
    positive_correct = sum(int(pred and label == 1) for pred, label in zip(predictions, labels))
    negative_correct = sum(int((not pred) and label == 0) for pred, label in zip(predictions, labels))

    return {
        "accuracy": accuracy,
        "positive_accuracy": positive_correct / positive_total if positive_total else 0.0,
        "negative_accuracy": negative_correct / negative_total if negative_total else 0.0,
        "positive_total": positive_total,
        "negative_total": negative_total,
    }


def find_best_threshold(similarities, labels, num_steps=200):
    min_sim = min(similarities)
    max_sim = max(similarities)
    best = {
        "threshold": 0.0,
        "accuracy": 0.0,
    }
    for threshold in np.linspace(min_sim, max_sim, num_steps):
        metrics = evaluate_pairs(similarities, labels, threshold)
        if metrics["accuracy"] > best["accuracy"]:
            best["accuracy"] = metrics["accuracy"]
            best["threshold"] = threshold
    return best


def main():
    parser = argparse.ArgumentParser(description="Evaluate LFW verification accuracy.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(r"C:\Users\kuoro\Documents\GitHub\FaceRecognitionCPP\data\data_LFW"),
        help="Root folder for LFW files.",
    )
    parser.add_argument(
        "--pairs-file",
        type=Path,
        default=r"C:\Users\kuoro\Documents\GitHub\FaceRecognitionCPP\data\data_LFW\pairs.csv",
        help="CSV file with pair definitions. Default: pairs.csv in the data directory.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "models" / "model_weights.pt",
        help="Path to the PyTorch model weights file.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Cosine similarity threshold for verification.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for embedding extraction.",
    )
    args = parser.parse_args()

    if args.pairs_file is None:
        args.pairs_file = args.data_dir / "pairs.csv"

    print(f"Using data dir: {args.data_dir}")
    print(f"Using pair file: {args.pairs_file}")
    print(f"Using weights: {args.weights}")
    print(f"Using threshold: {args.threshold}")

    if not args.data_dir.exists():
        raise FileNotFoundError(f"LFW data directory not found: {args.data_dir}")
    if not args.pairs_file.exists():
        raise FileNotFoundError(f"Pairs file not found: {args.pairs_file}")
    if not args.weights.exists():
        raise FileNotFoundError(f"Model weights not found: {args.weights}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(str(args.weights), device)

    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
    ])

    lfw_root = find_lfw_root(args.data_dir)
    pairs = parse_pairs_csv(args.pairs_file)
    print(f"Loaded {len(pairs)} image pairs.")

    image_paths = []
    for name1, num1, name2, num2, _ in pairs:
        image_paths.append(image_file_for(name1, num1, lfw_root))
        image_paths.append(image_file_for(name2, num2, lfw_root))

    embeddings = build_embeddings(model, image_paths, device, transform, batch_size=args.batch_size)
    similarities = []
    labels = []
    missing = 0
    for name1, num1, name2, num2, label in pairs:
        path1 = image_file_for(name1, num1, lfw_root)
        path2 = image_file_for(name2, num2, lfw_root)
        emb1 = embeddings.get(path1)
        emb2 = embeddings.get(path2)
        if emb1 is None or emb2 is None:
            missing += 1
            continue
        sim = float(torch.matmul(emb1, emb2).item())
        similarities.append(sim)
        labels.append(label)

    if missing:
        print(f"Warning: skipped {missing} pairs because some images were missing.")

    metrics = evaluate_pairs(similarities, labels, args.threshold)
    best = find_best_threshold(similarities, labels)

    print("\nResults:")
    print(f"Total pairs: {len(labels)}")
    print(f"Positive pairs: {metrics['positive_total']}")
    print(f"Negative pairs: {metrics['negative_total']}")
    print(f"Accuracy @ threshold {args.threshold:.3f}: {metrics['accuracy'] * 100:.2f}%")
    print(f"Positive accuracy: {metrics['positive_accuracy'] * 100:.2f}%")
    print(f"Negative accuracy: {metrics['negative_accuracy'] * 100:.2f}%")
    print(f"Best threshold: {best['threshold']:.4f}")
    print(f"Best accuracy: {best['accuracy'] * 100:.2f}%")


if __name__ == "__main__":
    main()
