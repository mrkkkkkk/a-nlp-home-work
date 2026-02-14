import argparse
import json
import os
import re
from typing import Iterable, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertTokenizerFast


class PairDataset(Dataset):
    def __init__(self, records: List[dict]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx]
        return record["text1"], record["text2"]


def normalize_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def apply_normalization(records: List[dict]) -> List[dict]:
    normalized = []
    for record in records:
        new_record = dict(record)
        new_record["text1"] = normalize_text(record["text1"])
        new_record["text2"] = normalize_text(record["text2"])
        normalized.append(new_record)
    return normalized


def read_jsonl(path: str) -> List[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl(path: str, records: Iterable[dict]) -> None:
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_collate_fn(tokenizer, max_length: int):
    def collate_fn(batch):
        texts1 = [item[0] for item in batch]
        texts2 = [item[1] for item in batch]
        inputs = tokenizer(
            texts1,
            texts2,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return inputs

    return collate_fn


def predict_probs(
    model,
    dataloader: DataLoader,
    device: torch.device,
    progress_desc: Optional[str] = None,
) -> List[float]:
    model.eval()
    all_probs = []
    iterator = (
        tqdm(dataloader, desc=progress_desc, leave=False)
        if progress_desc
        else dataloader
    )
    with torch.no_grad():
        for batch in iterator:
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()
            all_probs.extend(probs)
    return all_probs


def read_metrics(path: str) -> dict:
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def resolve_model_dir(
    model_dir_arg: Optional[str],
    output_dir: str,
    opt_metric: Optional[str],
) -> str:
    if model_dir_arg:
        candidate = os.path.expanduser(model_dir_arg)
        if os.path.isdir(candidate):
            return candidate
        cwd = os.getcwd()
        if not os.path.isabs(candidate):
            rel_candidate = os.path.join(cwd, candidate)
            if os.path.isdir(rel_candidate):
                return rel_candidate
        else:
            rel_candidate = os.path.join(cwd, candidate.lstrip(os.sep))
            if os.path.isdir(rel_candidate):
                return rel_candidate
        raise FileNotFoundError(
            "Provided --model_dir not found. "
            f"Got: {model_dir_arg}. "
            f"Use a valid local path, e.g. {os.path.join(cwd, 'outputs', 'best_acc')}."
        )

    candidates = []
    if opt_metric:
        candidates.append(os.path.join(output_dir, f"best_{opt_metric}"))
    candidates.extend(
        [
            os.path.join(output_dir, "best_acc"),
            os.path.join(output_dir, "best_f1"),
            os.path.join(output_dir, "best"),
        ]
    )
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(
        "No trained model directory found. "
        "Provide --model_dir or ensure outputs/best_acc exists."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", default="data/test.jsonl")
    parser.add_argument("--result_path", default="result.jsonl")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--metrics_path", default=None)
    parser.add_argument("--model_dir", default='outputs/best_acc')
    parser.add_argument("--opt_metric", choices=["f1", "acc"], default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--normalize_text", action="store_true")
    parser.add_argument("--tta", action="store_true")
    args = parser.parse_args()

    metrics_path = args.metrics_path or os.path.join(
        args.output_dir, "metrics.json"
    )
    metrics = read_metrics(metrics_path)
    opt_metric = args.opt_metric or metrics.get("opt_metric")
    threshold = (
        args.threshold
        if args.threshold is not None
        else metrics.get("best_threshold", 0.5)
    )

    model_dir = resolve_model_dir(args.model_dir, args.output_dir, opt_metric)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizerFast.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir).to(device)

    test_records = read_jsonl(args.test_path)
    if args.normalize_text:
        test_records = apply_normalization(test_records)

    test_dataset = PairDataset(test_records)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=build_collate_fn(tokenizer, args.max_length),
    )

    test_probs = predict_probs(
        model, test_loader, device, progress_desc="Predict test"
    )

    if args.tta:
        swapped_records = [
            {"text1": r["text2"], "text2": r["text1"]}
            for r in test_records
        ]
        swapped_dataset = PairDataset(swapped_records)
        swapped_loader = DataLoader(
            swapped_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            collate_fn=build_collate_fn(tokenizer, args.max_length),
        )
        swapped_probs = predict_probs(
            model, swapped_loader, device, progress_desc="Predict test (TTA)"
        )
        test_probs = [
            (p1 + p2) / 2 for p1, p2 in zip(test_probs, swapped_probs)
        ]

    responses = [
        {"response": "1" if prob >= threshold else "0"}
        for prob in test_probs
    ]
    write_jsonl(args.result_path, responses)
    print(f"Saved predictions to {args.result_path}")


if __name__ == "__main__":
    main()
