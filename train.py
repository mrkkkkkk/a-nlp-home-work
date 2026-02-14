#train
import argparse
import json
import math
import os
import random
import re
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    BertForSequenceClassification,
    BertTokenizerFast,
    get_linear_schedule_with_warmup,
)


class PairDataset(Dataset):
    def __init__(self, records: List[dict], has_label: bool) -> None:
        self.records = records
        self.has_label = has_label

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx]
        text1 = record["text1"]
        text2 = record["text2"]
        if self.has_label:
            return text1, text2, int(record["label"])
        return text1, text2


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def ensure_file(path: str, label: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{label} not found: {path}. Please prepare splits first."
        )


def build_collate_fn(tokenizer, max_length: int, has_label: bool):
    def collate_fn(batch):
        texts1 = []
        texts2 = []
        labels = []
        for item in batch:
            if has_label:
                text1, text2, label = item
                labels.append(label)
            else:
                text1, text2 = item
            texts1.append(text1)
            texts2.append(text2)
        inputs = tokenizer(
            texts1,
            texts2,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        if has_label:
            inputs["labels"] = torch.tensor(labels, dtype=torch.long)
        return inputs

    return collate_fn


def compute_class_weights(records: List[dict]) -> Optional[torch.Tensor]:
    counts = {0: 0, 1: 0}
    for record in records:
        label = int(record["label"])
        counts[label] = counts.get(label, 0) + 1
    total = counts[0] + counts[1]
    if total == 0 or counts[0] == 0 or counts[1] == 0:
        return None
    weights = [total / (2 * counts[0]), total / (2 * counts[1])]
    return torch.tensor(weights, dtype=torch.float)


def compute_metrics(y_true: List[int], y_pred: List[int]) -> Tuple[float, float]:
    if not y_true:
        return 0.0, 0.0
    tp = fp = fn = tn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 1:
            tp += 1
        elif yt == 0 and yp == 1:
            fp += 1
        elif yt == 1 and yp == 0:
            fn += 1
        else:
            tn += 1
    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return f1, accuracy


def find_best_threshold(
    probs: List[float],
    labels: List[int],
    min_t: float,
    max_t: float,
    steps: int,
) -> Tuple[float, float, float]:
    if not labels:
        return 0.5, 0.0, 0.0
    best_f1 = -1.0
    best_acc = 0.0
    best_t = 0.5
    if steps < 2:
        steps = 2
    for i in range(steps):
        t = min_t + (max_t - min_t) * i / (steps - 1)
        preds = [1 if p >= t else 0 for p in probs]
        f1, acc = compute_metrics(labels, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_acc = acc
            best_t = t
    return best_t, best_f1, best_acc


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
            labels = batch.pop("labels", None)
            if labels is not None:
                labels = labels.to(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()
            all_probs.extend(probs)
    return all_probs


def predict_with_labels(
    model,
    dataloader: DataLoader,
    device: torch.device,
    progress_desc: Optional[str] = None,
) -> Tuple[List[float], List[int]]:
    model.eval()
    all_probs = []
    all_labels = []
    iterator = (
        tqdm(dataloader, desc=progress_desc, leave=False)
        if progress_desc
        else dataloader
    )
    with torch.no_grad():
        for batch in iterator:
            labels = batch.pop("labels")
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()
            all_probs.extend(probs)
            all_labels.extend(labels.tolist())
    return all_probs, all_labels


def print_device_info(device: torch.device) -> None:
    if device.type != "cuda":
        print("Warning: CUDA not available, running on CPU.")
        return
    gpu_count = torch.cuda.device_count()
    if gpu_count > 1:
        print(f"Using GPUs: {list(range(gpu_count))}")
    else:
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")


class FocalLoss(nn.Module):


    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean"
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            logits, targets, weight=self.alpha, reduction="none"
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class FGM:

    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 1.0,
        emb_name: str = "word_embeddings"
    ):
        self.model = model
        self.epsilon = epsilon
        self.emb_name = emb_name
        self.backup: Dict[str, torch.Tensor] = {}

    def attack(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.backup[name] = param.data.clone()
                if param.grad is not None:
                    norm = torch.norm(param.grad)
                    if norm != 0 and not torch.isnan(norm):
                        r_at = self.epsilon * param.grad / norm
                        param.data.add_(r_at)

    def restore(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if name in self.backup:
                    param.data = self.backup[name]
        self.backup = {}


class EMA:

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}
        self._register()

    def _register(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                new_average = (
                    self.decay * self.shadow[name] +
                    (1.0 - self.decay) * param.data
                )
                self.shadow[name] = new_average

    def apply_shadow(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


class EarlyStopping:

    def __init__(
        self,
        patience: int = 3,
        min_delta: float = 0.0,
        mode: str = "max"
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:

        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        return False


def compute_rdrop_loss(
    logits1: torch.Tensor,
    logits2: torch.Tensor,
    labels: torch.Tensor,
    loss_fn: nn.Module,
    kl_weight: float = 0.3
) -> torch.Tensor:

    ce_loss = (loss_fn(logits1, labels) + loss_fn(logits2, labels)) / 2

    p = F.softmax(logits1, dim=-1)
    q = F.softmax(logits2, dim=-1)

    kl_loss = (
        F.kl_div(F.log_softmax(logits1, dim=-1), q, reduction="batchmean") +
        F.kl_div(F.log_softmax(logits2, dim=-1), p, reduction="batchmean")
    ) / 2

    return ce_loss + kl_weight * kl_loss


def build_llrd_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    decay_rate: float = 0.9,
    num_layers: int = 12
) -> torch.optim.Optimizer:

    no_decay = {"bias", "LayerNorm.weight", "LayerNorm.bias"}

    param_groups = []

    emb_lr = lr * (decay_rate ** num_layers)
    emb_decay_params = []
    emb_no_decay_params = []

    layer_decay_params = [[] for _ in range(num_layers)]
    layer_no_decay_params = [[] for _ in range(num_layers)]
    top_decay_params = []
    top_no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        is_no_decay = any(nd in name for nd in no_decay)

        if "embeddings" in name:
            if is_no_decay:
                emb_no_decay_params.append(param)
            else:
                emb_decay_params.append(param)
        elif "encoder.layer" in name:

            layer_num = int(name.split("encoder.layer.")[1].split(".")[0])
            if is_no_decay:
                layer_no_decay_params[layer_num].append(param)
            else:
                layer_decay_params[layer_num].append(param)
        else:

            if is_no_decay:
                top_no_decay_params.append(param)
            else:
                top_decay_params.append(param)

    if emb_decay_params:
        param_groups.append({
            "params": emb_decay_params,
            "lr": emb_lr,
            "weight_decay": weight_decay
        })
    if emb_no_decay_params:
        param_groups.append({
            "params": emb_no_decay_params,
            "lr": emb_lr,
            "weight_decay": 0.0
        })

    for layer_num in range(num_layers):
        layer_lr = lr * (decay_rate ** (num_layers - layer_num - 1))
        if layer_decay_params[layer_num]:
            param_groups.append({
                "params": layer_decay_params[layer_num],
                "lr": layer_lr,
                "weight_decay": weight_decay
            })
        if layer_no_decay_params[layer_num]:
            param_groups.append({
                "params": layer_no_decay_params[layer_num],
                "lr": layer_lr,
                "weight_decay": 0.0
            })

    if top_decay_params:
        param_groups.append({
            "params": top_decay_params,
            "lr": lr,
            "weight_decay": weight_decay
        })
    if top_no_decay_params:
        param_groups.append({
            "params": top_no_decay_params,
            "lr": lr,
            "weight_decay": 0.0
        })

    return torch.optim.AdamW(param_groups)


def build_optimizer(model, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    no_decay = {"bias", "LayerNorm.weight"}
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(nd in name for nd in no_decay):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(param_groups, lr=lr)


def train_one_epoch_advanced(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    loss_fn: nn.Module,
    grad_accum_steps: int,
    max_grad_norm: float,
    scaler: Optional[torch.cuda.amp.GradScaler],
    progress_desc: str,
    use_rdrop: bool = False,
    rdrop_alpha: float = 0.3,
    use_fgm: bool = False,
    fgm: Optional[FGM] = None,
    ema: Optional[EMA] = None,
) -> float:

    model.train()
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    total_steps = len(dataloader)
    progress = tqdm(dataloader, desc=progress_desc, leave=False)

    for step, batch in enumerate(progress, start=1):
        labels = batch.pop("labels").to(device)
        batch = {key: value.to(device) for key, value in batch.items()}

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            if use_rdrop:
                logits1 = model(**batch).logits
                logits2 = model(**batch).logits
                loss = compute_rdrop_loss(
                    logits1, logits2, labels, loss_fn, rdrop_alpha
                )
            else:
                logits = model(**batch).logits
                loss = loss_fn(logits, labels)

            loss = loss / grad_accum_steps

        if scaler is None:
            loss.backward()
        else:
            scaler.scale(loss).backward()

        if use_fgm and fgm is not None:
            fgm.attack()
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                if use_rdrop:
                    adv_logits1 = model(**batch).logits
                    adv_logits2 = model(**batch).logits
                    adv_loss = compute_rdrop_loss(
                        adv_logits1, adv_logits2, labels, loss_fn, rdrop_alpha
                    )
                else:
                    adv_logits = model(**batch).logits
                    adv_loss = loss_fn(adv_logits, labels)
                adv_loss = adv_loss / grad_accum_steps

            if scaler is None:
                adv_loss.backward()
            else:
                scaler.scale(adv_loss).backward()
            fgm.restore()

        total_loss += loss.item() * grad_accum_steps

        if step % grad_accum_steps == 0 or step == total_steps:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            if scaler is None:
                optimizer.step()
            else:
                scaler.step(optimizer)
                scaler.update()

            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            if ema is not None:
                ema.update()

            avg_loss = total_loss / step
            progress.set_postfix(loss=f"{avg_loss:.4f}")

    return total_loss / max(1, len(dataloader))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Advanced training for Chinese sentence pair similarity"
    )

    parser.add_argument("--model_dir", default="chinese-roberta-wwm-ext")
    parser.add_argument("--train_split_path", default="outputs/splits/train.jsonl")
    parser.add_argument("--val_split_path", default="outputs/splits/val.jsonl")
    parser.add_argument("--test_path", default="data/test.jsonl")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--use_class_weight", action="store_true")
    parser.add_argument("--normalize_text", action="store_true")
    parser.add_argument("--save_best", action="store_true", default=True)
    parser.add_argument("--no_save_best", action="store_false", dest="save_best")
    parser.add_argument("--save_last", action="store_true")
    parser.add_argument("--search_threshold", action="store_true", default=True)
    parser.add_argument("--no_search_threshold", action="store_false", dest="search_threshold")
    parser.add_argument("--threshold_min", type=float, default=0.1)
    parser.add_argument("--threshold_max", type=float, default=0.9)
    parser.add_argument("--threshold_steps", type=int, default=81)
    parser.add_argument("--predict_test", action="store_true")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--output_scores", action="store_true")
    parser.add_argument("--load_best_for_test", action="store_true", default=True)
    parser.add_argument("--no_load_best_for_test", action="store_false", dest="load_best_for_test")
    parser.add_argument(
        "--best_metric",
        choices=["f1", "acc"],
        default="f1",
        help="Which best model to load for test inference.",
    )

    parser.add_argument(
        "--use_focal_loss",
        action="store_true",
        help="Use Focal Loss instead of Cross-Entropy"
    )
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=2.0,
        help="Focal Loss gamma parameter (focusing strength)"
    )

    parser.add_argument(
        "--use_rdrop",
        action="store_true",
        help="Use R-Drop consistency regularization"
    )
    parser.add_argument(
        "--rdrop_alpha",
        type=float,
        default=0.3,
        help="R-Drop KL divergence weight"
    )

    parser.add_argument(
        "--use_fgm",
        action="store_true",
        help="Use FGM adversarial training"
    )
    parser.add_argument(
        "--fgm_epsilon",
        type=float,
        default=1.0,
        help="FGM perturbation magnitude"
    )

    parser.add_argument(
        "--use_llrd",
        action="store_true",
        help="Use Layer-wise Learning Rate Decay"
    )
    parser.add_argument(
        "--llrd_decay",
        type=float,
        default=0.9,
        help="LLRD decay rate per layer"
    )

    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Use Exponential Moving Average"
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.999,
        help="EMA decay rate"
    )

    parser.add_argument(
        "--early_stopping",
        action="store_true",
        help="Enable early stopping"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Early stopping patience (epochs)"
    )

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_device_info(device)

    techniques = []
    if args.use_focal_loss:
        techniques.append(f"Focal Loss (gamma={args.focal_gamma})")
    if args.use_rdrop:
        techniques.append(f"R-Drop (alpha={args.rdrop_alpha})")
    if args.use_fgm:
        techniques.append(f"FGM (epsilon={args.fgm_epsilon})")
    if args.use_llrd:
        techniques.append(f"LLRD (decay={args.llrd_decay})")
    if args.use_ema:
        techniques.append(f"EMA (decay={args.ema_decay})")
    if args.early_stopping:
        techniques.append(f"Early Stopping (patience={args.patience})")

    if techniques:
        print(f"Enabled techniques: {', '.join(techniques)}")
    else:
        print("No advanced techniques enabled (baseline mode)")

    ensure_file(args.train_split_path, "Train split")
    ensure_file(args.val_split_path, "Validation split")

    train_records = read_jsonl(args.train_split_path)
    val_records = read_jsonl(args.val_split_path)
    if args.normalize_text:
        train_records = apply_normalization(train_records)
        val_records = apply_normalization(val_records)

    print(f"Split sizes -> train: {len(train_records)}, val: {len(val_records)}")

    tokenizer = BertTokenizerFast.from_pretrained(args.model_dir)
    model = BertForSequenceClassification.from_pretrained(
        args.model_dir, num_labels=2
    )
    model.to(device)

    gpu_count = torch.cuda.device_count() if device.type == "cuda" else 0
    if device.type == "cuda" and gpu_count > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(gpu_count)))

    train_dataset = PairDataset(train_records, has_label=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=build_collate_fn(tokenizer, args.max_length, has_label=True),
    )

    val_loader = None
    if val_records:
        val_dataset = PairDataset(val_records, has_label=True)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            collate_fn=build_collate_fn(tokenizer, args.max_length, has_label=True),
        )

    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
    total_steps = max(1, steps_per_epoch * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)

    base_model = model.module if hasattr(model, "module") else model
    if args.use_llrd:
        optimizer = build_llrd_optimizer(
            base_model, args.lr, args.weight_decay, args.llrd_decay
        )
        print(f"Using LLRD optimizer with decay={args.llrd_decay}")
    else:
        optimizer = build_optimizer(model, args.lr, args.weight_decay)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    class_weights = None
    if args.use_class_weight or args.use_focal_loss:
        class_weights = compute_class_weights(train_records)
        if class_weights is not None:
            class_weights = class_weights.to(device)
            print(f"Using class weights: {class_weights.tolist()}")

    if args.use_focal_loss:
        loss_fn = FocalLoss(alpha=class_weights, gamma=args.focal_gamma)
        print(f"Using Focal Loss with gamma={args.focal_gamma}")
    else:
        def loss_fn(logits, labels):
            return F.cross_entropy(logits, labels, weight=class_weights)

    fgm = None
    if args.use_fgm:
        fgm = FGM(base_model, epsilon=args.fgm_epsilon)
        print(f"Using FGM adversarial training with epsilon={args.fgm_epsilon}")

    ema = None
    if args.use_ema:
        ema = EMA(base_model, decay=args.ema_decay)
        print(f"Using EMA with decay={args.ema_decay}")

    early_stopper = None
    if args.early_stopping:
        early_stopper = EarlyStopping(patience=args.patience, mode="max")
        print(f"Using Early Stopping with patience={args.patience}")

    use_amp = args.fp16 and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_f1 = -1.0
    best_f1_threshold = 0.5
    best_f1_acc = 0.0
    best_f1_epoch = 0

    best_acc = -1.0
    best_acc_threshold = 0.5
    best_acc_f1 = 0.0
    best_acc_epoch = 0

    metrics_path = os.path.join(args.output_dir, "metrics.json")
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch_advanced(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            loss_fn,
            args.grad_accum_steps,
            args.max_grad_norm,
            scaler if use_amp else None,
            progress_desc=f"Train {epoch}/{args.epochs}",
            use_rdrop=args.use_rdrop,
            rdrop_alpha=args.rdrop_alpha,
            use_fgm=args.use_fgm,
            fgm=fgm,
            ema=ema,
        )
        print(f"Epoch {epoch}/{args.epochs} - train loss: {train_loss:.4f}")

        if val_loader is None:
            continue

        if ema is not None:
            ema.apply_shadow()

        val_probs, val_labels = predict_with_labels(
            model, val_loader, device, progress_desc="Validate"
        )

        if ema is not None:
            ema.restore()

        val_preds = [1 if p >= 0.5 else 0 for p in val_probs]
        f1, acc = compute_metrics(val_labels, val_preds)
        print(f"Validation (threshold=0.5) - F1: {f1:.4f} | Acc: {acc:.4f}")

        if args.search_threshold:
            threshold, best_t_f1, best_t_acc = find_best_threshold(
                val_probs,
                val_labels,
                args.threshold_min,
                args.threshold_max,
                args.threshold_steps,
            )
            print(
                "Validation (best threshold) - "
                f"F1: {best_t_f1:.4f} | Acc: {best_t_acc:.4f} | "
                f"thr={threshold:.3f}"
            )
        else:
            threshold, best_t_f1, best_t_acc = 0.5, f1, acc

        if best_t_f1 > best_f1:
            best_f1 = best_t_f1
            best_f1_acc = best_t_acc
            best_f1_threshold = threshold
            best_f1_epoch = epoch
            if args.save_best:
                best_dir = os.path.join(args.output_dir, "best_f1")
                model_to_save = base_model
                if ema is not None:
                    ema.apply_shadow()
                model_to_save.save_pretrained(best_dir)
                if ema is not None:
                    ema.restore()
                tokenizer.save_pretrained(best_dir)
                print(f"Saved best F1 model to {best_dir}")

        if best_t_acc > best_acc:
            best_acc = best_t_acc
            best_acc_f1 = best_t_f1
            best_acc_threshold = threshold
            best_acc_epoch = epoch
            if args.save_best:
                best_dir = os.path.join(args.output_dir, "best_acc")
                model_to_save = base_model
                if ema is not None:
                    ema.apply_shadow()
                model_to_save.save_pretrained(best_dir)
                if ema is not None:
                    ema.restore()
                tokenizer.save_pretrained(best_dir)
                print(f"Saved best Acc model to {best_dir}")


        if early_stopper is not None:
            if early_stopper(best_t_f1):
                print(f"Early stopping triggered at epoch {epoch}")
                break

    if args.save_last:
        last_dir = os.path.join(args.output_dir, "last")
        model_to_save = base_model
        if ema is not None:
            ema.apply_shadow()
        model_to_save.save_pretrained(last_dir)
        if ema is not None:
            ema.restore()
        tokenizer.save_pretrained(last_dir)
        print(f"Saved last model to {last_dir}")

    metrics_payload = {
        "best_f1_epoch": best_f1_epoch,
        "best_f1": best_f1,
        "best_f1_acc": best_f1_acc,
        "best_f1_threshold": best_f1_threshold,
        "best_acc_epoch": best_acc_epoch,
        "best_acc": best_acc,
        "best_acc_f1": best_acc_f1,
        "best_acc_threshold": best_acc_threshold,
        "epochs": epoch,  
        "techniques": {
            "focal_loss": args.use_focal_loss,
            "focal_gamma": args.focal_gamma if args.use_focal_loss else None,
            "rdrop": args.use_rdrop,
            "rdrop_alpha": args.rdrop_alpha if args.use_rdrop else None,
            "fgm": args.use_fgm,
            "fgm_epsilon": args.fgm_epsilon if args.use_fgm else None,
            "llrd": args.use_llrd,
            "llrd_decay": args.llrd_decay if args.use_llrd else None,
            "ema": args.use_ema,
            "ema_decay": args.ema_decay if args.use_ema else None,
            "early_stopping": args.early_stopping,
            "patience": args.patience if args.early_stopping else None,
        }
    }
    write_json(metrics_path, metrics_payload)
    print(f"Saved metrics to {metrics_path}")

    if args.predict_test and os.path.exists(args.test_path):
        if args.load_best_for_test:
            best_dir_name = "best_f1" if args.best_metric == "f1" else "best_acc"
            best_dir = os.path.join(args.output_dir, best_dir_name)
            if os.path.isdir(best_dir):
                model = BertForSequenceClassification.from_pretrained(best_dir).to(device)
                if device.type == "cuda" and gpu_count > 1:
                    model = torch.nn.DataParallel(model, device_ids=list(range(gpu_count)))
                tokenizer = BertTokenizerFast.from_pretrained(best_dir)
                print(f"Loaded {args.best_metric} best model from {best_dir} for test")
            else:
                print(f"Warning: {best_dir} not found, using current model for test.")

        test_records = read_jsonl(args.test_path)
        if args.normalize_text:
            test_records = apply_normalization(test_records)

        test_dataset = PairDataset(test_records, has_label=False)
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            collate_fn=build_collate_fn(tokenizer, args.max_length, has_label=False),
        )

        test_probs = predict_probs(model, test_loader, device, progress_desc="Predict test")

        if args.tta:
            swapped_records = [
                {"text1": r["text2"], "text2": r["text1"]}
                for r in test_records
            ]
            swapped_dataset = PairDataset(swapped_records, has_label=False)
            swapped_loader = DataLoader(
                swapped_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
                collate_fn=build_collate_fn(tokenizer, args.max_length, has_label=False),
            )
            swapped_probs = predict_probs(
                model, swapped_loader, device, progress_desc="Predict test (TTA)"
            )
            test_probs = [
                (p1 + p2) / 2 for p1, p2 in zip(test_probs, swapped_probs)
            ]

        if val_records:
            threshold = (
                best_f1_threshold if args.best_metric == "f1" else best_acc_threshold
            )
        else:
            threshold = 0.5
        test_preds = [1 if p >= threshold else 0 for p in test_probs]

        pred_ones = sum(test_preds)
        pred_zeros = len(test_preds) - pred_ones
        print(f"Test predictions: {pred_ones} positive ({100*pred_ones/len(test_preds):.1f}%), "
              f"{pred_zeros} negative ({100*pred_zeros/len(test_preds):.1f}%)")

        output_path = os.path.join(args.output_dir, "test_pred.jsonl")
        with open(output_path, "w", encoding="utf-8") as file:
            for record, pred, prob in zip(test_records, test_preds, test_probs):
                out = {
                    "text1": record["text1"],
                    "text2": record["text2"],
                    "pred_label": int(pred),
                }
                if args.output_scores:
                    out["score"] = float(prob)
                file.write(json.dumps(out, ensure_ascii=False) + "\n")
        print(f"Saved test predictions to {output_path}")


if __name__ == "__main__":
    main()
