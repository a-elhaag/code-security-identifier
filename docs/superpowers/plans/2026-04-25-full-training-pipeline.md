# Full Training Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a single `train.py` script that runs SCL + R-Drop + EDAT + hyperparameter sweep, saves per-run checkpoints, and selects the best model by macro F1 — with AMP for Colab T4/A100.

**Architecture:** Single Python script loads the pre-built token cache (`tokens_maxlen512.pt`), runs configurable experiments sequentially, saves a checkpoint per run, then copies the best checkpoint to `checkpoints/best_overall.pt`. Tokenization (notebook `01_tokenization.ipynb`) is a prerequisite — `train.py` does NOT re-tokenize.

**Tech Stack:** PyTorch, HuggingFace Transformers, PEFT (LoRA), scikit-learn, PyYAML, tqdm, `torch.amp` (AMP/FP16 for Colab)

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `train.py` | **Create** | Full training pipeline — all experiments, EDAT, SCL, R-Drop, AMP, checkpointing, best-model selection |
| `config.yaml` | **Modify** | Add new fields: `lora_target_modules` with Q/K/V, EDAT params, SCL/R-Drop weights, experiment sweep grid |
| `checkpoints/` | Auto-created | Per-run checkpoints + `best_overall.pt` |
| `logs/` | Auto-created | Per-run JSON logs + `experiment_summary.json` |

---

## Task 1: Update config.yaml with all new fields

**Files:**
- Modify: `config.yaml`

- [ ] **Step 1: Replace config.yaml with full expanded config**

```yaml
# CSI Training Configuration

# ── Paths ────────────────────────────────────────────────────────────────────
data_dir: datasets
unified_jsonl: datasets/UNIFIED.jsonl
mappings_json: datasets/UNIFIED_mappings.json
token_cache_dir: datasets/token_cache
token_cache_file: datasets/token_cache/tokens_maxlen512.pt
checkpoint_dir: checkpoints
log_dir: logs

# ── Model ────────────────────────────────────────────────────────────────────
model_name: microsoft/graphcodebert-base
num_cwe_classes: 8
hidden_size: 768

# ── LoRA (baseline — overridden by sweep) ────────────────────────────────────
lora_r: 16
lora_alpha: 32
lora_dropout: 0.1
lora_target_modules:
  - query
  - key
  - value

# ── Tokenization (already done — do not re-run) ──────────────────────────────
max_length: 512
chunk_stride: 256

# ── Training (baseline — overridden by sweep) ────────────────────────────────
batch_size: 8
epochs: 10
learning_rate: 2.0e-5
weight_decay: 0.01
warmup_ratio: 0.2
max_grad_norm: 1.0
seed: 42
early_stopping_patience: 3
save_every_n_steps: 200
log_every_n_steps: 50

# ── AMP (Colab T4/A100) ──────────────────────────────────────────────────────
use_amp: true          # Set false on CPU/MPS

# ── Focal Loss ───────────────────────────────────────────────────────────────
focal_gamma: 2.0

# ── SCL (Supervised Contrastive Learning) ────────────────────────────────────
scl_weight: 0.3
scl_temperature: 0.07

# ── R-Drop ───────────────────────────────────────────────────────────────────
rdrop_alpha: 4.0

# ── EDAT (Embedding-space Data Augmentation Training) ────────────────────────
edat_enabled: true
edat_epsilon: 0.01      # PGD perturbation magnitude
edat_steps: 3           # PGD inner loop steps
edat_step_size: 0.005   # PGD step size
edat_kl_weight: 1.0     # Weight of adversarial KL loss

# ── Experiment Sweep Grid ────────────────────────────────────────────────────
# Each combination is one run. Best macro F1 wins.
sweep:
  batch_size: [4, 8]
  learning_rate: [1.0e-5, 2.0e-5, 5.0e-5]
  lora_r: [4, 8, 16]
  focal_gamma: [1.0, 2.0, 3.0]
```

- [ ] **Step 2: Verify config loads without error**

```bash
python -c "import yaml; cfg=yaml.safe_load(open('config.yaml')); print('sweep combinations:', len(cfg['sweep']['batch_size']) * len(cfg['sweep']['learning_rate']) * len(cfg['sweep']['lora_r']) * len(cfg['sweep']['focal_gamma']))"
```

Expected output: `sweep combinations: 54`

---

## Task 2: Create train.py — imports, config, device, seed

**Files:**
- Create: `train.py`

- [ ] **Step 1: Write the file header with all imports and setup**

```python
"""
Full CSI training pipeline.
Prerequisite: run 01_tokenization.ipynb first to generate token cache.
Usage: python train.py [--config config.yaml] [--run-sweep] [--single]
"""
import argparse
import copy
import json
import os
import random
import time
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import (classification_report, f1_score,
                              precision_score, recall_score)
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoModel, get_linear_schedule_with_warmup
from tqdm.auto import tqdm


CWE_8_CLASSES = [
    "CWE-077", "CWE-601", "CWE-022", "CWE-094",
    "CWE-089", "CWE-352", "CWE-079", "unknown",
]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(cfg: dict) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--run-sweep", action="store_true",
                   help="Run full hyperparameter sweep")
    p.add_argument("--single", action="store_true",
                   help="Run single experiment with base config values")
    return p.parse_args()
```

- [ ] **Step 2: Verify imports resolve**

```bash
python -c "import train" 2>&1 | head -20
```

Expected: no ImportError (may warn about missing model weights — that's fine)

---

## Task 3: Dataset class

**Files:**
- Modify: `train.py` (append)

- [ ] **Step 1: Append VulnerabilityDataset class**

```python
class VulnerabilityDataset(Dataset):
    def __init__(self, cache: dict, split: str):
        if split == "all":
            indices = list(range(len(cache["split_origins"])))
        else:
            indices = [i for i, s in enumerate(cache["split_origins"]) if s == split]
        self.input_ids      = cache["input_ids"][indices]
        self.attention_mask = cache["attention_mask"][indices]
        self.cwe_labels     = cache["cwe_labels"][indices]
        self.binary_labels  = cache["binary_labels"][indices]
        self.global_ids     = cache["global_ids"][indices]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "cwe_label":      self.cwe_labels[idx],
            "binary_label":   self.binary_labels[idx],
            "global_id":      self.global_ids[idx],
        }


def make_loaders(cache: dict, batch_size: int):
    train_ds = VulnerabilityDataset(cache, "train")
    val_ds   = VulnerabilityDataset(cache, "val")

    # Weighted sampler to balance CWE classes
    labels   = train_ds.cwe_labels.tolist()
    counts   = torch.bincount(torch.tensor(labels), minlength=8).float()
    weights  = 1.0 / counts.clamp(min=1)
    sample_w = weights[torch.tensor(labels)]
    sampler  = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=2, pin_memory=True, prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=2, pin_memory=True,
    )
    return train_loader, val_loader, counts
```

---

## Task 4: Model, loss functions

**Files:**
- Modify: `train.py` (append)

- [ ] **Step 1: Append model and loss classes**

```python
# ── Classification head ───────────────────────────────────────────────────────

class CWEClassificationHead(nn.Module):
    def __init__(self, hidden_size: int, num_classes: int = 8, dropout: float = 0.1):
        super().__init__()
        mid = hidden_size // 2
        self.norm = nn.LayerNorm(hidden_size)
        self.fc1  = nn.Linear(hidden_size, mid)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2  = nn.Linear(mid, num_classes)

    def forward(self, x):
        x = self.norm(x)
        x = self.drop(self.act(self.fc1(x)))
        return self.fc2(x)


# ── Main model ────────────────────────────────────────────────────────────────

class GraphCodeBERTLoRACWEModel(nn.Module):
    def __init__(self, model_name: str, num_cwe_classes: int, lora_r: int,
                 lora_alpha: int, lora_dropout: float, focal_gamma: float,
                 class_weights: torch.Tensor | None = None):
        super().__init__()
        encoder = AutoModel.from_pretrained(model_name)
        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            target_modules=["query", "key", "value"],
        )
        self.encoder     = get_peft_model(encoder, lora_cfg)
        self.cwe_head    = CWEClassificationHead(encoder.config.hidden_size, num_cwe_classes)
        self.focal_gamma = focal_gamma
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    @staticmethod
    def _mean_pool(hidden_states, attention_mask):
        mask   = attention_mask.unsqueeze(-1).float()
        summed = (hidden_states * mask).sum(1)
        counts = mask.sum(1).clamp(min=1e-9)
        return summed / counts

    def forward(self, input_ids, attention_mask, embeds=None):
        """
        embeds: optional pre-computed embeddings (used by EDAT adversarial pass).
        Returns dict with 'logits' and 'features'.
        """
        if embeds is not None:
            out = self.encoder(inputs_embeds=embeds, attention_mask=attention_mask)
        else:
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        features = self._mean_pool(out.last_hidden_state, attention_mask)
        logits   = self.cwe_head(features)
        return {"logits": logits, "features": features}

    def focal_loss(self, logits, labels):
        ce_w  = F.cross_entropy(logits, labels, weight=self.class_weights, reduction="none")
        pt    = torch.exp(-F.cross_entropy(logits, labels, reduction="none"))
        return (((1 - pt) ** self.focal_gamma) * ce_w).mean()

    def get_input_embeddings(self, input_ids):
        """Return token embeddings from base encoder (used by EDAT)."""
        return self.encoder.base_model.model.embeddings.word_embeddings(input_ids)


# ── Supervised Contrastive Loss ───────────────────────────────────────────────

class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        features: (2B, H) — L2-normalised representations from two forward passes
        labels:   (2B,)  — CWE class indices repeated twice
        """
        features  = F.normalize(features, p=2, dim=1)
        B         = features.shape[0]
        labels_c  = labels.contiguous().view(-1, 1)
        mask      = torch.eq(labels_c, labels_c.T).float().to(features.device)

        logits    = torch.div(torch.matmul(features, features.T), self.temperature)
        logits    = logits - logits.max(dim=1, keepdim=True).values.detach()

        diag_mask = 1 - torch.eye(B, device=features.device)
        mask      = mask * diag_mask
        exp_logits = torch.exp(logits) * diag_mask
        log_prob   = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)
        loss       = -(mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)
        return loss.mean()


# ── R-Drop Loss ───────────────────────────────────────────────────────────────

class RDropLoss(nn.Module):
    def __init__(self, alpha: float = 4.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, logits1: torch.Tensor, logits2: torch.Tensor,
                target: torch.Tensor, class_weights: torch.Tensor | None = None) -> torch.Tensor:
        ce   = nn.CrossEntropyLoss(weight=class_weights)
        ce_loss  = 0.5 * (ce(logits1, target) + ce(logits2, target))
        p_loss   = F.kl_div(F.log_softmax(logits1, dim=-1), F.softmax(logits2, dim=-1), reduction="none")
        q_loss   = F.kl_div(F.log_softmax(logits2, dim=-1), F.softmax(logits1, dim=-1), reduction="none")
        kl_loss  = 0.5 * (p_loss.sum() + q_loss.sum()) / logits1.shape[0]
        return ce_loss + self.alpha * kl_loss
```

---

## Task 5: EDAT (Embedding-space Data Augmentation Training)

**Files:**
- Modify: `train.py` (append)

- [ ] **Step 1: Append EDAT adversarial perturbation function**

```python
def edat_adversarial_loss(
    model: GraphCodeBERTLoRACWEModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    steps: int,
    step_size: float,
    kl_weight: float,
    use_amp: bool,
    device: torch.device,
) -> torch.Tensor:
    """
    PGD perturbation in embedding space.
    1. Get clean embeddings (no grad).
    2. Add random init delta inside epsilon-ball.
    3. PGD steps: maximise KL(clean_logits || perturbed_logits).
    4. Return KL loss on best perturbation (used to regularise training).
    """
    model.eval()  # freeze BN/dropout for adversarial search
    with torch.no_grad():
        clean_out = model(input_ids, attention_mask)
        clean_logits = clean_out["logits"].detach()
        embeds = model.get_input_embeddings(input_ids).detach()

    delta = torch.zeros_like(embeds).uniform_(-epsilon, epsilon)
    delta.requires_grad_(True)

    for _ in range(steps):
        with torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
            adv_out = model(input_ids, attention_mask, embeds=embeds + delta)
            # Maximise KL divergence (adversary wants to push model off clean prediction)
            kl = F.kl_div(
                F.log_softmax(adv_out["logits"], dim=-1),
                F.softmax(clean_logits, dim=-1),
                reduction="batchmean",
            )
        kl.backward()
        with torch.no_grad():
            grad = delta.grad.detach()
            delta = delta + step_size * grad.sign()
            delta = delta.clamp(-epsilon, epsilon)
            delta = delta.detach().requires_grad_(True)

    model.train()  # restore training mode
    with torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
        adv_out_final = model(input_ids, attention_mask, embeds=(embeds + delta).detach())
        adv_kl = F.kl_div(
            F.log_softmax(adv_out_final["logits"], dim=-1),
            F.softmax(clean_logits, dim=-1),
            reduction="batchmean",
        )
    return kl_weight * adv_kl
```

---

## Task 6: Training and validation loops

**Files:**
- Modify: `train.py` (append)

- [ ] **Step 1: Append train_epoch() and validate() functions**

```python
def validate(model, loader, device, use_amp):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    n_batches  = 0
    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            lbls = batch["cwe_label"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
                out = model(ids, mask)
                loss = model.focal_loss(out["logits"], lbls)
            total_loss += loss.item()
            all_preds.extend(out["logits"].argmax(-1).cpu().tolist())
            all_labels.extend(lbls.cpu().tolist())
            n_batches += 1
    avg_loss   = total_loss / max(n_batches, 1)
    macro_f1   = f1_score(all_labels, all_preds, average="macro",    zero_division=0)
    macro_prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    macro_rec  = recall_score(all_labels, all_preds, average="macro",  zero_division=0)
    report     = classification_report(
        all_labels, all_preds, target_names=CWE_8_CLASSES, zero_division=0
    )
    return avg_loss, macro_f1, macro_prec, macro_rec, report


def train_one_run(model, train_loader, val_loader, cfg, run_cfg, device,
                  class_weights, run_id, log_dir, ckpt_dir):
    """
    Full training loop for one experiment configuration.
    Returns best_val_f1, path to best checkpoint.
    """
    use_amp     = cfg.get("use_amp", True) and device.type == "cuda"
    scaler      = GradScaler(enabled=use_amp)
    scl_crit    = SupervisedContrastiveLoss(temperature=cfg["scl_temperature"])
    rdrop_crit  = RDropLoss(alpha=cfg["rdrop_alpha"])

    optimizer  = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=run_cfg["learning_rate"], weight_decay=cfg["weight_decay"],
    )
    total_steps = len(train_loader) * run_cfg["epochs"]
    warmup_steps = int(total_steps * cfg["warmup_ratio"])
    scheduler  = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    ckpt_dir   = Path(ckpt_dir) / run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path   = Path(log_dir) / f"{run_id}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    best_f1         = 0.0
    best_ckpt_path  = None
    patience_count  = 0
    history         = []
    global_step     = 0

    for epoch in range(1, run_cfg["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        pbar = tqdm(train_loader, desc=f"[{run_id}] Epoch {epoch}/{run_cfg['epochs']}", leave=False)

        for step, batch in enumerate(pbar, 1):
            ids  = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            lbls = batch["cwe_label"].to(device, non_blocking=True)

            optimizer.zero_grad()

            # ── Dual forward pass (R-Drop + SCL) ─────────────────────────
            with torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
                out1 = model(ids, mask)
                out2 = model(ids, mask)

                loss_rdrop = rdrop_crit(out1["logits"], out2["logits"], lbls, class_weights)
                loss_scl   = scl_crit(
                    torch.cat([out1["features"], out2["features"]], dim=0),
                    torch.cat([lbls, lbls], dim=0),
                )
                loss = loss_rdrop + cfg["scl_weight"] * loss_scl

            # ── EDAT adversarial loss ─────────────────────────────────────
            if cfg.get("edat_enabled", False):
                loss_edat = edat_adversarial_loss(
                    model, ids, mask, lbls,
                    epsilon=cfg["edat_epsilon"],
                    steps=cfg["edat_steps"],
                    step_size=cfg["edat_step_size"],
                    kl_weight=cfg["edat_kl_weight"],
                    use_amp=use_amp,
                    device=device,
                )
                loss = loss + loss_edat

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()),
                cfg["max_grad_norm"],
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

            if global_step % cfg["save_every_n_steps"] == 0:
                step_ckpt = ckpt_dir / f"step_{global_step}.pt"
                torch.save({"epoch": epoch, "step": global_step,
                            "model_state_dict": model.state_dict()}, step_ckpt)

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # ── Validation ────────────────────────────────────────────────────
        val_loss, val_f1, val_prec, val_rec, report = validate(model, val_loader, device, use_amp)
        elapsed = time.time() - t0
        print(f"[{run_id}] Epoch {epoch} | train_loss={epoch_loss/len(train_loader):.4f} "
              f"| val_loss={val_loss:.4f} | val_f1={val_f1:.4f} | {elapsed:.1f}s")
        print(report)

        entry = {
            "epoch": epoch,
            "train_loss": round(epoch_loss / len(train_loader), 6),
            "val_loss": round(val_loss, 6),
            "val_f1": round(val_f1, 6),
            "val_prec": round(val_prec, 6),
            "val_rec": round(val_rec, 6),
            "elapsed_s": round(elapsed, 1),
        }
        history.append(entry)

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_count = 0
            best_ckpt_path = ckpt_dir / "best.pt"
            torch.save({
                "epoch": epoch,
                "val_f1": best_f1,
                "model_state_dict": model.state_dict(),
                "run_cfg": run_cfg,
                "base_cfg": cfg,
            }, best_ckpt_path)
            print(f"  ✓ New best F1={best_f1:.4f} → saved {best_ckpt_path}")
        else:
            patience_count += 1
            if patience_count >= cfg["early_stopping_patience"]:
                print(f"  Early stopping at epoch {epoch} (patience={cfg['early_stopping_patience']})")
                break

    # ── Save run log ──────────────────────────────────────────────────────
    run_log = {
        "run_id": run_id,
        "run_date": datetime.now().isoformat(),
        "device": str(device),
        "best_val_f1": round(best_f1, 6),
        "run_cfg": run_cfg,
        "history": history,
    }
    log_path.write_text(json.dumps(run_log, indent=2))
    return best_f1, best_ckpt_path
```

---

## Task 7: Model factory and experiment runner

**Files:**
- Modify: `train.py` (append)

- [ ] **Step 1: Append build_model() and run_experiment() helpers**

```python
def build_model(cfg, run_cfg, class_weights, device):
    model = GraphCodeBERTLoRACWEModel(
        model_name=cfg["model_name"],
        num_cwe_classes=cfg["num_cwe_classes"],
        lora_r=run_cfg["lora_r"],
        lora_alpha=run_cfg["lora_r"] * 2,      # alpha = 2 * r convention
        lora_dropout=cfg["lora_dropout"],
        focal_gamma=run_cfg["focal_gamma"],
        class_weights=class_weights.to(device),
    )
    return model.to(device)


def run_experiment(cfg, run_cfg, cache, device, log_dir, ckpt_dir):
    """Build model + loaders, run training, return (best_f1, ckpt_path, run_id)."""
    run_id = (
        f"lr{run_cfg['learning_rate']:.0e}"
        f"_bs{run_cfg['batch_size']}"
        f"_r{run_cfg['lora_r']}"
        f"_g{run_cfg['focal_gamma']}"
    )
    print(f"\n{'='*60}")
    print(f"RUN: {run_id}")
    print(f"{'='*60}")

    set_seed(cfg["seed"])
    train_loader, val_loader, counts = make_loaders(cache, run_cfg["batch_size"])
    class_weights = (1.0 / counts.clamp(min=1))
    class_weights = class_weights / class_weights.sum() * len(counts)

    model = build_model(cfg, run_cfg, class_weights, device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    best_f1, ckpt_path = train_one_run(
        model, train_loader, val_loader, cfg, run_cfg,
        device, class_weights.to(device), run_id, log_dir, ckpt_dir,
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best_f1, ckpt_path, run_id
```

---

## Task 8: Main entrypoint — sweep + best-model selection

**Files:**
- Modify: `train.py` (append)

- [ ] **Step 1: Append main() with sweep logic and best-model copy**

```python
def main():
    args = parse_args()
    cfg  = load_config(args.config)
    device = get_device(cfg)
    use_amp = cfg.get("use_amp", True) and device.type == "cuda"
    print(f"Device: {device} | AMP: {use_amp}")

    cache_path = Path(cfg["token_cache_file"])
    assert cache_path.exists(), f"Token cache not found: {cache_path}. Run 01_tokenization.ipynb first."
    print(f"Loading token cache from {cache_path} …")
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    print(f"  Chunks: {cache['num_records']} | Functions: {cache['num_functions']}")

    log_dir  = cfg["log_dir"]
    ckpt_dir = cfg["checkpoint_dir"]

    if args.single or not args.run_sweep:
        # Single run with base config values
        run_cfg = {
            "learning_rate": cfg["learning_rate"],
            "batch_size":    cfg["batch_size"],
            "lora_r":        cfg["lora_r"],
            "focal_gamma":   cfg["focal_gamma"],
            "epochs":        cfg["epochs"],
        }
        best_f1, best_ckpt, run_id = run_experiment(cfg, run_cfg, cache, device, log_dir, ckpt_dir)
        results = [{"run_id": run_id, "val_f1": best_f1, "ckpt": str(best_ckpt), "run_cfg": run_cfg}]
    else:
        # Full sweep
        sweep   = cfg["sweep"]
        grid    = list(product(
            sweep["batch_size"],
            sweep["learning_rate"],
            sweep["lora_r"],
            sweep["focal_gamma"],
        ))
        print(f"Sweep: {len(grid)} combinations")
        results = []
        for bs, lr, r, gamma in grid:
            run_cfg = {
                "learning_rate": lr,
                "batch_size":    bs,
                "lora_r":        r,
                "focal_gamma":   gamma,
                "epochs":        cfg["epochs"],
            }
            best_f1, best_ckpt, run_id = run_experiment(
                cfg, run_cfg, cache, device, log_dir, ckpt_dir
            )
            results.append({
                "run_id": run_id,
                "val_f1": best_f1,
                "ckpt": str(best_ckpt),
                "run_cfg": run_cfg,
            })

    # ── Select overall best, copy to checkpoints/best_overall.pt ─────────
    results.sort(key=lambda x: x["val_f1"], reverse=True)
    winner = results[0]
    import shutil
    best_overall_path = Path(ckpt_dir) / "best_overall.pt"
    best_overall_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(winner["ckpt"], best_overall_path)

    # ── Save experiment summary ───────────────────────────────────────────
    summary = {
        "run_date":  datetime.now().isoformat(),
        "device":    str(device),
        "winner":    winner,
        "all_runs":  results,
    }
    summary_path = Path(log_dir) / "experiment_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print(f"Best run:  {winner['run_id']}")
    print(f"Best F1:   {winner['val_f1']:.4f}")
    print(f"Checkpoint: {best_overall_path}")
    print(f"Summary:   {summary_path}")
    print("="*60)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify script parses and shows help**

```bash
python train.py --help
```

Expected output:
```
usage: train.py [-h] [--config CONFIG] [--run-sweep] [--single]
```

---

## Task 9: Smoke test — single batch, no GPU required

**Files:**
- No new files

- [ ] **Step 1: Run single-batch smoke test**

```bash
python - <<'EOF'
import torch, yaml
from train import (VulnerabilityDataset, build_model, SupervisedContrastiveLoss,
                   RDropLoss, validate, CWE_8_CLASSES)
from pathlib import Path

cfg = yaml.safe_load(open("config.yaml"))
cache = torch.load(cfg["token_cache_file"], map_location="cpu", weights_only=False)
device = torch.device("cpu")

from torch.utils.data import DataLoader
ds = VulnerabilityDataset(cache, "val")
loader = DataLoader(ds, batch_size=4, shuffle=False)
batch = next(iter(loader))

counts = torch.bincount(ds.cwe_labels, minlength=8).float()
cw = (1.0 / counts.clamp(min=1))
cw = cw / cw.sum() * 8

run_cfg = {"lora_r": 4, "focal_gamma": 2.0}
model = build_model(cfg, run_cfg, cw, device)
model.train()

ids  = batch["input_ids"]
mask = batch["attention_mask"]
lbls = batch["cwe_label"]

out1 = model(ids, mask)
out2 = model(ids, mask)

scl  = SupervisedContrastiveLoss()
rdrop = RDropLoss()

loss_s = scl(torch.cat([out1["features"], out2["features"]]), torch.cat([lbls, lbls]))
loss_r = rdrop(out1["logits"], out2["logits"], lbls, cw)
loss   = loss_r + 0.3 * loss_s
loss.backward()
print(f"smoke test PASSED | loss={loss.item():.4f} | logits shape={out1['logits'].shape}")
EOF
```

Expected: `smoke test PASSED | loss=<float> | logits shape=torch.Size([4, 8])`

---

## Task 10: Full single-run on Colab (verify AMP + checkpointing)

**Files:**
- No new files

- [ ] **Step 1: Run single experiment (Colab T4)**

On Colab, upload all project files, then:

```bash
python train.py --single --config config.yaml
```

Expected output sequence:
```
Device: cuda | AMP: True
Loading token cache from datasets/token_cache/tokens_maxlen512.pt …
  Chunks: 14522 | Functions: 4085
RUN: lr2e-05_bs8_r16_g2.0
Trainable params: 1,184,648 / 125,830,280 (0.94%)
[lr2e-05_bs8_r16_g2.0] Epoch 1 | train_loss=... | val_f1=...
  ✓ New best F1=... → saved checkpoints/lr2e-05_bs8_r16_g2.0/best.pt
...
TRAINING COMPLETE
Best F1:   >0.69
Checkpoint: checkpoints/best_overall.pt
```

- [ ] **Step 2: Verify best_overall.pt exists and loads**

```python
import torch
ckpt = torch.load("checkpoints/best_overall.pt", map_location="cpu", weights_only=False)
print("Keys:", list(ckpt.keys()))
print("Best F1:", ckpt["val_f1"])
print("Run config:", ckpt["run_cfg"])
```

Expected: `Keys: ['epoch', 'val_f1', 'model_state_dict', 'run_cfg', 'base_cfg']`

---

## Task 11: Full sweep (optional — run after single confirms working)

- [ ] **Step 1: Launch full sweep on Colab**

```bash
python train.py --run-sweep --config config.yaml
```

Expected: 54 runs, final `experiment_summary.json` in `logs/`, `best_overall.pt` is winner checkpoint.

- [ ] **Step 2: Inspect summary**

```python
import json
summary = json.load(open("logs/experiment_summary.json"))
print(f"Winner: {summary['winner']['run_id']} | F1={summary['winner']['val_f1']}")
for r in summary["all_runs"][:5]:
    print(f"  {r['run_id']}: F1={r['val_f1']:.4f}")
```

---

## Verification Checklist

| Check | Command | Pass Condition |
|-------|---------|---------------|
| Config loads | `python -c "import yaml; yaml.safe_load(open('config.yaml'))"` | No error |
| Script parses | `python train.py --help` | Shows usage |
| Smoke test | See Task 9 | `smoke test PASSED` |
| Single run | `python train.py --single` | `best_overall.pt` created, F1 > 0.69 |
| Checkpoint valid | See Task 10 Step 2 | Keys present, val_f1 populated |
| Sweep summary | See Task 11 Step 2 | Winner printed, 54 entries in all_runs |

---

## Notes

- **Tokenization is NOT re-run by train.py.** Run `01_tokenization.ipynb` once to produce `tokens_maxlen512.pt` before invoking `train.py`.
- **AMP auto-disables on CPU/MPS** — `use_amp` flag in config only takes effect on CUDA devices.
- **EDAT doubles memory per batch** due to PGD inner loop. If OOM on T4, set `edat_enabled: false` in config.yaml or reduce `edat_steps` to 1.
- **Sweep is 54 combinations** — expect ~8–12h on T4. Run `--single` first to confirm pipeline works end-to-end.
- **best_overall.pt** is always the single best checkpoint regardless of which run produced it.
