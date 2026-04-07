# Code Security Identifier (CSI)

**Python vulnerability detection system targeting 75%+ F1 on DetectVul benchmark.**

## Overview

CSI performs simultaneous multi-task vulnerability detection on Python code:
- **Binary classification**: vulnerable vs. secure
- **CWE classification**: maps to specific weakness types
- **Severity scoring**: CVSS-style risk quantification
- **Line-level localization**: pinpoints exact vulnerable code statements

Built by 9-person academic team using GraphCodeBERT + LoRA fine-tuning, supervised contrastive learning (SCL-CVD), and adversarial training (EDAT).

## Architecture

```
Input (Python code)
    ↓
GraphCodeBERT encoder (pretrained, LoRA fine-tuned)
    ├→ SCL-CVD supervised contrastive loss
    ├→ EDAT adversarial training
    ↓
Shared representation
    ├→ Binary head (vulnerability detection)
    ├→ CWE head (21 classes)
    ├→ Severity head (CVSS regression)
    └→ Localization head (statement-level)
    ↓
Output (4 predictions)
```

**Key design choice**: Transformers > GNNs for Python vulnerability detection per literature (Transformers excel at code understanding; GNNs require explicit graph construction which is slower on Python ASTs).

**Optional Phase 2**: VulBERTa dual-encoder fusion if F1 > 70%.

## Data

### Datasets
- **DetectVul/Vudenc** (~10k samples): Python-native, statement-level labels
- **DetectVul/CVEFixes** (~11k samples): Real CVE fixes with CWE mapping
- **dataset.jsonl** (121 samples): SecurityEval test-only set — **never use for training**
- **plain_* files**: Direct line-level supervision (rare, valuable for localization head)

Total: ~21,571 functions with statement-level ground truth.

### Critical Preprocessing
- Vudenc's `source` field is flattened; use `sourceWithComments` instead
- Split by function boundary, not by file
- Preserve statement indices for localization labels
- Exclude `dataset.jsonl` from all training/validation splits

## Team & Phases

**9 members**: Anas Elhaag (lead), Farida Hassan, Hend Elhout, Hesham Mahmoud, Jomana Mekheimar, Menna Amr, Menna Reda, Sohaila Tamer, Youstina Adel.

Work distributed across all phases; no role silos.

### Timeline
- **Week 1**: Dataset pipeline + baseline (COMPLETE)
- **Week 2**: SCL-CVD + EDAT training
- **Week 3**: Full training run + Streamlit deployment
- **Week 4**: Report + delivery

## Setup

### Prerequisites
- Python 3.9+
- PyTorch 2.0+
- Transformers 4.30+
- Google Colab (recommended) or local GPU

### Google Colab
```bash
python colab_setup.py
```
Initializes environment, mounts Google Drive, downloads datasets to `CSI_Project/datasets/`.

### Local
```bash
pip install -r requirements.txt
python src/download_datasets.py --output_dir ./data
```

## Key Files

```
github.com/a-elhaag/code-security-identifier/
├── src/
│   ├── dataset.py           # Dataset loading, preprocessing
│   ├── model.py             # GraphCodeBERT + 4 heads
│   ├── train.py             # SCL-CVD + EDAT training loop
│   ├── evaluate.py          # F1, precision, recall, etc.
│   └── deploy.py            # Streamlit app
├── notebooks/
│   ├── Complete_Dataset_Pipeline.ipynb  # EDA + preprocessing (complete)
│   └── colab_notebook.ipynb             # Colab training entrypoint
├── data/
│   ├── raw/                 # HuggingFace downloads (excluded from git)
│   ├── processed/           # Train/val/test splits (excluded from git)
│   └── plain_samples/       # Line-level annotations
├── checkpoints/             # Model checkpoints (excluded from git)
├── results/                 # Metrics, confusion matrices
├── colab_setup.py           # Colab environment init
└── COLAB_SETUP_GUIDE.md     # User-facing Colab instructions
```

## Benchmark Target

**DetectVul F1: 74.47%** (baseline transformer models)

Goal: **75%+** on Python-native test set (held-out functions from Vudenc + CVEFixes).

## Quick Start

### 1. Load Dataset
```python
from src.dataset import VulnerabilityDataset

dataset = VulnerabilityDataset(
    vudenc_path="data/raw/vudenc.jsonl",
    cvefix_path="data/raw/cvefix.jsonl",
    split="train"
)
```

### 2. Train Model
```bash
python src/train.py \
  --model_name graphcodebert-base \
  --batch_size 16 \
  --epochs 10 \
  --use_scl_cvd \
  --use_edat
```

### 3. Deploy
```bash
streamlit run src/deploy.py
```

## Research References

Core papers:
- **GraphCodeBERT** (Guo et al., 2020): Code understanding via AST + data/control flow
- **SCL-CVD** (Wang et al., 2024): Supervised contrastive loss for vulnerability detection
- **EDAT** (Chen et al., 2025): Adversarial training for robustness
- **LineVul** (Fu & Tantithamthavorn, MSR 2022): Statement-level localization
- **ReGVD** (Nguyen et al., ICSE 2022): Revisiting graph-based vulnerability detection

## Collaboration

- **Task tracking**: Notion (database with dependencies, priority, assignee per task)
- **Code**: GitHub (main branch protected; PR review required)
- **Data + checkpoints**: Google Drive (`CSI_Project/` shared folder)
- **Communication**: Async updates; weekly sync on progress

## Known Issues

- Vudenc `source` field requires special handling (`sourceWithComments`); raw `source` is flattened
- Image-based PDFs in research papers need Tesseract OCR for extraction
- Transformers require GPU for real-time inference (~2 sec/function on V100)

## Next Steps

1. Train Phase 2 (Week 2): Implement SCL-CVD + EDAT losses
2. Full training run (Week 3): 10 epochs, measure F1 on held-out test
3. Deploy to Streamlit (Week 3): Real-time detection API
4. Phase 2 optional decision: If F1 > 70%, integrate VulBERTa dual encoder
5. Final report (Week 4): Results, ablations, comparison to DetectVul baseline

---

**Repo**: `github.com/a-elhaag/code-security-identifier`  
**Last updated**: April 2026  
**Status**: Phase 1 complete (dataset pipeline); Phase 2 in progress