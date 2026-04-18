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
Tokenize (AutoTokenizer) → Statement-level labels + CWE ID
    ↓
GraphCodeBERT-base encoder (LoRA fine-tuned)
    │ (Trainable: ~300K / 124M params = 0.24% via LoRA)
    ↓
[CLS] token representation
    ↓
CWE Classification Head (7 classes)
    ├→ CWE-077 (command injection)
    ├→ CWE-601 (open redirect)
    ├→ CWE-022 (path traversal)
    ├→ CWE-094 (RCE / code injection)
    ├→ CWE-089 (SQL injection)
    ├→ CWE-352 (CSRF)
    └→ CWE-079 (XSS)
    ↓
Output (7-class logits + cross-entropy loss)
```

**Phase 1** (Current):
- ✅ Dataset pipeline: preprocessing, deduplication, quality validation
- ✅ LoRA-wrapped model: GraphCodeBERT + lightweight CWE head
- ✅ Unified dataset: cross-reference mappings for train/val splits
- ⏳ Training loop: supervised fine-tuning on unified dataset

**Phase 2 (Planned)**: Extend to statement-level binary classification (vulnerable vs. safe) with supervised contrastive loss.

## Data

### Datasets
- **Vudenc** (~1,775 functions): Python-native, statement-level labels
- **Function-level dataset**: Buggy and fixed code pairs with CWE annotations
- **SecurityEval** (~100 functions): Additional test set
- **dataset.jsonl** (121 samples): Reserved for final test-only evaluation

Total after preprocessing: ~4,000+ deduplicated functions with CWE labels.

### Preprocessing Pipeline
3-stage Jupyter notebook pipeline (runs locally or in Colab with Google Drive):

**Stage 1: `00_setup.ipynb`** → Install dependencies + download HuggingFace models
- Downloads `microsoft/graphcodebert-base` and `microsoft/codebert-base`
- Caches to Google Drive (`CSI_Project/models/`) to avoid re-downloading

**Stage 2: `01_dataset_pipeline_local.ipynb`** → Combine + deduplicate + split
- Loads 3 raw datasets
- **Deduplication**: SHA256 hash of (code_lines, label, cwe_id) tuple; removes exact duplicates from function-level dataset
- **CWE Extraction** (2-strategy):
  - Strategy 1: Direct regex `CWE-\d+` from metadata
  - Strategy 2: Keyword fallback (e.g., "sql injection" → CWE-089, "xss" → CWE-079)
- **Output**: FINAL_train.jsonl (~90% split), FINAL_val.jsonl (~10% split)

**Stage 3: `02_validation_analysis.ipynb`** → Quality gate enforcement
- Validates record structure (required fields: lines, label, cwe_id, dataset_source, etc.)
- Enforces quality thresholds:
  - Duplicates: <15% (reduced from 16.64% via dedup)
  - Unknown CWE: <35% (reduced from 61.70% via keyword extraction)
  - Vulnerable functions: 3–80%
  - Vulnerable statements: 1–40%
- Outputs: Quality metrics JSON, final PASS/FAIL status

**Stage 4: `03_unified_mapping.ipynb`** → Unify + verify + model definition
- Loads FINAL_train.jsonl + FINAL_val.jsonl
- Validates identical record structure across splits
- Normalizes records with metadata: record_id, split_origin, global_id, num_statements, num_vulnerable
- Merges both splits → UNIFIED.jsonl
- Creates cross-reference indices: UNIFIED_mappings.json (by_split, by_source, by_cwe, by_vulnerability)
- Saves metadata: UNIFIED_metadata.json
- Instantiates GraphCodeBERTLoRACWEModel with 7-class CWE head

### Record Structure
```json
{
  "lines": [...],                    # code lines as list of strings
  "raw_lines": [...],                # raw code (comments, whitespace preserved)
  "label": [0, 1, 0, ...],          # per-statement vulnerability labels (binary)
  "type": ["statement", ...],        # statement type
  "cwe_id": "CWE-089",              # function-level CWE classification
  "dataset_source": "vudenc",        # which dataset (vudenc, function-level, security-eval)
  "record_id": 42,                   # unique ID within split
  "split_origin": "train",           # "train" or "val"
  "global_id": 1234,                 # unique ID across unified dataset
  "num_statements": 5,               # total statements in function
  "num_vulnerable": 2,               # count of vulnerable statements
  "is_vulnerable": true              # function-level vulnerability flag
}
```

## Team & Phases

**10 members**: Anas Elhaag (lead), Farida Hassan, Hend Elhout, Hesham Mahmoud, Jomana Mekheimar, Menna Amr, Menna Reda, Sohaila Tamer, Thjumana, Youstina Adel.

### Phase Status
- **Phase 1 (Data + Model Setup)**: ✅ COMPLETE
  - Dataset pipeline: collect, preprocess, deduplicate, validate
  - Model architecture: LoRA-wrapped GraphCodeBERT + 7-class CWE head
  - Unified dataset: cross-referenced train/val splits ready for training
  
- **Phase 2 (Training + Evaluation)**: ⏳ IN PROGRESS
  - Supervised fine-tuning on unified dataset
  - Evaluate F1, precision, recall on held-out validation split
  - Optional: extend to statement-level binary classification with SCL-CVD loss

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

### Notebooks (Main Entry Points)
- **`00_setup.ipynb`**: Install dependencies, download HuggingFace models, cache to Google Drive
- **`01_dataset_pipeline_local.ipynb`**: Load raw datasets → deduplicate → split → FINAL_train.jsonl + FINAL_val.jsonl
- **`02_validation_analysis.ipynb`**: Validate structure, enforce quality thresholds, output metrics
- **`03_unified_mapping.ipynb`**: 
  - Unify train/val splits → UNIFIED.jsonl
  - Create cross-reference mappings → UNIFIED_mappings.json
  - Define GraphCodeBERTLoRACWEModel + smoke test
  - Save metadata → UNIFIED_metadata.json

### Project Structure
```
code-security-identifier/
├── 00_setup.ipynb                              # Environment + model download
├── 01_dataset_pipeline_local.ipynb             # Data processing + dedup + split
├── 02_validation_analysis.ipynb                # Quality gate enforcement
├── 03_unified_mapping.ipynb                    # Unification + model definition
├── README.md                                   # This file
│
├── datasets/                                   # Outputs (created by notebooks)
│   ├── FINAL_train.jsonl                       # 90% train split
│   ├── FINAL_val.jsonl                         # 10% val split
│   ├── UNIFIED.jsonl                           # Train + val merged
│   ├── UNIFIED_mappings.json                   # Cross-reference indices
│   ├── UNIFIED_metadata.json                   # Dataset statistics
│   └── raw/                                    # Raw input data
│       ├── dataset.jsonl
│       ├── DataTable.csv
│       ├── plain_command_injection/
│       ├── plain_open_redirect/
│       ├── plain_path_disclosure/
│       ├── plain_remote_code_execution/
│       ├── plain_sql/
│       ├── plain_xsrf/
│       └── plain_xss/
│
├── raw/                                        # Alternate raw data location (local)
│   └── [raw dataset files]
│
├── models/                                     # Cached HuggingFace models (Google Drive)
│   ├── microsoft--graphcodebert-base/
│   └── microsoft--codebert-base/
│
└── .git/                                       # Version control
```

## Benchmark Target

**Phase 1 Goal** (Current): Build complete data pipeline + validate model architecture
- ✅ Deduplicated dataset: <15% duplicates, <35% unknown CWE  
- ✅ LoRA model: 0.24% trainable params (300K / 124M)
- ✅ Unified dataset: cross-referenced train/val ready for fine-tuning

**Phase 2 Goal**: Fine-tune on unified dataset
- Target: Early stopping when validation loss plateaus
- Evaluate via CWE classification accuracy on held-out val split
- Option: Extend to statement-level binary classification with supervised contrastive loss

## Quick Start

### Option 1: Local Development
```bash
# Clone repo
git clone https://github.com/a-elhaag/code-security-identifier.git
cd code-security-identifier

# Run notebooks in order
jupyter notebook 00_setup.ipynb              # Install + download models
jupyter notebook 01_dataset_pipeline_local.ipynb    # Process data
jupyter notebook 02_validation_analysis.ipynb       # Validate
jupyter notebook 03_unified_mapping.ipynb           # Unify + define model
```

### Option 2: Google Colab (Recommended)
1. Upload notebooks to Colab
2. Mount Google Drive (notebooks auto-detect paths)
3. Run cells sequentially:
   - `00_setup.ipynb` → downloads models to `/content/drive/MyDrive/CSI_Project/models/`
   - `01_dataset_pipeline_local.ipynb` → outputs to `/content/drive/MyDrive/CSI_Project/datasets/`
   - `02_validation_analysis.ipynb` → enforces quality gates
   - `03_unified_mapping.ipynb` → creates UNIFIED.jsonl + model

### Load Unified Dataset & Model
```python
import json
from pathlib import Path

# Load unified records
def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

unified = read_jsonl("datasets/UNIFIED.jsonl")
print(f"Loaded {len(unified):,} records")

# Example: get all vulnerable functions from train split
vulnerable_train = [
    r for r in unified 
    if r["split_origin"] == "train" and r["is_vulnerable"]
]
print(f"Vulnerable in train: {len(vulnerable_train):,}")

# Load cross-reference mappings
with open("datasets/UNIFIED_mappings.json") as f:
    mappings = json.load(f)

# Get CWE-089 (SQL injection) records
sql_injection = [unified[i] for i in mappings["by_cwe"].get("CWE-089", [])]
print(f"SQL injection functions: {len(sql_injection):,}")
```

### Train the Model
```python
import torch
from pathlib import Path
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset

# Use the model defined in 03_unified_mapping.ipynb
device = "cuda" if torch.cuda.is_available() else "cpu"
model = GraphCodeBERTLoRACWEModel().to(device)
tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")

# Create PyTorch Dataset from unified records
class VulnerabilityDataset(Dataset):
    def __init__(self, records, tokenizer, max_length=256):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, idx):
        record = self.records[idx]
        code = " ".join(record["lines"])
        cwe_id = record["cwe_id"]
        target = map_cwe_to_target(cwe_id)  # Returns 0-6 or -1
        
        encoded = self.tokenizer(
            code,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(target, dtype=torch.long)
        }

# Training loop (example)
train_records = [r for r in unified if r["split_origin"] == "train"]
train_dataset = VulnerabilityDataset(train_records, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
model.train()

for epoch in range(3):
    for batch in train_loader:
        optimizer.zero_grad()
        output = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["labels"].to(device)
        )
        loss = output["loss"]
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}: loss={loss.item():.4f}")
```

## Research References

Core papers:
- **GraphCodeBERT** (Guo et al., 2020): Pre-trained encoder for code understanding via AST + data/control flow
- **LoRA** (Hu et al., 2021): Low-rank adaptation for efficient fine-tuning of large models
- **PEFT** (Mangrulkar et al., 2023): Parameter-efficient fine-tuning library (PyTorch)
- **CWE Classification with Transformers** (various): Using [CLS] token + MLP head for multi-class vulnerability type prediction
- **LineVul** (Fu & Tantithamthavorn, MSR 2022): Statement-level localization baseline for code vulnerability detection

## Collaboration

- **Task tracking**: Notion (database with dependencies, priority, assignee per task)
- **Code**: GitHub (main branch protected; PR review required)
- **Data + checkpoints**: Google Drive (`CSI_Project/` shared folder)
- **Communication**: Async updates; weekly sync on progress

## Known Issues & Limitations

- **CWE Extraction**: Some raw datasets lack explicit CWE labels; keyword fallback helps but may introduce noise
- **Duplicate Detection**: SHA256 hashing assumes identical code + label + CWE = duplicate; edge cases (same code, different intent) not handled
- **7-class Subset**: Many raw records have CWE types outside the 7-class target (CWE-077, 601, 022, 094, 089, 352, 079); these map to label=-1 (ignored loss)
- **GPU Memory**: GraphCodeBERT backbone requires ~6 GB VRAM; batch_size=16 is typical for V100/A100
- **Google Drive Integration**: File sync delays (~30s) between notebook saves and Drive visibility

## Next Steps

1. **Implement training loop** (Week 2):
   - Create Trainer / custom train loop with optimizer + scheduler
   - Monitor train/val loss, CWE classification accuracy
   - Save best checkpoint based on val metrics

2. **Extend to binary classification** (Week 2–3):
   - Add statement-level binary head (vulnerable vs. safe per line)
   - Implement supervised contrastive loss (SCL-CVD) if needed
   - Evaluate on mixed tasks (CWE + binary classification)

3. **Evaluation & deployment** (Week 3–4):
   - Benchmark final model on held-out val split
   - Create Streamlit app for real-time CWE prediction
   - Generate final report with ablations and comparisons

4. **Optional Phase 2** (if F1 > 75%):
   - Integrate dual-encoder architecture (GraphCodeBERT + CodeBERT)
   - Adversarial training for robustness
   - Semantic-aware contrastive learning

---

**Repo**: `github.com/a-elhaag/code-security-identifier`  
**Last updated**: April 2026  
**Status**: Phase 1 complete (data pipeline + model setup); Phase 2 ready (training loop)