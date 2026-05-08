# Trained Model Weights

This directory contains the fine-tuned weights for the Code Security Identifier models.

## Files

- **graphCodeBert.pt** (513 MB)
  - GraphCodeBERT model with LoRA fine-tuning
  - 7-class CWE classification head
  - Trained on unified dataset with statement-level vulnerability labels

- **dualEncoder.pt** (1 GB)
  - Fusion of GraphCodeBERT + VulBERTa
  - Dual-encoder architecture for higher accuracy
  - Also 7-class CWE classification

## Loading Weights

```python
import torch
from pathlib import Path

# Load weights for inference
weights_dir = Path("weights")
device = "cuda" if torch.cuda.is_available() else "cpu"

# GraphCodeBERT
graphcodebert_state = torch.load(
    weights_dir / "graphCodeBert.pt",
    map_location=device
)

# Dual Encoder
dualencoder_state = torch.load(
    weights_dir / "dualEncoder.pt",
    map_location=device
)
```

## Model Architecture

Both models expect:
- Input: Python code (tokenized via AutoTokenizer from "microsoft/graphcodebert-base")
- Output: 7-class logits for CWE classification
  - 0: CWE-077 (Command Injection)
  - 1: CWE-601 (Open Redirect)
  - 2: CWE-022 (Path Traversal)
  - 3: CWE-094 (Code Injection / RCE)
  - 4: CWE-089 (SQL Injection)
  - 5: CWE-352 (CSRF)
  - 6: CWE-079 (XSS)

## Training Details

- **Base Models**: GraphCodeBERT (microsoft/graphcodebert-base) + VulBERTa (claudios/VulBERTa-MLP-D2A)
- **Dataset**: ~4,000 deduplicated functions across 3 sources (Vudenc, Function-level, SecurityEval)
- **Training Framework**: PyTorch with LoRA (Low-Rank Adaptation)
- **Optimizer**: AdamW (lr=2e-5)
- **Loss**: Cross-entropy with class weighting for imbalanced CWE distribution

## Source

These weights are the output of Phase 2 training on the Code Security Identifier project:
- GitHub: https://github.com/a-elhaag/code-security-identifier
- HuggingFace Hub: https://huggingface.co/a-elhaag/code-security-identifier-weights

## License

MIT (same as main project)
