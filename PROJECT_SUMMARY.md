# Code Security Identifier - Project Summary

## Overview

**Code Security Identifier (CSI)** is a static vulnerability detection tool for Python code using GraphCodeBERT and VulBERTa transformer models.

It detects **8 vulnerability types** (7 CWE classes + Unknown):
- CWE-077: Command Injection
- CWE-601: Open Redirect
- CWE-022: Path Traversal
- CWE-094: Code Injection / RCE
- CWE-089: SQL Injection
- CWE-352: CSRF
- CWE-079: XSS

## Getting Started (1 Command)

```bash
python setup.py
```

This single command:
1. Installs Python dependencies
2. Downloads HuggingFace models (~750MB)
3. Launches Streamlit app at `http://localhost:8501`

**No Docker. No complex setup. Just Python.**

## Architecture

```
User Input (Python Code)
    ↓
AutoTokenizer (GraphCodeBERT vocab)
    ↓
Transformer Encoder (GraphCodeBERT or VulBERTa)
    ↓
[CLS] Token Representation
    ↓
7-Class CWE Head (Fine-tuned LoRA)
    ↓
Output (CWE Prediction + Confidence Score)
```

### Models

| Model | Type | Size | Purpose |
|-------|------|------|---------|
| **GraphCodeBERT-base** | Encoder | 350 MB | Code understanding via AST + data/control flow |
| **VulBERTa** | Encoder | 400 MB | Vulnerability-specific pre-training |
| **Fine-tuned Weights** | LoRA | 1.5 GB | 7-class CWE classification head |

## Dataset

- **Sources**: Vudenc, Function-level dataset, SecurityEval
- **Total**: ~4,000 deduplicated functions
- **Labels**: 8 CWE classes + statement-level vulnerability annotations
- **Format**: JSONL with cross-reference mappings
- **Splits**: 90% train, 10% validation

## Project Phases

### Phase 1: Complete ✅
- Dataset collection, preprocessing, deduplication
- Model architecture (GraphCodeBERT + LoRA + 7-class head)
- Unified dataset with cross-reference indices

### Phase 2: Complete ✅
- Supervised fine-tuning on unified dataset
- Model evaluation on held-out validation split
- Trained weights available at HuggingFace Hub

### Phase 3: Planned 🔮
- Extended to statement-level binary classification
- Supervised contrastive learning (optional)
- Production deployment with API

## File Structure

```
code-security-identifier/
├── setup.py                    # 👈 Start here (one command setup)
├── run_setup.py                # Downloads HF models
├── app.py                      # Streamlit web UI
├── requirements.txt            # Python dependencies
├── README.md                   # Quick reference
├── SETUP_GUIDE.md              # Detailed setup guide
├── PROJECT_SUMMARY.md          # This file
├── push_to_hf.sh               # Upload weights to HuggingFace Hub
│
├── weights/                    # Fine-tuned model weights
│   ├── graphCodeBert.pt        # LoRA-tuned GraphCodeBERT (513 MB)
│   ├── dualEncoder.pt          # Dual-encoder fusion (1 GB)
│   └── README.md               # Weight documentation
│
├── datasets/                   # Training datasets
│   ├── UNIFIED.jsonl           # 4,000+ training examples
│   ├── UNIFIED_mappings.json   # Cross-reference indices
│   └── raw/                    # Original dataset files
│
├── notebooks/                  # Preprocessing & analysis
│   ├── 00_setup.ipynb
│   ├── 01_dataset_pipeline_local.ipynb
│   ├── 02_validation_analysis.ipynb
│   └── 03_unified_mapping.ipynb
│
└── .git/                       # Version control
```

## Key Features

### Simple Setup
- Single `python setup.py` command
- Automatic dependency installation
- Automatic model downloading
- Works on any OS (Windows, Mac, Linux)

### Production-Ready
- ~4,000 deduplicated training examples
- Fine-tuned on real vulnerability data
- Confidence scores for predictions
- Extensible architecture

### Well-Documented
- Detailed setup guide (SETUP_GUIDE.md)
- Comprehensive notebooks for reproduction
- Clear model architecture documentation
- Weight documentation

## Technology Stack

- **Framework**: PyTorch 2.0+
- **Models**: Transformers (HuggingFace)
- **Fine-tuning**: LoRA (Parameter-Efficient)
- **Web UI**: Streamlit
- **Data**: JSONL format with cross-reference mappings
- **Version Control**: Git + GitHub

## Performance

- **Training Data**: ~4,000 functions
- **CWE Classes**: 8 (7 specific + Unknown)
- **Model Size**: 124M parameters (GraphCodeBERT-base)
- **Trainable Params**: ~300K via LoRA (0.24%)
- **Inference Speed**: ~2-3 seconds per function (GPU-accelerated)

## Team & Contributors

10-member team from diverse backgrounds:
- **Anas Elhaag** (Lead)
- Farida Hassan, Hend Elhout, Hesham Mahmoud
- Jomana Mekheimar, Menna Amr, Menna Reda
- Sohaila Tamer, Youstina Adel

## Usage Examples

### Basic (Web UI)
```bash
python setup.py
# Open http://localhost:8501
# Paste code → Click "Analyze" → See vulnerabilities
```

### Advanced (Python API)
```python
import torch
from transformers import AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")

# Load your code
code = """
import os
os.system(user_input)  # CWE-077: Command Injection
"""

# Tokenize & predict (with your fine-tuned model)
encoded = tokenizer(code, return_tensors="pt")
logits = model(**encoded)
```

## Contributing

### Local Development
```bash
# Clone
git clone https://github.com/a-elhaag/code-security-identifier.git
cd code-security-identifier

# Setup
python setup.py

# Modify app.py or models, then restart Streamlit
```

### Pushing Weights to HuggingFace
```bash
bash push_to_hf.sh
```

## Deployment

### Docker (Optional)
For containerized deployment, build your own Dockerfile:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
RUN python run_setup.py
CMD ["streamlit", "run", "app.py"]
```

### Cloud (Hugging Face Spaces)
You can deploy directly to HF Spaces:
1. Create a space at https://huggingface.co/spaces
2. Upload this repository
3. Add a `requirements.txt` (already included)
4. HF will auto-launch with `streamlit run app.py`

## Resources

- **GitHub**: https://github.com/a-elhaag/code-security-identifier
- **HuggingFace Weights**: https://huggingface.co/a-elhaag/code-security-identifier-weights
- **Paper References**:
  - GraphCodeBERT (Guo et al., 2020)
  - LoRA (Hu et al., 2021)
  - VulBERTa (Nix et al., 2021)

## Known Limitations

1. **CWE Coverage**: Currently 7 common classes; other CWEs map to "Unknown"
2. **Training Data**: Limited to ~4,000 functions; performance scales with more data
3. **Code Context**: Models see function-level code; larger files may need truncation
4. **Language**: Python-specific; not directly applicable to other languages

## Next Steps

- [ ] Statement-level binary classification (vulnerable vs. safe per line)
- [ ] API endpoint for integration with CI/CD pipelines
- [ ] Interactive explanations (which tokens triggered vulnerability detection?)
- [ ] Extended CWE coverage (beyond 7 classes)
- [ ] Adversarial robustness evaluation

## License

MIT

---

**Last Updated**: May 2026  
**Status**: Phase 2 Complete | Phase 3 Planning  
**Questions?** Open an issue on GitHub!
