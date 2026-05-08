# Setup Guide

This project uses a simplified, automated setup process. Just run one command and everything is installed and ready to go.

## Quick Start

```bash
python setup.py
```

That's it! The script will:
1. Install Python dependencies from `requirements.txt`
2. Download HuggingFace models (~750MB) to `~/.cache/huggingface/hub/`
3. Launch the Streamlit app at `http://localhost:8501`

## What Happens Under the Hood

### Step 1: Install Dependencies (`pip install -r requirements.txt`)

Installs all required Python packages:
- **streamlit**: Web UI framework
- **torch**: Deep learning framework
- **transformers**: HuggingFace model loading
- **peft**: Parameter-efficient fine-tuning (for LoRA models)
- **numpy, pandas**: Data processing

### Step 2: Download Models (`python run_setup.py`)

Downloads two pre-trained models from HuggingFace Hub:

| Model | Size | Purpose |
|-------|------|---------|
| **microsoft/graphcodebert-base** | ~350 MB | Code understanding via AST + data/control flow |
| **claudios/VulBERTa-MLP-D2A** | ~400 MB | Vulnerability-specific pre-training |

Models are cached at: `~/.cache/huggingface/hub/`

**Why cache locally?** 
- Avoids re-downloading on subsequent runs
- Works offline after first download
- Shared across projects if you use HuggingFace models elsewhere

### Step 3: Launch App (`streamlit run app.py`)

Starts the Streamlit server at `http://localhost:8501`

The app loads:
- Base models from cache
- Fine-tuned weights from `weights/` directory
- Provides interactive UI for code vulnerability detection

## File Structure

```
code-security-identifier/
├── setup.py                          # Main entry point (RUN THIS)
├── run_setup.py                      # Downloads HF models
├── app.py                            # Streamlit app
├── requirements.txt                  # Python dependencies
├── weights/                          # Fine-tuned model weights
│   ├── graphCodeBert.pt (513 MB)
│   └── dualEncoder.pt (1 GB)
└── datasets/                         # Training data
    ├── UNIFIED.jsonl                 # 4,000+ labeled examples
    └── UNIFIED_mappings.json         # Cross-reference indices
```

## System Requirements

- **Python**: 3.9 or newer
- **RAM**: 8+ GB (for inference)
- **GPU**: Optional (CUDA 11.8+ recommended, but CPU works too)
- **Disk**: ~2 GB (for models + weights)

## Troubleshooting

### "ModuleNotFoundError: No module named 'transformers'"

The dependencies weren't installed. Try again:
```bash
python -m pip install -r requirements.txt
```

### "Failed to download model from HuggingFace"

This can happen if:
- **No internet connection**: Download only works with internet
- **HuggingFace is down**: Try again later
- **Disk space**: Ensure you have 2+ GB free

Check your connection and try:
```bash
python run_setup.py
```

### App won't start after models downloaded

Try launching manually:
```bash
streamlit run app.py
```

If it still fails, check that models are cached:
```bash
ls ~/.cache/huggingface/hub/
```

Should see two folders:
- `models--microsoft--graphcodebert-base/`
- `models--claudios--VulBERTa-MLP-D2A/`

## Manual Setup (Advanced)

If you prefer to run commands separately:

```bash
# 1. Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download models
python run_setup.py

# 4. Launch app
streamlit run app.py
```

## Pushing Weights to HuggingFace (For Maintainers)

To upload trained weights to your HuggingFace Hub repo:

```bash
bash push_to_hf.sh
```

This will:
1. Check your HuggingFace authentication
2. Upload `weights/graphCodeBert.pt` and `weights/dualEncoder.pt`
3. Create a commit on your model repository

**First time?** You'll be prompted to authenticate:
```bash
huggingface-cli login
# Paste your API token from: https://huggingface.co/settings/tokens
```

## What If I Want to Use Custom Models?

Edit `run_setup.py` to change the model IDs:

```python
MODELS = [
    {
        "name": "Your Model Name",
        "model_id": "your-org/your-model-id",  # Change this
        "tokenizer_fn": AutoTokenizer.from_pretrained,
        "model_fn": AutoModel.from_pretrained,
    },
]
```

Then run `python setup.py` again.

## Next Steps

After the app launches at `http://localhost:8501`:

1. **Paste Python code** into the editor
2. **Click "Analyze"** to detect vulnerabilities
3. **View results** showing detected CWE classes

The app will display:
- **Detected CWE type** (Command Injection, SQL Injection, XSS, etc.)
- **Confidence score** (0-100%)
- **Explanation** of the vulnerability

## Environment Variables (Optional)

You can customize behavior via environment variables:

```bash
# Set HuggingFace cache location (default: ~/.cache/huggingface/hub/)
export HF_HOME=/custom/hf/cache

# Set HuggingFace token (if using private models)
export HF_TOKEN=hf_xxxxxxxxxxxxxxxx

# Then run:
python setup.py
```

## Questions?

- **GitHub Issues**: https://github.com/a-elhaag/code-security-identifier/issues
- **HuggingFace Model Card**: https://huggingface.co/a-elhaag/code-security-identifier-weights
