#!/usr/bin/env python3
"""
Setup script to download models and prepare the environment for running the app.
Run once with: python run_setup.py
Then run the app with: streamlit run app.py
"""

import os
import sys
from pathlib import Path
from transformers import AutoModel, AutoTokenizer, RobertaModel, RobertaTokenizer

# Model configurations
MODELS = [
    {
        "name": "GraphCodeBERT",
        "model_id": "microsoft/graphcodebert-base",
        "tokenizer_fn": RobertaTokenizer.from_pretrained,
        "model_fn": AutoModel.from_pretrained,
    },
    {
        "name": "VulBERTa",
        "model_id": "claudios/VulBERTa-MLP-D2A",
        "tokenizer_fn": AutoTokenizer.from_pretrained,
        "model_fn": RobertaModel.from_pretrained,
    },
]


def setup_models():
    """Download and cache all required models."""
    print("=" * 60)
    print("Code Security Identifier - Model Setup")
    print("=" * 60)
    print()

    hf_home = Path.home() / ".cache" / "huggingface" / "hub"
    print(f"Models will be cached at: {hf_home}")
    print()

    for model_config in MODELS:
        name = model_config["name"]
        model_id = model_config["model_id"]
        print(f"Downloading {name} ({model_id})...")

        try:
            # Download tokenizer
            print(f"  → Tokenizer...", end=" ", flush=True)
            model_config["tokenizer_fn"](model_id)
            print("✓")

            # Download model
            print(f"  → Model weights...", end=" ", flush=True)
            model_config["model_fn"](model_id)
            print("✓")

            print(f"✓ {name} cached successfully")
            print()

        except Exception as e:
            print(f"✗ Failed to download {name}: {e}", file=sys.stderr)
            return False

    print("=" * 60)
    print("✓ All models downloaded and cached!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Run the Streamlit app:")
    print("     streamlit run app.py")
    print()
    print("  2. Or if you prefer using streamlit with custom options:")
    print("     streamlit run app.py --logger.level=error")
    print()
    return True


if __name__ == "__main__":
    success = setup_models()
    sys.exit(0 if success else 1)
