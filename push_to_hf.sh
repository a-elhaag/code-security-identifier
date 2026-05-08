#!/bin/bash
# Push trained weights to Hugging Face Hub
# Usage: bash push_to_hf.sh

set -e

REPO_NAME="a-elhaag/code-security-identifier-weights"
WEIGHTS_DIR="./weights"

echo "=========================================="
echo "Pushing weights to HuggingFace Hub"
echo "=========================================="
echo ""
echo "Repository: $REPO_NAME"
echo "Weights: $(ls -lh $WEIGHTS_DIR | tail -n +2 | awk '{print $9, "(" $5 ")"}')"
echo ""

# Check if weights exist
if [ ! -d "$WEIGHTS_DIR" ]; then
    echo "Error: weights directory not found at $WEIGHTS_DIR"
    exit 1
fi

if [ ! -f "$WEIGHTS_DIR/graphCodeBert.pt" ] || [ ! -f "$WEIGHTS_DIR/dualEncoder.pt" ]; then
    echo "Error: Required weight files not found (graphCodeBert.pt, dualEncoder.pt)"
    exit 1
fi

# Log in if needed
echo "Checking HuggingFace authentication..."
huggingface-cli whoami || {
    echo ""
    echo "Not authenticated. Running 'huggingface-cli login'..."
    echo "You'll need your HuggingFace API token from: https://huggingface.co/settings/tokens"
    huggingface-cli login
}

echo ""
echo "Uploading weights..."
echo ""

# Upload using git-based approach (preferred by HF for large files)
cd "$WEIGHTS_DIR" || exit 1

# Create a temporary commit message file
COMMIT_MSG="Upload trained model weights

- graphCodeBert.pt: LoRA-fine-tuned GraphCodeBERT model (513MB)
- dualEncoder.pt: Dual-encoder fusion model (1GB)

Uploaded via push_to_hf.sh script"

# Use HF Hub to push files
python3 << 'EOF'
import os
from huggingface_hub import HfApi, CommitOperationAdd

repo_name = "a-elhaag/code-security-identifier-weights"
weights_dir = "../weights"

api = HfApi()

print(f"Uploading weights from {weights_dir} to {repo_name}...")

operations = []
for filename in ["graphCodeBert.pt", "dualEncoder.pt"]:
    filepath = os.path.join(weights_dir, filename)
    if os.path.exists(filepath):
        operations.append(CommitOperationAdd(
            path_in_repo=filename,
            path_or_fileobj=filepath
        ))
        print(f"  ✓ Queued {filename}")
    else:
        print(f"  ✗ Not found: {filename}")

if operations:
    commit_info = api.create_commit(
        repo_id=repo_name,
        operations=operations,
        commit_message="Upload trained model weights (graphCodeBert.pt + dualEncoder.pt)"
    )
    print(f"\n✓ Upload successful!")
    print(f"  Commit: {commit_info.commit_url}")
else:
    print("No files to upload")
EOF

cd - > /dev/null

echo ""
echo "=========================================="
echo "Done!"
echo "=========================================="
