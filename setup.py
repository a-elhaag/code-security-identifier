#!/usr/bin/env python3
"""
Complete setup script for Code Security Identifier
Installs dependencies, downloads models, and runs the app
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"📦 {description}")
    print(f"{'='*60}")
    try:
        subprocess.run(cmd, check=True, shell=True)
        print(f"✓ {description} - Done!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} - Failed!")
        print(f"Error: {e}")
        return False

def main():
    print("""
╔════════════════════════════════════════════════════════════════╗
║    Code Security Identifier - Complete Setup                  ║
║    This script will:                                           ║
║    1. Install Python dependencies                             ║
║    2. Download HuggingFace models                             ║
║    3. Launch the Streamlit app                                ║
╚════════════════════════════════════════════════════════════════╝
    """)

    # Step 1: Install dependencies
    if not run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing Python dependencies"
    ):
        print("\n❌ Failed to install dependencies. Exiting.")
        sys.exit(1)

    # Step 2: Download models
    if not run_command(
        f"{sys.executable} run_setup.py",
        "Downloading HuggingFace models (~750MB)"
    ):
        print("\n❌ Failed to download models. Exiting.")
        sys.exit(1)

    # Step 3: Launch app
    print(f"\n{'='*60}")
    print("🚀 Launching Streamlit app...")
    print(f"{'='*60}")
    print("\n✓ App will open at: http://localhost:8501")
    print("✓ Press Ctrl+C to stop the app\n")

    os.system(f"{sys.executable} -m streamlit run app.py")

if __name__ == "__main__":
    main()
