#!/bin/bash

echo "=== Setting up Detect Pretrain Code Environment ==="

# Create virtual environment
echo "Creating virtual environment..."
python -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

echo "✅ Setup complete!"
echo ""
echo "To run detection:"
echo "1. Activate environment: source venv/bin/activate"
echo "2. Run detection: python run.py --target_model gpt2 --ref_model distilgpt2 --data my_chunks.jsonl --key_name input"
echo ""
