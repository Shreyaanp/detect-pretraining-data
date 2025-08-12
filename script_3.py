# Create the remaining support files

# 4. Create sample data file
import json

sample_data = [
    {"input": "The quick brown fox jumps over the lazy dog", "label": 0},
    {"input": "To be or not to be, that is the question", "label": 0},
    {"input": "Machine learning models require careful hyperparameter tuning for optimal performance", "label": 0},
    {"input": "This is a completely unique sentence created specifically for testing purposes", "label": 0},
    {"input": "Python is a popular programming language for data science and artificial intelligence", "label": 0}
]

with open('my_chunks.jsonl', 'w') as f:
    for item in sample_data:
        f.write(json.dumps(item) + '\n')
print("✅ Created my_chunks.jsonl")

# 5. Create requirements.txt
requirements = """torch>=1.9.0
transformers>=4.20.0
datasets>=2.0.0
numpy>=1.21.0
matplotlib>=3.5.0
scikit-learn>=1.0.0
tqdm>=4.60.0
openai>=0.28.0"""

with open('requirements.txt', 'w') as f:
    f.write(requirements)
print("✅ Created requirements.txt")

# 6. Create setup script for Linux/Mac
setup_sh = """#!/bin/bash

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
"""

with open('setup.sh', 'w') as f:
    f.write(setup_sh)
print("✅ Created setup.sh")

# 7. Create setup script for Windows
setup_bat = """@echo off

echo === Setting up Detect Pretrain Code Environment ===

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo Activating virtual environment...
call venv\\Scripts\\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo Installing dependencies...
pip install -r requirements.txt

echo ✅ Setup complete!
echo.
echo To run detection:
echo 1. Activate environment: venv\\Scripts\\activate.bat
echo 2. Run detection: python run.py --target_model gpt2 --ref_model distilgpt2 --data my_chunks.jsonl --key_name input
echo.
pause
"""

with open('setup.bat', 'w') as f:
    f.write(setup_bat)
print("✅ Created setup.bat")

print("\nAll core files created! Final step: instructions...")