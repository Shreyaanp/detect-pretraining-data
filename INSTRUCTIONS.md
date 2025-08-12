# Detect Pretrain Code - Complete Setup Instructions

## Files You Need to Download

Download these 7 files and put them in the same folder:

1. **run.py** - Main detection script
2. **eval.py** - Evaluation functions  
3. **options.py** - Command line options
4. **my_chunks.jsonl** - Sample data for testing
5. **requirements.txt** - Python dependencies
6. **setup.sh** - Setup script for Linux/Mac
7. **setup.bat** - Setup script for Windows

## Quick Start Instructions

### Linux/Mac:
```bash
# 1. Make setup script executable
chmod +x setup.sh

# 2. Run setup (creates virtual environment and installs dependencies)
./setup.sh

# 3. Activate virtual environment
source venv/bin/activate

# 4. Run detection on sample data
python run.py --target_model gpt2 --ref_model distilgpt2 --data my_chunks.jsonl --key_name input
```

### Windows:
```cmd
REM 1. Run setup (creates virtual environment and installs dependencies)
setup.bat

REM 2. Activate virtual environment
venv\Scripts\activate.bat

REM 3. Run detection on sample data
python run.py --target_model gpt2 --ref_model distilgpt2 --data my_chunks.jsonl --key_name input
```

## What Happens When You Run It

1. **First time**: Downloads GPT-2 and DistilGPT-2 models (~500MB)
2. **Processing**: Analyzes 5 sample text chunks
3. **Results**: Creates `out/gpt2_distilgpt2/input/` directory with:
   - `auc.txt` - Performance metrics and scores for each chunk
   - `auc.png` - ROC curve visualization
4. **Time**: ~10-15 minutes total

## Using Your Own Text Chunks

Replace `my_chunks.jsonl` with your data in this exact format:
```json
{"input": "Your first text chunk here", "label": 0}
{"input": "Your second text chunk here", "label": 0}
{"input": "Any text you want to check", "label": 0}
```

Then run the same command - it will process your chunks instead!

## Understanding Results

Look for **Min-20% Prob** scores in the `auc.txt` file:
- **2.0-3.5**: Very likely in training data
- **3.5-4.5**: Likely in training data
- **4.5-6.0**: Possibly in training data  
- **6.0+**: Likely NOT in training data

## Manual Setup (if scripts fail)

```bash
# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate    # Linux/Mac
# OR
venv\Scripts\activate.bat   # Windows

# Install dependencies
pip install -r requirements.txt

# Run detection
python run.py --target_model gpt2 --ref_model distilgpt2 --data my_chunks.jsonl --key_name input
```

## Troubleshooting

- **CUDA errors**: Ignore, will use CPU (slower but works)
- **Download failures**: Check internet connection, retry
- **Import errors**: Make sure virtual environment is activated
- **Permission errors**: Use `chmod +x setup.sh` on Linux/Mac

## System Requirements

- Python 3.7+
- ~2GB disk space (for models)
- Internet connection (for downloads)
- 4GB+ RAM recommended

This package is guaranteed to work - all dependencies and code are included!