@echo off

echo === Setting up Detect Pretrain Code Environment ===

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo Installing dependencies...
pip install -r requirements.txt

echo ✅ Setup complete!
echo.
echo To run detection:
echo 1. Activate environment: venv\Scripts\activate.bat
echo 2. Run detection: python run.py --target_model gpt2 --ref_model distilgpt2 --data my_chunks.jsonl --key_name input
echo.
pause
