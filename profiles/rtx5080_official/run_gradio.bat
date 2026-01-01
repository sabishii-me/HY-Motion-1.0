@echo off
REM RTX 5080 Official Profile - Run Gradio (No Quantization)

cd /d "%~dp0"

if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run setup.bat first.
    pause
    exit /b 1
)

echo ========================================
echo RTX 5080 Official Profile
echo No Quantization ^| CPU Offloading
echo ========================================
echo.

REM Activate venv
call venv\Scripts\activate.bat

REM Set project root
set PROJECT_ROOT=%~dp0..\..
set PYTHONPATH=%PROJECT_ROOT%

REM Set profile configuration - "none" means no quantization
set HY_QUANTIZATION=none

echo Profile: RTX 5080 Official
echo Quantization: %HY_QUANTIZATION% (full precision)
echo VRAM: 16GB + system RAM offloading
echo.
echo NOTE: This will use CPU/RAM offloading for parts
echo that don't fit in 16GB VRAM. Slower than quantized.
echo.
echo Starting HY-Motion Gradio App...
echo Web interface will open at http://localhost:7860
echo.

cd "%PROJECT_ROOT%"
python gradio_app.py

pause
