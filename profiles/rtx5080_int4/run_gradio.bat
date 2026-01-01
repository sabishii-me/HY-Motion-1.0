@echo off
REM RTX 5080 INT4 Profile - Run Gradio

cd /d "%~dp0"

if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run setup.bat first.
    pause
    exit /b 1
)

echo ========================================
echo RTX 5080 INT4 Profile
echo 16GB VRAM ^| Blackwell ^| CUDA 13.0
echo ========================================
echo.

REM Activate venv
call venv\Scripts\activate.bat

REM Set project root
set PROJECT_ROOT=%~dp0..\..
set PYTHONPATH=%PROJECT_ROOT%

REM Set profile configuration
set HY_QUANTIZATION=int4

echo Profile: RTX 5080 INT4
echo Quantization: %HY_QUANTIZATION%
echo VRAM Usage: ~6-8GB
echo.
echo Starting HY-Motion Gradio App...
echo Web interface will open at http://localhost:7860
echo.

cd "%PROJECT_ROOT%"
python gradio_app.py

pause
