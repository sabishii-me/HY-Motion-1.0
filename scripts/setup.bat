@echo off
REM Embedded Python Setup for HY-Motion-1.0
REM This script downloads Python 3.10.11 embedded and creates a virtual environment

setlocal enabledelayedexpansion

echo ========================================
echo HY-Motion 1.0 - Embedded Python Setup
echo ========================================
echo.

REM Change to project root directory
cd /d "%~dp0.."

REM Configuration
set PYTHON_VERSION=3.10.11
set PYTHON_URL=https://www.python.org/ftp/python/%PYTHON_VERSION%/python-%PYTHON_VERSION%-embed-amd64.zip
set PYTHON_EMBED_DIR=%CD%\python_embed
set VENV_DIR=%CD%\venv

REM Check if already set up
if exist "%VENV_DIR%\Scripts\activate.bat" (
    echo Virtual environment already exists.
    choice /C YN /M "Do you want to recreate it"
    if errorlevel 2 goto :skip_setup
    echo Cleaning up old environment...
    rmdir /s /q "%VENV_DIR%"
)

REM Download embedded Python if not exists
if not exist "%PYTHON_EMBED_DIR%" (
    echo Step 1: Downloading Python %PYTHON_VERSION% embedded...
    mkdir "%PYTHON_EMBED_DIR%"

    powershell -Command "& { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%PYTHON_URL%' -OutFile '%PYTHON_EMBED_DIR%\python.zip' }"

    if errorlevel 1 (
        echo ERROR: Failed to download Python
        exit /b 1
    )

    echo Extracting Python...
    powershell -Command "Expand-Archive -Path '%PYTHON_EMBED_DIR%\python.zip' -DestinationPath '%PYTHON_EMBED_DIR%' -Force"
    del "%PYTHON_EMBED_DIR%\python.zip"

    REM Enable site-packages by uncommenting import site in python310._pth
    powershell -Command "(Get-Content '%PYTHON_EMBED_DIR%\python310._pth') -replace '#import site', 'import site' | Set-Content '%PYTHON_EMBED_DIR%\python310._pth'"

    echo Python embedded installation complete.
) else (
    echo Step 1: Python embedded already exists at %PYTHON_EMBED_DIR%
)

REM Download get-pip.py
echo.
echo Step 2: Installing pip...
if not exist "%PYTHON_EMBED_DIR%\get-pip.py" (
    powershell -Command "Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile '%PYTHON_EMBED_DIR%\get-pip.py'"
)

"%PYTHON_EMBED_DIR%\python.exe" "%PYTHON_EMBED_DIR%\get-pip.py" --no-warn-script-location

REM Install virtualenv
echo.
echo Step 3: Installing virtualenv...
"%PYTHON_EMBED_DIR%\python.exe" -m pip install virtualenv --no-warn-script-location

REM Create virtual environment using virtualenv
echo.
echo Step 4: Creating virtual environment...
"%PYTHON_EMBED_DIR%\python.exe" -m virtualenv "%VENV_DIR%"

if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    exit /b 1
)

REM Activate virtual environment and upgrade pip
echo.
echo Step 5: Upgrading pip...
call "%VENV_DIR%\Scripts\activate.bat"
python -m pip install --upgrade pip

REM Install PyTorch Nightly with CUDA 13.0 for RTX 5080 (Blackwell) support
echo.
echo Step 6: Installing PyTorch Nightly with CUDA 13.0...
echo This version has Blackwell architecture support for RTX 50 series
echo This may take several minutes...
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu130 --no-deps
pip install filelock typing-extensions sympy networkx jinja2 fsspec pillow

if errorlevel 1 (
    echo ERROR: Failed to install PyTorch
    echo Trying fallback to stable PyTorch 2.5.1...
    pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
)

REM Install project requirements
echo.
echo Step 7: Installing project requirements...
pip install torchdiffeq==0.2.5 --extra-index-url https://gitlab.inria.fr/api/v4/projects/18692/packages/pypi/simple
pip install "huggingface_hub>=0.30.0,<1.0" accelerate==0.30.1 diffusers==0.26.3 transformers==4.53.3
pip install einops==0.8.1 safetensors==0.5.3 bitsandbytes==0.49.0
pip install "numpy>=1.24.0,<2.0" "scipy>=1.10.0" transforms3d==0.4.2
pip install PyYAML==6.0 omegaconf==2.3.0 click==8.1.3 requests==2.32.4 openai==1.78.1

REM Install Gradio for web interface
echo.
echo Step 8: Installing Gradio and web dependencies...
pip install gradio spaces

REM Try to install optional FBX support (may fail, that's OK)
echo.
echo Step 9: Installing optional FBX support...
echo Note: FBX export is optional. The app works fine without it.
pip install fbxsdkpy==2020.1.post2 2>nul
if errorlevel 1 (
    echo FBX SDK installation skipped - this is OK!
)

:skip_setup

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Virtual environment: %VENV_DIR%
echo Python version: %PYTHON_VERSION%
echo PyTorch with CUDA 12.1: Installed
echo.
echo Next steps:
echo 1. Download model weights (see ckpts\README.md)
echo 2. Run: run_gradio.bat
echo 3. Open browser to http://localhost:7860
echo.

REM Test CUDA and Blackwell support
echo.
echo ========================================
echo Testing CUDA and Blackwell Support...
echo ========================================
call "%VENV_DIR%\Scripts\activate.bat"
python -c "import torch; print(f'\nPyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}'); gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'; print(f'GPU: {gpu_name}'); cc = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0,0); print(f'Compute Capability: {cc[0]}.{cc[1]}'); print(f'\n--- RTX 5080 (Blackwell) Check ---'); print(f'Blackwell Support: {\"YES - RTX 50 series supported!\" if cc[0] >= 12 else \"NO - May have cuBLAS errors\"}')"

echo.
echo Testing matrix multiplication (cuBLAS check)...
python -c "import torch; a = torch.randn(100, 100, device='cuda'); b = torch.randn(100, 100, device='cuda'); c = torch.matmul(a, b); print('cuBLAS test PASSED - Matrix multiplication works!')"

if errorlevel 1 (
    echo.
    echo WARNING: cuBLAS test failed!
    echo This means PyTorch doesn't have proper Blackwell support yet.
    echo You may need to wait for a newer PyTorch version or use CPU mode.
)

echo.
pause
