@echo off
REM RTX 5080 INT4 Profile Setup
REM Optimized for 16GB VRAM with Blackwell architecture

cd /d "%~dp0"

set PROFILE_NAME=rtx5080_int4
set PYTHON_VERSION=3.10.11
set PYTHON_URL=https://www.python.org/ftp/python/%PYTHON_VERSION%/python-%PYTHON_VERSION%-embed-amd64.zip
set PYTHON_EMBED_DIR=%CD%\python_embed
set VENV_DIR=%CD%\venv
set PROJECT_ROOT=%CD%\..\..

echo ========================================
echo RTX 5080 INT4 Profile Setup
echo 16GB VRAM ^| Blackwell ^| CUDA 13.0
echo ========================================
echo.

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
    echo Downloading Python %PYTHON_VERSION% embedded...
    mkdir "%PYTHON_EMBED_DIR%"
    powershell -Command "& { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%PYTHON_URL%' -OutFile '%PYTHON_EMBED_DIR%\python.zip' }"

    if errorlevel 1 (
        echo ERROR: Failed to download Python
        pause
        exit /b 1
    )

    echo Extracting Python...
    powershell -Command "Expand-Archive -Path '%PYTHON_EMBED_DIR%\python.zip' -DestinationPath '%PYTHON_EMBED_DIR%' -Force"
    del "%PYTHON_EMBED_DIR%\python.zip"

    REM Enable site-packages
    powershell -Command "(Get-Content '%PYTHON_EMBED_DIR%\python310._pth') -replace '#import site', 'import site' | Set-Content '%PYTHON_EMBED_DIR%\python310._pth'"
)

REM Install pip
echo Installing pip...
if not exist "%PYTHON_EMBED_DIR%\get-pip.py" (
    powershell -Command "Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile '%PYTHON_EMBED_DIR%\get-pip.py'"
)
"%PYTHON_EMBED_DIR%\python.exe" "%PYTHON_EMBED_DIR%\get-pip.py" --no-warn-script-location

REM Install virtualenv
echo Installing virtualenv...
"%PYTHON_EMBED_DIR%\python.exe" -m pip install virtualenv --no-warn-script-location

REM Create virtual environment
echo Creating virtual environment...
"%PYTHON_EMBED_DIR%\python.exe" -m virtualenv "%VENV_DIR%"

REM Activate and upgrade pip
call "%VENV_DIR%\Scripts\activate.bat"
python -m pip install --upgrade pip

REM Install PyTorch Nightly with CUDA 13.0 (Blackwell support)
echo.
echo Installing PyTorch Nightly with CUDA 13.0 for RTX 5080...
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu130 --no-deps
pip install filelock typing-extensions sympy networkx jinja2 fsspec pillow

REM Install requirements
echo.
echo Installing project requirements...
pip install torchdiffeq==0.2.5 --extra-index-url https://gitlab.inria.fr/api/v4/projects/18692/packages/pypi/simple
pip install "huggingface_hub>=0.30.0,<1.0" accelerate==0.30.1 diffusers==0.26.3 transformers==4.53.3
pip install einops==0.8.1 safetensors==0.5.3 bitsandbytes==0.49.0
pip install "numpy>=1.24.0,<2.0" "scipy>=1.10.0" transforms3d==0.4.2
pip install PyYAML==6.0 omegaconf==2.3.0 click==8.1.3 requests==2.32.4 openai==1.78.1
pip install gradio spaces

:skip_setup

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo Profile: %PROFILE_NAME%
echo Config: INT4 quantization, CUDA 13.0
echo.
echo Run: run_gradio.bat
echo.
pause
