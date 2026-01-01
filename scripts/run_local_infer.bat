@echo off
REM Run local inference for HY-Motion-1.0

cd /d "%~dp0.."

if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run setup.bat first.
    pause
    exit /b 1
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

set PYTHONPATH=%CD%

echo.
echo Running HY-Motion-1.0 local inference...
echo Output will be saved to: output/local_infer
echo.
echo NOTE: Add --disable_duration_est and --disable_rewrite if you don't have the LLM module
echo.

python local_infer.py --model_path ckpts/tencent/HY-Motion-1.0 --disable_duration_est --disable_rewrite %*

pause
