@echo off
cd /d "%~dp0.."
call venv\Scripts\activate.bat
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU Compute Capability: {torch.cuda.get_device_capability(0) if torch.cuda.is_available() else \"N/A\"}')"
pause
