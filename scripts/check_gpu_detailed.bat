@echo off
cd /d "%~dp0.."
echo Checking NVIDIA GPU and CUDA setup...
echo.

echo === NVIDIA Driver Info ===
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv
echo.

echo === PyTorch Info ===
call venv\Scripts\activate.bat
python -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'PyTorch CUDA Version: {torch.version.cuda}'); print(f'cuDNN Version: {torch.backends.cudnn.version()}'); device_count = torch.cuda.device_count(); print(f'GPU Count: {device_count}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)} (Compute {torch.cuda.get_device_capability(i)[0]}.{torch.cuda.get_device_capability(i)[1]})') for i in range(device_count)]"
echo.

pause
