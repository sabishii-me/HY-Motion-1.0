# HY-Motion 1.0 - RTX 5080 Setup

## Quick Start

```batch
setup.bat       # First time only
run_gradio.bat  # Run the app
```

Open http://localhost:7860

## Your Configuration

- **GPU**: RTX 5080 16GB (Blackwell)
- **Quantization**: INT4 (~6-8GB VRAM)
- **PyTorch**: Nightly with CUDA 13.0
- **Speed**: Very fast (~15-30s per 3s motion)

## Files

**Run these**:
- `setup.bat` - Install everything
- `run_gradio.bat` - Start web interface

**Specs & optimization**:
- `RTX5080_SPECS.md` - Full hardware specs, benchmarks, tips

**Utilities** (scripts/ folder):
- `scripts\check_gpu.bat` - GPU check
- `scripts\check_gpu_detailed.bat` - Full diagnostics
- `scripts\run_local_infer.bat` - CLI mode

## Change Quality

Edit `gradio_app.py` line 540:

```python
quantization="int4"  # Current (6-8GB, fastest)
quantization="int8"  # Better quality (12-13GB)
```

Don't use `"none"` - won't fit in 16GB.
