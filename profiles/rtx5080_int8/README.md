# RTX 5080 INT8 Profile

Optimized for RTX 5080 with 16GB VRAM using INT8 quantization for higher quality.

## Specifications

- **GPU**: RTX 5080 (16GB VRAM)
- **Architecture**: Blackwell (Compute 12.0)
- **PyTorch**: Nightly with CUDA 13.0
- **Quantization**: INT8
- **VRAM Usage**: ~12-13GB
- **Quality**: Near-perfect
- **Speed**: Fast

## Setup

```batch
setup.bat
```

## Run

```batch
run_gradio.bat
```

## Configuration

- **HY_QUANTIZATION**: int8
- **CUDA Version**: 13.0 (cu130)
- **Python**: 3.10.11 embedded

## Notes

- INT8 uses more VRAM than INT4 (~12-13GB vs 6-8GB)
- Only 3-4GB headroom - may OOM with long prompts
- Better quality than INT4
- Use for final renders when quality matters
