# RTX 5080 INT4 Profile

Optimized for RTX 5080 with 16GB VRAM using INT4 quantization.

## Specifications

- **GPU**: RTX 5080 (16GB VRAM)
- **Architecture**: Blackwell (Compute 12.0)
- **PyTorch**: Nightly with CUDA 13.0
- **Quantization**: INT4
- **VRAM Usage**: ~6-8GB
- **Quality**: Excellent (minimal degradation)
- **Speed**: Very fast

## Setup

```batch
setup.bat
```

## Run

```batch
run_gradio.bat
```

## Configuration

- **HY_QUANTIZATION**: int4
- **CUDA Version**: 13.0 (cu130)
- **Python**: 3.10.11 embedded

## Notes

- INT4 leaves plenty of VRAM headroom (8GB+)
- Recommended for stable daily use
- Fast inference with minimal quality loss
