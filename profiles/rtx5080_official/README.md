# RTX 5080 Official Profile

Matches official setup with no quantization. Uses CPU/RAM offloading for parts that exceed 16GB VRAM.

## Specifications

- **GPU**: RTX 5080 (16GB VRAM)
- **Architecture**: Blackwell (Compute 12.0)
- **PyTorch**: Nightly with CUDA 13.0
- **Quantization**: None (full precision FP16/FP32)
- **VRAM Usage**: 16GB (full) + system RAM offloading
- **Quality**: 100% (identical to official)
- **Speed**: Slower (due to CPU offloading)

## Setup

```batch
setup.bat
```

## Run

```batch
run_gradio.bat
```

## Configuration

- **HY_QUANTIZATION**: none
- **CUDA Version**: 13.0 (cu130)
- **Python**: 3.10.11 embedded
- **Offloading**: Automatic via device_map="auto"

## Notes

- Uses full precision (no quality loss)
- Parts that don't fit in 16GB VRAM are offloaded to system RAM
- **Slower** than INT4/INT8 due to CPU offloading overhead
- Use this to compare quality vs quantized profiles
- Requires sufficient system RAM (~16GB+)

## Performance

- **Loading**: Slower (offloading layers to RAM)
- **Inference**: Slower (CPU/GPU data transfers)
- **Quality**: Reference quality (100%)

## When to Use

- Quality comparison testing
- When you need absolute maximum quality
- When speed is not a concern
- For final production renders

Most users should use INT4 or INT8 profiles for better performance.
