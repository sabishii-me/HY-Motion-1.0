# RTX 5080 Configuration & Performance

## Hardware Specifications

### GPU Details
- **Model**: NVIDIA GeForce RTX 5080
- **Architecture**: Blackwell (GB202)
- **CUDA Cores**: 10,752
- **Tensor Cores**: 336 (5th generation)
- **RT Cores**: 84 (4th generation)
- **VRAM**: 16GB GDDR7
- **Memory Bus**: 256-bit
- **Memory Bandwidth**: 960 GB/s
- **TGP**: 360W
- **Compute Capability**: 12.0 (SM 12.0)

### System Configuration
- **OS**: Windows 11 (64-bit)
- **NVIDIA Driver**: 581.57 or newer
- **CUDA Toolkit**: 13.0 (via PyTorch nightly)
- **Python**: 3.10.11 embedded
- **PyTorch**: Nightly build (cu130)

## HY-Motion Performance Optimization

### Quantization Strategy

The RTX 5080's 16GB VRAM requires quantization to run HY-Motion efficiently:

| Configuration | VRAM Usage | Quality | Speed | Status |
|--------------|------------|---------|-------|--------|
| **INT4** | 6-8GB | Excellent | Very Fast | ✅ **Recommended** |
| INT8 | 12-13GB | Near Perfect | Fast | ✅ Usable |
| Full (FP16/32) | 24-26GB | Perfect | Baseline | ❌ Won't fit |

### INT4 Configuration (Current Setup)

**Location**: `gradio_app.py` line 540

```python
model_inference = ModelInference(
    final_model_path,
    use_prompt_engineering=False,
    use_text_encoder=True,
    quantization="int4"
)
```

**Benefits**:
- **VRAM**: ~6-8GB (leaves 8GB+ headroom)
- **Speed**: Fastest inference (INT4 math operations)
- **Quality**: ✅ **Tested - No visible difference vs INT8 or full precision**
- **Stability**: No OOM errors on 16GB VRAM

**Test Results** (RTX 5080 16GB):
- INT4 vs INT8: No perceptible quality difference
- INT4 vs Official (full precision): No perceptible quality difference, much faster
- **Recommendation**: Use INT4 - other options offer no quality benefit

### Technical Implementation

**Quantization Library**: bitsandbytes
```python
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
```

**Applied to**: Text encoders (largest VRAM consumers)
- Llama3 8B text encoder: ~16GB → ~4GB
- UMT-XXL text encoder: ~8GB → ~2GB

## Blackwell Architecture Support

### Why PyTorch Nightly?

RTX 5080 uses Blackwell architecture (compute capability 12.0), which requires:

1. **CUDA 13.0**: New kernel support for SM 12.0
2. **cuBLAS updates**: Blackwell-specific optimizations
3. **PyTorch nightly**: Built with CUDA 13.0 toolchain

**Stable PyTorch Issue**:
```
cuBLAS API failed with status 15
```
This error occurs because PyTorch 2.5.1 (CUDA 12.1) lacks Blackwell kernels.

**Solution**: PyTorch nightly with CUDA 13.0
```batch
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu130
```

## Performance Benchmarks

### Generation Speed (INT4)

Based on typical workloads:

| Task | Duration | VRAM Peak | Speed vs CPU |
|------|----------|-----------|--------------|
| 3s motion (single seed) | ~15-30s | 7GB | ~50x faster |
| 5s motion (single seed) | ~25-45s | 8GB | ~50x faster |
| Multi-seed (4 seeds) | Not recommended | >16GB | N/A |

**Note**: CPU-only mode would take 15-30 minutes per generation.

### VRAM Usage Profile

Typical VRAM allocation during generation:

```
Model Loading:     4-5GB  (quantized weights)
Text Encoding:     1-2GB  (prompt processing)
Diffusion:         2-3GB  (denoising steps)
Peak Usage:        7-8GB
Free VRAM:         8-9GB  (safety margin)
```

## Troubleshooting

### Common Issues

**1. cuBLAS Error (status 15)**
- **Cause**: PyTorch lacks Blackwell support
- **Fix**: Run `setup.bat` to install PyTorch nightly

**2. Out of Memory**
- **Cause**: Other GPU applications, or INT8/full precision
- **Fix**: Close other GPU apps, verify INT4 in `gradio_app.py:540`

**3. Slow Performance**
- **Cause**: CPU fallback (CUDA not available)
- **Fix**: Check driver (581.57+), verify CUDA with `scripts\check_gpu.bat`

### Verification Commands

**Check Blackwell Support**:
```batch
call venv\Scripts\activate.bat
python -c "import torch; cc = torch.cuda.get_device_capability(0); print(f'Compute Capability: {cc[0]}.{cc[1]}'); print(f'Blackwell: {\"YES\" if cc[0] >= 12 else \"NO\"}')"
```

**Expected Output**:
```
Compute Capability: 12.0
Blackwell: YES
```

**Check VRAM Usage**:
```batch
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

## Optimization Tips

### For Best Performance

1. **Use INT4 quantization** (current default)
2. **Single seed generation** (multiple seeds = linear VRAM multiply)
3. **Keep prompts concise** (<50 words optimal)
4. **Close background GPU apps** (browsers, games, etc.)
5. **Monitor VRAM** with Task Manager or `nvidia-smi`

### If You Need Higher Quality

Switch to INT8 (requires 12-13GB):
```python
# gradio_app.py line 540
model_inference = ModelInference(..., quantization="int8")
```

**Warning**: This leaves only 3-4GB headroom. May OOM with long prompts or multi-seed.

### Multi-Seed Limitation

**Automatic safety feature** in `hymotion/utils/gradio_runtime.py:190-194`:

```python
# Automatically limits to single seed for 16GB VRAM
if len(seeds) > 1:
    print(f">>> Warning: Multiple seeds ({len(seeds)}) not recommended for 16GB VRAM")
    seeds = seeds[:1]
```

**Why:**
- Multiple seeds multiply VRAM usage linearly
- Safety feature to prevent OOM crashes
- With INT4 (6-8GB), you likely have headroom for 2-3 seeds
- Can be disabled if you want to test multi-seed with INT4

## Comparison: RTX 5080 vs Other GPUs

| GPU | VRAM | HY-Motion Config | Status |
|-----|------|------------------|--------|
| **RTX 5080** | 16GB | INT4 quantization | ✅ Fast & stable |
| RTX 4090 | 24GB | INT8 or full | ✅ Full quality |
| RTX 5090 | 32GB | Full precision | ✅ Best quality |
| RTX 4080 | 16GB | INT4 quantization | ✅ Similar to 5080 |
| RTX 3090 | 24GB | INT8 recommended | ⚠️ Slower (older arch) |

**RTX 5080 Advantages**:
- Blackwell architecture (newer, more efficient)
- GDDR7 memory (higher bandwidth)
- 5th gen Tensor Cores (INT4 optimizations)
- Lower power than 4090 (360W vs 450W)

## Additional Resources

- [NVIDIA RTX 5080 Specs](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5080/)
- [Blackwell Architecture Whitepaper](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)
- [PyTorch CUDA Compatibility](https://pytorch.org/get-started/locally/)
- [bitsandbytes Quantization](https://github.com/TimDettmers/bitsandbytes)
