# HY-Motion 1.0 - RTX 5080 Setup

Quick setup guide for running HY-Motion on RTX 5080 with INT4 quantization.

## Quick Start

```batch
launch.bat
```

Select profile → Setup (first time) → Run

## Profiles

### [1] RTX 5080 INT4 ✅ Recommended
- **VRAM**: 6-8GB
- **Quality**: Excellent (tested - no difference vs higher settings)
- **Speed**: Very fast
- **Use**: Daily use

### [2] RTX 5080 INT8
- **VRAM**: 12-13GB
- **Quality**: Same as INT4
- **Speed**: Fast
- **Use**: Testing only

### [3] RTX 5080 Official
- **VRAM**: 16GB + RAM offloading
- **Quality**: Same as INT4
- **Speed**: Much slower
- **Use**: Reference comparison

## Test Results

All three profiles produce **identical visual quality** on RTX 5080 16GB.

**INT4 is the clear winner** - use it for everything.

## Documentation

- **[profiles/RTX5080_SPECS.md](profiles/RTX5080_SPECS.md)** - Full hardware specs & benchmarks
- **[profiles/README_RTX5080.md](profiles/README_RTX5080.md)** - Quick reference
- **Profile READMEs** - Individual profile details

## What Changed From Official Repo

1. **PyTorch Nightly with CUDA 13.0** - Blackwell (RTX 5080) support
2. **INT4 Quantization** - Fits in 16GB VRAM
3. **Profile System** - Isolated environments for testing
4. **Single Seed Limit** - Prevents OOM on 16GB

See profiles/RTX5080_SPECS.md for detailed changes.
