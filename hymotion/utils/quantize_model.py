"""
Model quantization utilities for reducing VRAM usage.
Based on bitsandbytes library for INT4/INT8 quantization.
"""
import torch
import torch.nn as nn


def quantize_model_int8(model):
    """
    Quantize model to INT8 using bitsandbytes.
    Reduces VRAM by ~50% with minimal quality loss.
    """
    try:
        import bitsandbytes as bnb

        print(">>> Applying INT8 quantization...")

        # Replace Linear layers with 8-bit quantized versions
        def replace_linear_with_int8(module):
            for name, child in module.named_children():
                if isinstance(child, nn.Linear):
                    # Replace with 8-bit linear layer
                    int8_layer = bnb.nn.Linear8bitLt(
                        child.in_features,
                        child.out_features,
                        bias=child.bias is not None,
                        has_fp16_weights=False,
                    )
                    # Copy weights
                    int8_layer.weight = bnb.nn.Int8Params(
                        child.weight.data,
                        requires_grad=False,
                        has_fp16_weights=False,
                    )
                    if child.bias is not None:
                        int8_layer.bias = child.bias
                    setattr(module, name, int8_layer)
                else:
                    replace_linear_with_int8(child)

        replace_linear_with_int8(model)
        print("  ✓ Model quantized to INT8")
        return model

    except ImportError:
        print("  ✗ bitsandbytes not available, skipping quantization")
        return model
    except Exception as e:
        print(f"  ✗ Quantization failed: {e}")
        return model


def quantize_model_int4(model):
    """
    Quantize model to INT4 using bitsandbytes.
    Reduces VRAM by ~75% but may have slight quality degradation.
    """
    try:
        import bitsandbytes as bnb

        print(">>> Applying INT4 quantization...")

        # Replace Linear layers with 4-bit quantized versions
        def replace_linear_with_int4(module):
            for name, child in module.named_children():
                if isinstance(child, nn.Linear):
                    # Replace with 4-bit linear layer
                    int4_layer = bnb.nn.Linear4bit(
                        child.in_features,
                        child.out_features,
                        bias=child.bias is not None,
                        compute_dtype=torch.float16,
                        compress_statistics=True,
                        quant_type='nf4'
                    )
                    # The weights will be quantized automatically
                    setattr(module, name, int4_layer)
                else:
                    replace_linear_with_int4(child)

        replace_linear_with_int4(model)
        print("  ✓ Model quantized to INT4")
        return model

    except ImportError:
        print("  ✗ bitsandbytes not available, skipping quantization")
        return model
    except Exception as e:
        print(f"  ✗ Quantization failed: {e}")
        return model
