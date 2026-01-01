import os
import threading
import time
import uuid
from typing import List, Optional, Tuple, Union

import torch
import yaml

from ..prompt_engineering.prompt_rewrite import PromptRewriter
from .loaders import load_object
from .visualize_mesh_web import save_visualization_data, generate_static_html_content

try:
    import fbx

    FBX_AVAILABLE = True
    print(">>> FBX module found.")
except ImportError:
    FBX_AVAILABLE = False
    print(">>> FBX module not found.")


def _now():
    t = time.time()
    ms = int((t - int(t)) * 1000)
    return time.strftime("%Y%m%d_%H%M%S", time.localtime(t)) + f"{ms:03d}"

_MODEL_CACHE = None


class SimpleRuntime(torch.nn.Module):
    def __init__(self, config_path, ckpt_name, load_prompt_engineering=False, load_text_encoder=False, quantization="none"):
        super().__init__()
        self.load_prompt_engineering = load_prompt_engineering
        self.load_text_encoder = load_text_encoder
        self.quantization = quantization
        # prompt engineering
        if self.load_prompt_engineering:
            print(f"[{self.__class__.__name__}] Loading prompt engineering...")
            self.prompt_rewriter = PromptRewriter(
                host=None, model_path=None, device="cpu"
            )
        else:
            self.prompt_rewriter = None
        # text encoder
        if self.load_text_encoder:
            print(f"[{self.__class__.__name__}] Loading text encoder...")
            print(f"[{self.__class__.__name__}] Quantization: {quantization}")
            _text_encoder_module = "hymotion/network/text_encoders/text_encoder.HYTextModel"
            _text_encoder_cfg = {
                "llm_type": "qwen3",
                "max_length_llm": 128,
                "quantization": quantization if quantization != "none" else None
            }
            text_encoder = load_object(_text_encoder_module, _text_encoder_cfg)
        else:
            text_encoder = None
        # 2. load model
        print(f"[{self.__class__.__name__}] Loading model...")
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        pipeline = load_object(
            config["train_pipeline"],
            config["train_pipeline_args"],
            network_module=config["network_module"],
            network_module_args=config["network_module_args"],
        )
        print(f"[{self.__class__.__name__}] Loading ckpt: {ckpt_name}")
        pipeline.load_in_demo(
            os.path.join(os.path.dirname(config_path), ckpt_name),
            build_text_encoder=False,
            allow_empty_ckpt=False,
        )
        pipeline.text_encoder = text_encoder

        # Note: float16 optimization disabled due to dtype mismatch issues in generate()
        # The pipeline has many hardcoded tensor creations without dtype specification
        # For RTX 5080 16GB: Use CPU offloading or wait for proper fix
        print(f"[{self.__class__.__name__}] Model loaded (float32)")
        print(f"  Note: For 16GB VRAM GPUs, model may need CPU offloading")

        self.pipeline = pipeline
        #
        self.fbx_available = FBX_AVAILABLE
        if self.fbx_available:
            try:
                from .smplh2woodfbx import SMPLH2WoodFBX

                self.fbx_converter = SMPLH2WoodFBX()
            except Exception as e:
                print(f">>> Failed to initialize FBX converter: {e}")
                self.fbx_available = False
                self.fbx_converter = None
        else:
            self.fbx_converter = None
            print(">>> FBX module not found. FBX export will be disabled.")


    def _generate_html_content(
        self,
        timestamp: str,
        file_path: str,
        output_dir: Optional[str] = None,
    ) -> str:
        """
        Generate static HTML content with embedded data for iframe srcdoc.
        All JavaScript code is embedded directly in the HTML, no external static resources needed.

        Args:
            timestamp: Timestamp string for logging
            file_path: Base filename (without extension)
            output_dir: Directory where NPZ/meta files are stored

        Returns:
            HTML content string (to be used in iframe srcdoc)
        """
        print(f">>> Generating static HTML content, timestamp: {timestamp}")
        gradio_dir = output_dir if output_dir is not None else "output/gradio"

        try:
            # Generate static HTML content with embedded data (all JS is embedded in template)
            html_content = generate_static_html_content(
                folder_name=gradio_dir,
                file_name=file_path,
                hide_captions=False,
            )

            print(f">>> Static HTML content generated for: {file_path}")
            return html_content

        except Exception as e:
            print(f">>> Failed to generate static HTML content: {e}")
            import traceback

            traceback.print_exc()
            # Return error HTML
            return f"<html><body><h1>Error generating visualization</h1><p>{str(e)}</p></body></html>"


    def _generate_fbx_files(
        self,
        visualization_data: dict,
        output_dir: Optional[str] = None,
        fbx_filename: Optional[str] = None,
    ) -> List[str]:
        assert "smpl_data" in visualization_data, "smpl_data not found in visualization_data"
        fbx_files = []
        if output_dir is None:
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            output_dir = os.path.join(root_dir, "output", "gradio")

        smpl_data_list = visualization_data["smpl_data"]

        unique_id = str(uuid.uuid4())[:8]
        text = visualization_data["text"]
        timestamp = visualization_data["timestamp"]
        for bb in range(len(smpl_data_list)):
            smpl_data = smpl_data_list[bb]
            if fbx_filename is None:
                fbx_filename_bb = f"{timestamp}_{unique_id}_{bb:03d}.fbx"
            else:
                fbx_filename_bb = f"{fbx_filename}_{bb:03d}.fbx"
            fbx_path = os.path.join(output_dir, fbx_filename_bb)
            success = self.fbx_converter.convert_npz_to_fbx(smpl_data, fbx_path)
            if success:
                fbx_files.append(fbx_path)
                print(f"\t>>> FBX file generated: {fbx_path}")
                txt_path = fbx_path.replace(".fbx", ".txt")
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(text)
                fbx_files.append(txt_path)

        return fbx_files

    def generate_motion(
        self,
        text: str,
        seeds_csv: str,
        motion_duration: float,
        cfg_scale: float,
        output_format: str = "fbx",
        output_dir: Optional[str] = None,
        output_filename: Optional[str] = None,
        original_text: Optional[str] = None,
        use_special_game_feat: bool = False,
    ) -> Tuple[Union[str, list[str]], dict]:
        seeds = [int(s.strip()) for s in seeds_csv.split(",") if s.strip() != ""]

        # Force single seed for low VRAM GPUs (RTX 5080 16GB)
        if len(seeds) > 1:
            print(f">>> Warning: Multiple seeds ({len(seeds)}) not recommended for 16GB VRAM")
            print(f">>> Using only first seed to conserve memory")
            seeds = seeds[:1]

        print(f"[{self.__class__.__name__}] Generating motion...")
        print(f"[{self.__class__.__name__}] text: {text}")
        if self.load_prompt_engineering:
            duration, rewritten_text = self.prompt_rewriter.rewrite_prompt_and_infer_time(f"{text}")
        else:
            rewritten_text = text
            duration = motion_duration

        pipeline = self.pipeline
        pipeline.eval()

        # When skip_text=True (debug mode), use blank text features
        if not self.load_text_encoder:
            print(">>> [Debug Mode] Using blank text features (skip_text=True)")
            device = next(pipeline.parameters()).device
            dtype = next(pipeline.parameters()).dtype
            batch_size = len(seeds) if seeds else 1
            # Create blank hidden_state_dict using null features with matching dtype
            hidden_state_dict = {
                "text_vec_raw": pipeline.null_vtxt_feat.expand(batch_size, -1, -1).to(device, dtype=dtype),
                "text_ctxt_raw": pipeline.null_ctxt_input.expand(batch_size, -1, -1).to(device, dtype=dtype),
                "text_ctxt_raw_length": torch.tensor([1] * batch_size, device=device),
            }
            # Disable CFG in debug mode (use cfg_scale=1.0)
            model_output = pipeline.generate(
                rewritten_text,
                seeds,
                duration,
                cfg_scale=1.0,
                use_special_game_feat=False,
                hidden_state_dict=hidden_state_dict,
            )
        else:
            model_output = pipeline.generate(
                rewritten_text, seeds, duration, cfg_scale=cfg_scale, use_special_game_feat=use_special_game_feat
            )

        ts = _now()
        save_data, base_filename = save_visualization_data(
            output=model_output,
            text=text if original_text is None else original_text,
            rewritten_text=rewritten_text,
            timestamp=ts,
            output_dir=output_dir,
            output_filename=output_filename,
        )

        html_content = self._generate_html_content(
            timestamp=ts,
            file_path=base_filename,
            output_dir=output_dir,
        )

        if output_format == "fbx" and not self.fbx_available:
            print(">>> Warning: FBX export requested but FBX SDK is not available. Falling back to dict format.")
            output_format = "dict"

        if output_format == "fbx" and self.fbx_available:
            fbx_files = self._generate_fbx_files(
                visualization_data=save_data,
                output_dir=output_dir,
                fbx_filename=output_filename,
            )
            return html_content, fbx_files, model_output
        elif output_format == "dict":
            # Return HTML content and empty list for fbx_files when using dict format
            return html_content, [], model_output
        else:
            raise ValueError(f">>> Invalid output format: {output_format}")

class ModelInference:
    """
    Handles model inference and data processing for Depth Anything 3.
    """

    def __init__(self, model_path, use_prompt_engineering, use_text_encoder, quantization="int8"):
        """Initialize the model inference handler.

        Args:
            quantization: "none", "int8", or "int4" - reduces VRAM usage
                - none: Full precision (24-26GB VRAM)
                - int8: 8-bit quantization (~12-13GB VRAM) [Recommended for 16GB GPUs]
                - int4: 4-bit quantization (~6-8GB VRAM) [May reduce quality]

        Note: Do not store model in instance variable to avoid
        cross-process state issues with @spaces.GPU decorator.
        """
        # No instance variables - model cached in global variable
        self.model_path = model_path
        self.use_prompt_engineering = use_prompt_engineering
        self.use_text_encoder = use_text_encoder
        self.quantization = quantization
        self.fbx_available = FBX_AVAILABLE

    def initialize_model(self, device: str = "cuda"):
        """
        Initialize the DepthAnything3 model using global cache.

        Optimization: Load model to CPU first, then move to GPU when needed.
        This is faster than reloading from disk each time.

        This uses a global variable which is safe because @spaces.GPU
        runs in isolated subprocess, each with its own global namespace.
        Args:
            device: Device to run inference on (will move model to this device)

        Returns:
            Model instance ready for inference on specified device
        """
        global _MODEL_CACHE

        if _MODEL_CACHE is None:
            # First time loading in this subprocess
            # Load to CPU first (faster than loading directly to GPU)
            _MODEL_CACHE = SimpleRuntime(
                config_path=os.path.join(self.model_path, "config.yml"),
                ckpt_name="latest.ckpt",
                load_prompt_engineering=self.use_prompt_engineering,
                load_text_encoder=self.use_text_encoder,
                quantization=self.quantization
            )
            # Load to CPU first (faster, and allows reuse)
            _MODEL_CACHE = _MODEL_CACHE.to("cpu")
            _MODEL_CACHE.eval()
            print("✅ Model loaded to CPU memory (cached in subprocess)")

        # Move to target device for inference
        if device != "cpu" and next(_MODEL_CACHE.parameters()).device.type != device:
            print(f"🚀 Moving model from {next(_MODEL_CACHE.parameters()).device} to {device}...")
            _MODEL_CACHE = _MODEL_CACHE.to(device)
            print(f"✅ Model ready on {device}")
        elif device == "cpu":
            # Already on CPU or requested CPU
            pass

        return _MODEL_CACHE

    def run_inference(
        self, *args, **kwargs
    ):
        """
        Run DepthAnything3 model inference on images.
        Args:
            target_dir: Directory containing images
            apply_mask: Whether to apply mask for ambiguous depth classes
            mask_edges: Whether to mask edges
            filter_black_bg: Whether to filter black background
            filter_white_bg: Whether to filter white background
            process_res_method: Method for resizing input images
            show_camera: Whether to show camera in 3D view
            selected_first_frame: Selected first frame filename
            save_percentage: Percentage of points to save (0-100)
            infer_gs: Whether to infer 3D Gaussian Splatting
        Returns:
            Tuple of (prediction, processed_data)
        """
        print(f"[{self.__class__.__name__}] Running inference...")
        # Device check
        device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)

        # Initialize model if needed - get model instance (not stored in self)
        model = self.initialize_model(device)


        with torch.no_grad():
            print(f"[{self.__class__.__name__}] Running inference with torch.no_grad")
            html_content, fbx_files, model_output = model.generate_motion(*args, **kwargs)
        # CRITICAL: Move all CUDA tensors to CPU before returning
        # This prevents CUDA initialization in main process during unpickling
        for k, val in model_output.items():
            if isinstance(val, torch.Tensor):
                model_output[k] = val.detach().cpu()
        # # Clean up
        torch.cuda.empty_cache()

        return html_content, fbx_files

if __name__ == "__main__":
    # python -m hymotion.utils.gradio_runtime
    runtime = SimpleRuntime(config_path="assets/config_simplified.yml", ckpt_name="latest.ckpt", load_prompt_engineering=False, load_text_encoder=False)
    print(runtime.pipeline)