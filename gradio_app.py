import argparse
import codecs as cs
import json
import os
import os.path as osp
import random
import re
import textwrap
from typing import List, Optional, Tuple, Union

import gradio as gr
from hymotion.utils.gradio_runtime import ModelInference
from hymotion.utils.gradio_utils import try_to_download_model, try_to_download_text_encoder
from hymotion.utils.gradio_css import get_placeholder_html, APP_CSS, HEADER_BASE_MD, FOOTER_MD

# Import spaces for Hugging Face Zero GPU support
import spaces

# define data sources
DATA_SOURCES = {
    "example_prompts": "examples/example_prompts/example_subset.json",
}


def load_examples_from_txt(txt_path: str, example_record_fps=30, max_duration=12):
    """Load examples from txt file."""

    def _parse_line(line: str) -> Optional[Tuple[str, float]]:
        line = line.strip()
        if line and not line.startswith("#"):
            parts = line.split("#")
            if len(parts) >= 2:
                text = parts[0].strip()
                duration = int(parts[1]) / example_record_fps
                duration = min(duration, max_duration)
            else:
                text = line.strip()
                duration = 5.0
            return text, duration
        return None

    examples: List[Tuple[str, float]] = []
    if os.path.exists(txt_path):
        try:
            if txt_path.endswith(".txt"):
                with cs.open(txt_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    for line in lines:
                        result = _parse_line(line)
                        if result is None:
                            continue
                        text, duration = result
                        examples.append((text, duration))
            elif txt_path.endswith(".json"):
                with cs.open(txt_path, "r", encoding="utf-8") as f:
                    lines = json.load(f)
                    for key, value in lines.items():
                        if "_raw_chn" in key or "GENERATE_PROMPT_FORMAT" in key:
                            continue
                        for line in value:
                            result = _parse_line(line)
                            if result is None:
                                continue
                            text, duration = result
                            examples.append((text, duration))
            print(f">>> Loaded {len(examples)} examples from {txt_path}")
        except Exception as e:
            print(f">>> Failed to load examples from {txt_path}: {e}")
    else:
        print(f">>> Examples file not found: {txt_path}")

    return examples


@spaces.GPU(duration=120)  # Request GPU for up to 120 seconds per inference
def generate_motion_func(
    # text input
    original_text: str,
    rewritten_text: str,
    # model input
    seed_input: str,
    motion_duration: float,
    cfg_scale: float,
) -> Tuple[str, List[str]]:
    use_prompt_engineering = False
    output_dir = "output/gradio"
    # When rewrite is not available, use original_text directly
    if use_prompt_engineering:
        text_to_use = rewritten_text.strip()
        if not text_to_use:
            return "Error: Rewritten text is empty, please rewrite the text first", []
    else:
        text_to_use = original_text.strip()
        if not text_to_use:
            return "Error: Input text is empty, please enter text first", []

    try:
        # Use runtime from global if available (for Zero GPU), otherwise use self.runtime
        fbx_ok = model_inference.fbx_available
        req_format = "fbx" if fbx_ok else "dict"

        # Use GPU-decorated wrapper function for Zero GPU support
        # This ensures the GPU decorator receives proper Gradio context for user authentication
        html_content, fbx_files = model_inference.run_inference(
            text=text_to_use,
            seeds_csv=seed_input,
            motion_duration=motion_duration,
            cfg_scale=cfg_scale,
            output_format=req_format,
            original_text=original_text,
            output_dir=output_dir,
        )
        print(f"Running inference...after gpu_inference_wrapper")
        # Escape HTML content for srcdoc attribute
        escaped_html = html_content.replace('"', "&quot;")
        # Return iframe with srcdoc - directly embed HTML content
        iframe_html = f"""
            <iframe
                srcdoc="{escaped_html}"
                width="100%"
                height="750px"
                style="border: none; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.1);"
            ></iframe>
        """
        return iframe_html, fbx_files
    except Exception as e:
        print(f"\t>>> Motion generation failed: {e}")
        return (
            f"❌ Motion generation failed: {str(e)}\n\nPlease check the input parameters or try again later",
            [],
        )


class T2MGradioUI:
    def __init__(self, args):
        self.output_dir = args.output_dir
        print(f"[{self.__class__.__name__}] output_dir: {self.output_dir}")
        # self.args = args
        self.prompt_engineering_available = args.use_prompt_engineering
        self.all_example_data = {}
        self._init_example_data()

    def _init_example_data(self):
        for source_name, file_path in DATA_SOURCES.items():
            examples = load_examples_from_txt(file_path)
            if examples:
                self.all_example_data[source_name] = examples
            else:
                # provide default examples as fallback
                self.all_example_data[source_name] = [
                    ("Twist at the waist and punch across the body.", 3.0),
                    ("A person is running then takes big leap.", 3.0),
                    ("A person holds a railing and walks down a set of stairs.", 5.0),
                    (
                        "A man performs a fluid and rhythmic hip-hop style dance, incorporating body waves, arm gestures, and side steps.",
                        5.0,
                    ),
                ]
        print(f">>> Loaded data sources: {list(self.all_example_data.keys())}")

    def _get_header_text(self):
        return HEADER_BASE_MD

    def _generate_random_seeds(self):
        seeds = [random.randint(0, 999) for _ in range(4)]
        return ",".join(map(str, seeds))

    def _prompt_engineering(
        self, text: str, duration: float, enable_rewrite: bool = True, enable_duration_est: bool = True
    ):
        if not text.strip():
            return "", gr.update(interactive=False), gr.update()

        call_llm = enable_rewrite or enable_duration_est
        if not call_llm:
            print(f"\t>>> Using original duration and original text...")
            predicted_duration = duration
            rewritten_text = text
        else:
            print(f"\t>>> Using LLM to estimate duration/rewrite text...")
            try:
                predicted_duration, rewritten_text = model_inference.rewrite_text_and_infer_time(text=text)
            except Exception as e:
                print(f"\t>>> Text rewriting/duration prediction failed: {e}")
                return (
                    f"❌ Text rewriting/duration prediction failed: {str(e)}",
                    gr.update(interactive=False),
                    gr.update(),
                )
            if not enable_rewrite:
                rewritten_text = text
            if not enable_duration_est:
                predicted_duration = duration

        return rewritten_text, gr.update(interactive=True), gr.update(value=predicted_duration)

    def _get_example_choices(self):
        """Get all example choices from all data sources"""
        choices = ["Custom Input"]
        for source_name in self.all_example_data:
            example_data = self.all_example_data[source_name]
            for text, _ in example_data:
                display_text = f"{text[:50]}..." if len(text) > 50 else text
                choices.append(display_text)
        return choices

    def _on_example_select(self, selected_example):
        """When selecting an example, the callback function"""
        if selected_example == "Custom Input":
            return "", self._generate_random_seeds(), gr.update()
        else:
            # find the corresponding example from all data sources
            for source_name in self.all_example_data:
                example_data = self.all_example_data[source_name]
                for text, duration in example_data:
                    display_text = f"{text[:50]}..." if len(text) > 50 else text
                    if display_text == selected_example:
                        return text, self._generate_random_seeds(), gr.update(value=duration)
            return "", self._generate_random_seeds(), gr.update()

    def build_ui(self):
        with gr.Blocks(css=APP_CSS) as demo:
            # Create State components for non-UI values that need to be passed to event handlers
            self.use_prompt_engineering_state = gr.State(self.prompt_engineering_available)
            self.output_dir_state = gr.State(self.output_dir)

            self.header_md = gr.Markdown(HEADER_BASE_MD, elem_classes=["main-header"])

            with gr.Row():
                # Left control panel
                with gr.Column(scale=2, elem_classes=["left-panel"]):
                    # Input textbox
                    if self.prompt_engineering_available:
                        input_place_holder = "Enter text to generate motion, support Chinese and English text input."
                    else:
                        input_place_holder = (
                            "Enter text to generate motion, please use `A person ...` format to describe the motion"
                        )

                    self.text_input = gr.Textbox(
                        label="📝 Input Text",
                        placeholder=input_place_holder,
                    )
                    # Rewritten textbox
                    self.rewritten_text = gr.Textbox(
                        label="✏️ Rewritten Text",
                        placeholder="Rewritten text will be displayed here, you can further edit",
                        interactive=True,
                        visible=False,
                    )
                    # Duration slider
                    self.duration_slider = gr.Slider(
                        minimum=0.5,
                        maximum=12,
                        value=5.0,
                        step=0.1,
                        label="⏱️ Action Duration (seconds)",
                        info="Feel free to adjust the action duration",
                    )

                    # Execute buttons
                    with gr.Row():
                        if self.prompt_engineering_available:
                            self.rewrite_btn = gr.Button(
                                "🔄 Rewrite Text",
                                variant="secondary",
                                size="lg",
                                elem_classes=["rewrite-button"],
                            )
                        else:
                            # Create a hidden/disabled placeholder button
                            self.rewrite_btn = gr.Button(
                                "🔄 Rewrite Text (Unavailable)",
                                variant="secondary",
                                size="lg",
                                elem_classes=["rewrite-button"],
                                interactive=False,
                                visible=False,
                            )

                        self.generate_btn = gr.Button(
                            "🚀 Generate Motion",
                            variant="primary",
                            size="lg",
                            elem_classes=["generate-button"],
                            interactive=not self.prompt_engineering_available,  # Enable directly if rewrite not available
                        )

                    if not self.prompt_engineering_available:
                        gr.Markdown(
                            "> ⚠️ **Prompt engineering is not available.** Text rewriting and duration estimation are disabled. Your input text and duration will be used directly."
                        )

                    # Example selection dropdown
                    self.example_dropdown = gr.Dropdown(
                        choices=self._get_example_choices(),
                        value="Custom Input",
                        label="📚 Test Examples",
                        info="Select a preset example or input your own text above",
                        interactive=True,
                    )

                    # Advanced settings
                    with gr.Accordion("🔧 Advanced Settings", open=False):
                        self._build_advanced_settings()

                    # Status message depends on whether rewrite is available
                    if self.prompt_engineering_available:
                        status_msg = "Please click the [🔄 Rewrite Text] button to rewrite the text first"
                    else:
                        status_msg = "Enter your text and click [🚀 Generate Motion] directly."

                    self.status_output = gr.Textbox(
                        label="📊 Status Information",
                        value=status_msg,
                    )

                    # FBX Download section
                    with gr.Row(visible=False) as self.fbx_download_row:
                        if model_inference.fbx_available:
                            self.fbx_files = gr.File(
                                label="📦 Download FBX Files",
                                file_count="multiple",
                                interactive=False,
                            )
                        else:
                            self.fbx_files = gr.State([])

                # Right display area
                with gr.Column(scale=3):
                    self.output_display = gr.HTML(
                        value=get_placeholder_html(), show_label=False, elem_classes=["flask-display"]
                    )

            # Footer
            gr.Markdown(FOOTER_MD, elem_classes=["footer"])

            self._bind_events()
            demo.load(fn=self._get_header_text, outputs=[self.header_md])
            return demo

    def _build_advanced_settings(self):
        # Only show rewrite options if rewrite is available
        if self.prompt_engineering_available:
            with gr.Group():
                gr.Markdown("### 🔄 Text Rewriting Options")
                with gr.Row():
                    self.enable_rewrite = gr.Checkbox(
                        label="Enable Text Rewriting",
                        value=True,
                        info="Automatically optimize text prompt to get better motion generation",
                    )

            with gr.Group():
                gr.Markdown("### ⏱️ Duration Settings")
                self.enable_duration_est = gr.Checkbox(
                    label="Enable Duration Estimation",
                    value=True,
                    info="Automatically estimate the duration of the motion",
                )
        else:
            # Create hidden placeholders with default values (disabled)
            self.enable_rewrite = gr.Checkbox(
                label="Enable Text Rewriting",
                value=False,
                visible=False,
            )
            self.enable_duration_est = gr.Checkbox(
                label="Enable Duration Estimation",
                value=False,
                visible=False,
            )
            with gr.Group():
                gr.Markdown("### ⚠️ Prompt Engineering Unavailable")
                gr.Markdown(
                    "Text rewriting and duration estimation are not available. "
                    "Your input text and duration will be used directly."
                )

        with gr.Group():
            gr.Markdown("### ⚙️ Generation Parameters")
            with gr.Row():
                with gr.Column(scale=3):
                    self.seed_input = gr.Textbox(
                        label="🎯 Random Seed List (comma separated)",
                        value="0,1,2,3",
                        placeholder="Enter comma separated seed list (e.g.: 0,1,2,3)",
                        info="Random seeds control the diversity of generated motions",
                    )
                with gr.Column(scale=1, min_width=60, elem_classes=["dice-container"]):
                    self.dice_btn = gr.Button(
                        "🎲 Lucky Button",
                        variant="secondary",
                        size="sm",
                        elem_classes=["dice-button"],
                    )

            self.cfg_slider = gr.Slider(
                minimum=1,
                maximum=10,
                value=5.0,
                step=0.1,
                label="⚙️ CFG Strength",
                info="Text fidelity: higher = more faithful to the prompt",
            )

    def _bind_events(self):
        # Generate random seeds
        self.dice_btn.click(self._generate_random_seeds, outputs=[self.seed_input])

        # Bind example selection event
        self.example_dropdown.change(
            fn=self._on_example_select,
            inputs=[self.example_dropdown],
            outputs=[self.text_input, self.seed_input, self.duration_slider],
        )

        # Rewrite text logic (only bind when rewrite is available)
        if self.prompt_engineering_available:
            self.rewrite_btn.click(fn=lambda: "Rewriting text, please wait...", outputs=[self.status_output]).then(
                self._prompt_engineering,
                inputs=[
                    self.text_input,
                    self.duration_slider,
                    self.enable_rewrite,
                    self.enable_duration_est,
                ],
                outputs=[self.rewritten_text, self.generate_btn, self.duration_slider],
            ).then(
                fn=lambda: (
                    gr.update(visible=True),
                    "Text rewriting completed! Please check and edit the rewritten text, then click [🚀 Generate Motion]",
                ),
                outputs=[self.rewritten_text, self.status_output],
            )

        # Generate motion logic
        self.generate_btn.click(
            fn=lambda: "Generating motion, please wait... (It takes some extra time for the first generation)",
            outputs=[self.status_output],
        ).then(
            generate_motion_func,
            inputs=[self.text_input, self.rewritten_text, self.seed_input, self.duration_slider, self.cfg_slider],
            outputs=[self.output_display, self.fbx_files],
        ).then(
            fn=lambda fbx_list: (
                (
                    "🎉 Motion generation completed! You can view the motion visualization result on the right. FBX files are ready for download."
                    if fbx_list
                    else "🎉 Motion generation completed! You can view the motion visualization result on the right"
                ),
                gr.update(visible=bool(fbx_list)),
            ),
            inputs=[self.fbx_files],
            outputs=[self.status_output, self.fbx_download_row],
        )

        # Reset logic - different behavior based on rewrite availability
        if self.prompt_engineering_available:
            self.text_input.change(
                fn=lambda: (
                    gr.update(visible=False),
                    gr.update(interactive=False),
                    "Please click the [🔄 Rewrite Text] button to rewrite the text first",
                ),
                outputs=[self.rewritten_text, self.generate_btn, self.status_output],
            )
        else:
            # When rewrite is not available, enable generate button directly when text is entered
            self.text_input.change(
                fn=lambda text: (
                    gr.update(visible=False),
                    gr.update(interactive=bool(text.strip())),
                    (
                        "Ready to generate! Click [🚀 Generate Motion] to start."
                        if text.strip()
                        else "Enter your text and click [🚀 Generate Motion] directly."
                    ),
                ),
                inputs=[self.text_input],
                outputs=[self.rewritten_text, self.generate_btn, self.status_output],
            )
        # Only bind rewritten_text change when rewrite is available
        if self.prompt_engineering_available:
            self.rewritten_text.change(
                fn=lambda text: (
                    gr.update(interactive=bool(text.strip())),
                    (
                        "Rewritten text has been modified, you can click [🚀 Generate Motion]"
                        if text.strip()
                        else "Rewritten text cannot be empty, please enter valid text"
                    ),
                ),
                inputs=[self.rewritten_text],
                outputs=[self.generate_btn, self.status_output],
            )


def create_demo(final_model_path):
    """Create the Gradio demo with Zero GPU support."""

    class Args:
        model_path = final_model_path
        output_dir = "output/gradio"
        use_prompt_engineering = False
        use_text_encoder = True
        prompt_engineering_host = os.environ.get("PROMPT_HOST", None)
        prompt_engineering_model_path = os.environ.get("PROMPT_MODEL_PATH", None)
        disable_prompt_engineering = os.environ.get("DISABLE_PROMPT_ENGINEERING", False)

    args = Args()

    # Check required files:
    cfg = osp.join(args.model_path, "config.yml")
    ckpt = osp.join(args.model_path, "latest.ckpt")
    if not osp.exists(cfg):
        raise FileNotFoundError(f">>> Configuration file not found: {cfg}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # For Zero GPU: Don't load model at startup, use lazy loading
    # Create a minimal runtime for UI initialization (without model loading)
    ui = T2MGradioUI(args=args)
    demo = ui.build_ui()
    return demo


# Create demo at module level for Hugging Face Spaces
# Pre-download text encoder models first (without loading)


if __name__ == "__main__":
    # Create demo at module level for Hugging Face Spaces
    try_to_download_text_encoder()
    # Then download the main model
    final_model_path = try_to_download_model()

    # Read quantization from environment variable (set by profile)
    # Options: "none" (24-26GB), "int8" (12-13GB), "int4" (6-8GB)
    import os
    quantization = os.environ.get('HY_QUANTIZATION', 'int4')  # Default to int4 for safety
    print(f"Using quantization: {quantization}")

    model_inference = ModelInference(final_model_path, use_prompt_engineering=False, use_text_encoder=True, quantization=quantization)

    # Try to use CUDA if available, fallback to CPU
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f">>> Using device: {device}")
    if device == "cuda":
        print(f">>> GPU: {torch.cuda.get_device_name(0)}")
        print(f">>> VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    model_inference.initialize_model(device=device)
    demo = create_demo(final_model_path)
    demo.launch(server_name="0.0.0.0")
