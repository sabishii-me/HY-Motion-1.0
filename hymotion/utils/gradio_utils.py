import os
from huggingface_hub import snapshot_download

# 本地模型路径配置（如果已经下载，直接使用本地路径）
QWEN_LOCAL_PATH = "ckpts/Qwen3-8B"
CLIP_LOCAL_PATH = "ckpts/clip-vit-large-patch14"

def try_to_download_text_encoder():
    """
    Pre-download text encoder models (Qwen3-8B and CLIP) to local cache.
    This ensures the models are cached locally before they are needed,
    so later loading will not require downloading again.

    If models already exist in local paths (ckpts/), skip downloading.
    """
    # Text encoder model IDs (same as in hymotion/network/text_encoders/text_encoder.py)
    QWEN_REPO_ID = "Qwen/Qwen3-8B"
    CLIP_REPO_ID = "openai/clip-vit-large-patch14"

    token = os.environ.get("HF_TOKEN", None)
    if token is None:
        token = ""

    # 检查 Qwen3-8B 是否已存在
    if os.path.exists(QWEN_LOCAL_PATH) and os.path.isdir(QWEN_LOCAL_PATH):
        print(f">>> Found local Qwen model at: {QWEN_LOCAL_PATH}, skipping download.")
    else:
        print(f">>> Pre-downloading text encoder: {QWEN_REPO_ID} to {QWEN_LOCAL_PATH}")
        try:
            snapshot_download(
                repo_id=QWEN_REPO_ID,
                local_dir=QWEN_LOCAL_PATH,
                token=token,
            )
            print(f">>> Successfully pre-downloaded: {QWEN_REPO_ID}")
        except Exception as e:
            print(f">>> [WARNING] Failed to pre-download {QWEN_REPO_ID}: {e}")

    # 检查 CLIP 是否已存在
    if os.path.exists(CLIP_LOCAL_PATH) and os.path.isdir(CLIP_LOCAL_PATH):
        print(f">>> Found local CLIP model at: {CLIP_LOCAL_PATH}, skipping download.")
    else:
        print(f">>> Pre-downloading text encoder: {CLIP_REPO_ID} to {CLIP_LOCAL_PATH}")
        try:
            snapshot_download(
                repo_id=CLIP_REPO_ID,
                local_dir=CLIP_LOCAL_PATH,
                token=token,
            )
            print(f">>> Successfully pre-downloaded: {CLIP_REPO_ID}")
        except Exception as e:
            print(f">>> [WARNING] Failed to pre-download {CLIP_REPO_ID}: {e}")

    print(">>> Text encoder pre-download complete.")


def try_to_download_model():
    repo_id = "tencent/HY-Motion-1.0"
    target_folder = "HY-Motion-1.0-Lite"
    print(f">>> start download ", repo_id, target_folder)
    token = os.environ.get("HF_TOKEN", None)
    if token is None:
        token = ""
    local_dir = snapshot_download(
        repo_id=repo_id,
        allow_patterns=f"{target_folder}/*",
        local_dir="./downloaded_models",
        token=token
    )
    final_model_path = os.path.join(local_dir, target_folder)
    print(f">>> Final model path: {final_model_path}")
    return final_model_path