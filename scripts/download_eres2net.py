"""
Download ERes2NetV2 pretrained weights from GPT-SoVITS.

ERes2NetV2 is the speaker encoder used in GPT-SoVITS for
voice timbre extraction. This script downloads the pretrained checkpoint.
"""

import os
from huggingface_hub import hf_hub_download

REPO_ID = "lj1995/GPT-SoVITS"
FILENAME = "sv/pretrained_eres2netv2w24s4ep4.ckpt"
LOCAL_DIR = "pretrained_models/sv"
LOCAL_PATH = os.path.join(LOCAL_DIR, "pretrained_eres2netv2w24s4ep4.ckpt")


def download_eres2net():
    """Download ERes2NetV2 pretrained weights."""
    os.makedirs(LOCAL_DIR, exist_ok=True)

    print(f"Downloading {FILENAME} from {REPO_ID}...")
    print(f"Saving to {LOCAL_PATH}")

    try:
        hf_hub_download(
            repo_id=REPO_ID,
            filename=FILENAME,
            local_dir=LOCAL_DIR,
            local_dir_use_symlinks=False
        )
        print(f"✅ Downloaded to {LOCAL_PATH}")
        return LOCAL_PATH
    except Exception as e:
        print(f"❌ Failed to download: {e}")
        print(f"\nYou can download manually from:")
        print(f"https://huggingface.co/{REPO_ID}/blob/main/{FILENAME}")
        print(f"\nAnd save to: {LOCAL_PATH}")
        return None


if __name__ == "__main__":
    download_eres2net()
