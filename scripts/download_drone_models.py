"""Download the Semantic-Drone-Dataset U-Net checkpoints.

The checkpoints released by Fábio Luiz Junior
(https://github.com/fabioluiz1987/semantic_segmentation_models) are stored
with Git LFS. This script fetches the actual binaries via the raw *media*
endpoint (no ``git-lfs`` required) into ``models/<Backbone>/<file>.pt`` — the
exact layout ``DroneUNetPreProcessor`` expects.

Usage::

    python scripts/download_drone_models.py                 # all 5 models
    python scripts/download_drone_models.py efficientnet-b4 # a single one
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.parse
import urllib.request

_MEDIA_BASE = (
    "https://media.githubusercontent.com/media/"
    "fabioluiz1987/semantic_segmentation_models/main/models"
)

# friendly name → (remote sub-path, local sub-path)
MODELS: dict[str, tuple[str, str]] = {
    "resnet50": ("ResNet 50/resnet50.pt", "ResNet 50/resnet50.pt"),
    "resnet101": ("ResNet 101/resnet101.pt", "ResNet 101/resnet101.pt"),
    "efficientnet-b4": ("EfficientNet-B4/efficientnet-b4.pt", "EfficientNet-B4/efficientnet-b4.pt"),
    "efficientnet-b5": ("EfficientNet-B5/efficientnet-b5.pt", "EfficientNet-B5/efficientnet-b5.pt"),
    "mobilenet_v2": ("MobileNet V2/Unet-Mobilenet_v2.pt", "MobileNet V2/Unet-Mobilenet_v2.pt"),
}

# All released backbones (downloaded by default).
DEFAULT_MODELS = ["resnet50", "resnet101", "efficientnet-b4", "efficientnet-b5", "mobilenet_v2"]

_MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
)


def _progress(count: int, block: int, total: int) -> None:
    if total <= 0:
        return
    pct = min(100, count * block * 100 // total)
    if pct == getattr(_progress, "_last", -1):
        return  # only redraw when the whole-percent value changes
    _progress._last = pct
    mb = count * block / 1e6
    sys.stdout.write(f"\r    {pct:3d}%  ({mb:6.1f} MB)")
    sys.stdout.flush()


def _download(name: str, force: bool) -> None:
    remote, local = MODELS[name]
    url = f"{_MEDIA_BASE}/{urllib.parse.quote(remote)}"
    dest = os.path.join(_MODELS_DIR, local)
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    if os.path.isfile(dest) and not force and os.path.getsize(dest) > 1_000_000:
        print(f"[skip] {name} already present ({os.path.getsize(dest) / 1e6:.1f} MB)")
        return

    print(f"[get ] {name}  <-  {url}")
    urllib.request.urlretrieve(url, dest, _progress)
    print(f"\n[ok  ] {name}  ->  {dest} ({os.path.getsize(dest) / 1e6:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "models",
        nargs="*",
        help=f"Models to download (default: all {DEFAULT_MODELS}).",
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-download even if present."
    )
    args = parser.parse_args()

    targets = args.models if args.models else DEFAULT_MODELS

    unknown = [m for m in targets if m not in MODELS]
    if unknown:
        print(f"Unknown model(s): {unknown}. Available: {list(MODELS)}")
        sys.exit(1)

    for name in targets:
        _download(name, args.force)

    print("\nDone. Checkpoints live under models/. Select them in the pipeline as "
          "'drone_<name>' (e.g. drone_efficientnet-b4).")


if __name__ == "__main__":
    main()
