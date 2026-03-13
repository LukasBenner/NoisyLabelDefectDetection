#!/usr/bin/env python3
"""
Remove white backgrounds from generated images and composite onto
real background images.

Run this after generate.py has produced images with white backgrounds.

Install:
    pip install Pillow transparent-background

Usage:
    python composite.py \
        --input_dir ./output/raw/water_stain \
        --background_dir ./backgrounds \
        --out_dir ./output/composite/water_stain
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

from PIL import Image


IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

_remover = None

def get_remover():
    global _remover
    if _remover is None:
        from transparent_background import Remover
        _remover = Remover()
    return _remover


def remove_background(img: Image.Image) -> Image.Image:
    """Remove background using transparent-background (InSPyReNet). Returns RGBA."""
    return get_remover().process(img.convert("RGB"), type="rgba")


def composite_on_background(
    foreground: Image.Image,
    background: Image.Image,
) -> Image.Image:
    """Remove background and composite onto a background image."""
    fg_rgba = remove_background(foreground)
    bg = background.convert("RGBA").resize(fg_rgba.size, Image.Resampling.LANCZOS)
    bg.paste(fg_rgba, (0, 0), fg_rgba)
    return bg.convert("RGB")


def list_images(directory: Path) -> List[Path]:
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")
    paths = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    if not paths:
        raise ValueError(f"No images found in {directory}")
    return sorted(paths)


@dataclass
class ProgressBar:
    total: int
    width: int = 30
    stream: Any = sys.stdout
    _done: int = 0
    _last_len: int = 0
    _start_time: float = time.time()

    def update(self, message: str = "") -> None:
        self._done += 1
        total = max(self.total, 1)
        filled = int(self.width * self._done / total)
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = max(time.time() - self._start_time, 1e-6)
        avg = elapsed / self._done
        remaining = max(self.total - self._done, 0)
        eta = remaining * avg
        line = f"[{bar}] {self._done}/{self.total}"
        line += f" | elapsed {elapsed:6.1f}s avg {avg:5.2f}s ETA {eta:6.1f}s"
        if message:
            line += f" {message}"
        padded = line + " " * max(0, self._last_len - len(line))
        self.stream.write("\r" + padded)
        self.stream.flush()
        self._last_len = len(line)

    def finish(self) -> None:
        if self._last_len:
            self.stream.write("\n")
            self.stream.flush()


def main() -> None:
    p = argparse.ArgumentParser(description="Composite generated images onto backgrounds")
    p.add_argument("--input_dir", required=True, help="Directory with white-background images")
    p.add_argument("--background_dir", required=True, help="Directory of background images")
    p.add_argument("--out_dir", required=True, help="Output directory for composited images")
    p.add_argument("--rng_seed", type=int, default=1337, help="RNG seed for background selection")
    args = p.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(input_dir)
    backgrounds = list_images(Path(args.background_dir).expanduser().resolve())
    rng = random.Random(args.rng_seed)

    print(f"Processing {len(images)} images with {len(backgrounds)} backgrounds.")

    progress = ProgressBar(total=len(images))

    for img_path in images:
        fg = Image.open(img_path)
        bg_path = rng.choice(backgrounds)
        bg = Image.open(bg_path)

        result = composite_on_background(fg, bg)
        result.save(out_dir / img_path.name)

        progress.update(message=img_path.name)

    progress.finish()
    print("All done.")


if __name__ == "__main__":
    main()
