#!/usr/bin/env python3
"""Generate BLIP2 captions for all images in a folder.

Creates a text file next to each image named <image_stem>.txt.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def iter_images(folder: Path):
    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def main() -> None:
    parser = argparse.ArgumentParser(description="Caption images with BLIP2.")
    parser.add_argument("folder", type=Path, help="Folder with images")
    parser.add_argument(
        "--model",
        type=str,
        default="Salesforce/blip2-flan-t5-xl",
        help="BLIP2 model id",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Describe this image in a concise caption.",
        help="Prompt to guide captioning",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .txt captions",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=40,
        help="Maximum tokens for generated caption",
    )
    args = parser.parse_args()

    folder = args.folder
    if not folder.exists():
        raise SystemExit(f"Folder not found: {folder}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    processor = AutoProcessor.from_pretrained(args.model)
    model = Blip2ForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=dtype
    ).to(device)
    model.eval()

    for img_path in iter_images(folder):
        caption_path = img_path.with_suffix(".txt")
        if caption_path.exists() and not args.overwrite:
            continue

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        inputs = processor(images=image, text=args.prompt, return_tensors="pt").to(
            device, dtype
        )
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
            )
        caption = processor.decode(output[0], skip_special_tokens=True).strip()
        caption_path.write_text(caption + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
