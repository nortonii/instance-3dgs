#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Sam3Model, Sam3Processor


DEFAULT_PROMPTS: Sequence[str] = (
    "person",
    "human",
    "shopper",
    "shopping cart",
    "shopping trolley",
    "shopping basket",
    "handbag",
    "bag carried by person",
)


def parse_args() -> argparse.Namespace:
    project_root = Path(os.environ.get("OPENLORIS_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
    parser = argparse.ArgumentParser(
        description="Generate OpenLORIS ignore masks for dynamic supermarket objects with SAM3."
    )
    parser.add_argument("--model-dir", type=Path, default=Path(os.environ.get("SAM3_MODEL_DIR", "~/models/sam3")).expanduser())
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=project_root / "dataset_market1_1_fullspan" / "images",
    )
    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=project_root / "dataset_market1_1_fullspan" / "ignore_masks",
        help="Directory that gsplat will consume directly.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=project_root / "dataset_market1_1_fullspan" / "sam3_dynamic_masks",
        help="Directory for overlays and metadata.",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--min-mask-area", type=int, default=256)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--prompts", nargs="+", default=list(DEFAULT_PROMPTS))
    return parser.parse_args()


def discover_images(image_dir: Path, max_images: int | None) -> List[Path]:
    image_paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if max_images is not None:
        image_paths = image_paths[:max_images]
    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")
    return image_paths


def prepare_dirs(mask_dir: Path, artifact_dir: Path) -> Dict[str, Path]:
    paths = {
        "masks": mask_dir,
        "masked_images": artifact_dir / "masked_images",
        "overlays": artifact_dir / "overlays",
        "metadata": artifact_dir / "metadata",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def tensor_to_bool_masks(masks: torch.Tensor | np.ndarray, min_mask_area: int) -> List[np.ndarray]:
    if isinstance(masks, torch.Tensor):
        masks_np = masks.detach().cpu().numpy()
    else:
        masks_np = masks
    if masks_np.size == 0:
        return []
    masks_np = masks_np.astype(bool)
    return [mask for mask in masks_np if int(mask.sum()) >= min_mask_area]


def alpha_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    overlay = image.copy().astype(np.float32)
    color_arr = np.array((255, 0, 0), dtype=np.float32)
    overlay[mask] = (1.0 - alpha) * overlay[mask] + alpha * color_arr
    return np.clip(overlay, 0, 255).astype(np.uint8)


def load_model(model_dir: Path) -> tuple[Sam3Processor, Sam3Model, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    processor = Sam3Processor.from_pretrained(model_dir)
    model = Sam3Model.from_pretrained(model_dir, torch_dtype=dtype).to(device)
    model.eval()
    return processor, model, device


def run_prompt(
    processor: Sam3Processor,
    model: Sam3Model,
    device: torch.device,
    images: Sequence[Image.Image],
    prompt: str,
    threshold: float,
    mask_threshold: float,
) -> List[Dict]:
    inputs = processor(images=list(images), text=[prompt] * len(images), return_tensors="pt")
    target_sizes = inputs["original_sizes"].tolist()
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    with torch.inference_mode():
        outputs = model(**inputs)
    return processor.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        mask_threshold=mask_threshold,
        target_sizes=target_sizes,
    )


def main() -> None:
    args = parse_args()
    image_paths = discover_images(args.image_dir, args.max_images)
    output_paths = prepare_dirs(args.mask_dir, args.artifact_dir)
    processor, model, device = load_model(args.model_dir)

    summary = {
        "model_dir": str(args.model_dir),
        "image_dir": str(args.image_dir),
        "mask_dir": str(args.mask_dir),
        "artifact_dir": str(args.artifact_dir),
        "num_images": len(image_paths),
        "threshold": args.threshold,
        "mask_threshold": args.mask_threshold,
        "min_mask_area": args.min_mask_area,
        "prompts": list(args.prompts),
        "device": str(device),
    }

    for start in tqdm(range(0, len(image_paths), args.batch_size), desc="Masking"):
        batch_paths = image_paths[start : start + args.batch_size]
        pil_images = [Image.open(path).convert("RGB") for path in batch_paths]
        rgb_images = [np.array(image) for image in pil_images]
        combined_masks = [np.zeros(image.shape[:2], dtype=bool) for image in rgb_images]
        batch_metadata = [{"image": path.name, "prompts": [], "total_mask_area": 0} for path in batch_paths]

        for prompt in args.prompts:
            results = run_prompt(
                processor=processor,
                model=model,
                device=device,
                images=pil_images,
                prompt=prompt,
                threshold=args.threshold,
                mask_threshold=args.mask_threshold,
            )
            for idx, result in enumerate(results):
                masks = tensor_to_bool_masks(result.get("masks", np.empty((0,))), args.min_mask_area)
                if masks:
                    combined_masks[idx] |= np.logical_or.reduce(masks)
                batch_metadata[idx]["prompts"].append(
                    {
                        "prompt": prompt,
                        "num_instances": len(masks),
                        "scores": [float(score) for score in result.get("scores", [])[: len(masks)]],
                    }
                )

        for path, image_np, combined_mask, metadata in zip(batch_paths, rgb_images, combined_masks, batch_metadata):
            metadata["total_mask_area"] = int(combined_mask.sum())

            mask_img = Image.fromarray((combined_mask.astype(np.uint8)) * 255, mode="L")
            mask_img.save(output_paths["masks"] / path.name)

            masked_image = image_np.copy()
            masked_image[combined_mask] = 0
            Image.fromarray(masked_image).save(output_paths["masked_images"] / path.name)

            overlay_image = alpha_overlay(image_np, combined_mask)
            Image.fromarray(overlay_image).save(output_paths["overlays"] / path.name)

            with open(output_paths["metadata"] / f"{path.stem}.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

    with open(args.artifact_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
