"""
trt_engine.utils - Python utility functions for the TensorRT inference engine.
"""

import os
import sys
from typing import List, Optional, Tuple

import numpy as np


def load_image(path: str, size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Load and preprocess an image for inference.

    Reads an image from disk, resizes it, converts to CHW float32 format,
    and normalizes with ImageNet mean/std.

    Args:
        path: Path to the image file.
        size: Target (height, width) tuple.

    Returns:
        Preprocessed numpy array of shape (1, 3, H, W) in float32.
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError(
            "Pillow is required for image loading. Install with: pip install Pillow"
        )

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image not found: {path}")

    img = Image.open(path).convert("RGB")
    img = img.resize((size[1], size[0]), Image.BILINEAR)

    # Convert to numpy (H, W, C) uint8 -> float32
    arr = np.array(img, dtype=np.float32) / 255.0

    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std

    # HWC -> CHW
    arr = arr.transpose(2, 0, 1)

    # Add batch dimension -> (1, 3, H, W)
    arr = np.expand_dims(arr, axis=0)

    return arr.astype(np.float32)


def download_model(url: str, output_path: str) -> str:
    """Download a model file from a URL.

    Args:
        url: URL to fetch the model from.
        output_path: Local file path to save the downloaded model.

    Returns:
        The output_path on success.
    """
    try:
        import urllib.request
    except ImportError:
        raise ImportError("urllib is required for downloading models.")

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    print(f"Downloading model from {url} ...")
    urllib.request.urlretrieve(url, output_path)
    file_size = os.path.getsize(output_path)
    print(f"Downloaded to {output_path} ({file_size} bytes)")

    return output_path


def visualize_results(
    results: np.ndarray,
    labels: Optional[List[str]] = None,
    top_k: int = 5,
) -> None:
    """Print formatted classification results.

    Args:
        results: 1-D numpy array of class scores / probabilities.
        labels: Optional list mapping class index to label name.
        top_k: Number of top predictions to display.
    """
    if not isinstance(results, np.ndarray):
        results = np.array(results, dtype=np.float32)

    results = results.flatten()

    # Apply softmax if values don't sum to ~1
    total = results.sum()
    if abs(total - 1.0) > 0.1:
        exp_vals = np.exp(results - np.max(results))
        results = exp_vals / exp_vals.sum()

    top_indices = np.argsort(results)[::-1][:top_k]

    print(f"{'Rank':<6} {'Index':<8} {'Score':<12} {'Label'}")
    print("-" * 50)
    for rank, idx in enumerate(top_indices, 1):
        score = results[idx]
        label = labels[idx] if labels and idx < len(labels) else "N/A"
        print(f"{rank:<6} {idx:<8} {score:<12.6f} {label}")
