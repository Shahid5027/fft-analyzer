"""Image loader and validator for oscilloscope screenshots."""

from __future__ import annotations

import io
from pathlib import Path
from typing import BinaryIO, Optional, Tuple, Union
import numpy as np
from PIL import Image


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def load_and_validate_image(
    source: Union[str, Path, BinaryIO, bytes],
    filename: Optional[str] = None,
) -> Tuple[np.ndarray, Image.Image]:
    """Load and validate an image file, ensuring it is a valid PNG, JPG, or JPEG.

    Args:
        source: File path, bytes, or file-like object.
        filename: Optional filename for extension checking.

    Returns:
        Tuple of (numpy RGB array uint8, PIL Image object).

    Raises:
        ValueError: If file extension is unsupported or content is corrupted.
    """
    if filename is None and isinstance(source, (str, Path)):
        filename = str(source)

    if filename:
        ext = Path(filename).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported image format '{ext}'. Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

    try:
        if isinstance(source, (str, Path)):
            img = Image.open(source)
        elif isinstance(source, bytes):
            img = Image.open(io.BytesIO(source))
        else:
            img = Image.open(source)

        # Force verification of image integrity
        img.verify()

        # Reopen after verify() because verify() exhausts image state
        if isinstance(source, (str, Path)):
            img = Image.open(source)
        elif isinstance(source, bytes):
            img = Image.open(io.BytesIO(source))
        else:
            source.seek(0)
            img = Image.open(source)

        # Convert to RGB (handles RGBA, grayscale, indexed palettes)
        img_rgb = img.convert("RGB")
        img_array = np.array(img_rgb, dtype=np.uint8)

        if img_array.ndim != 3 or img_array.shape[2] != 3:
            raise ValueError("Failed to convert image to standard 3-channel RGB.")

        return img_array, img_rgb

    except Exception as exc:
        raise ValueError(f"Corrupted or invalid image file: {str(exc)}") from exc
