"""Tests for image loader and validation."""

import io
import numpy as np
import pytest
from PIL import Image

from vga_emi.ingestion.image_loader import load_and_validate_image
from vga_emi.extraction.trace_detection import extract_trace_pixels


def test_load_valid_png_bytes():
    # Create simple valid test image in memory
    img = Image.new("RGB", (200, 100), color=(15, 20, 30))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    arr, pil_img = load_and_validate_image(buf.getvalue(), filename="test.png")
    assert arr.shape == (100, 200, 3)
    assert pil_img.size == (200, 100)


def test_reject_corrupted_image():
    corrupted_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDRcorruptedcontent"
    with pytest.raises(ValueError, match="Corrupted or invalid image"):
        load_and_validate_image(corrupted_bytes, filename="corrupted.png")


def test_reject_unsupported_format():
    with pytest.raises(ValueError, match="Unsupported image format"):
        load_and_validate_image(b"dummy", filename="document.pdf")


def test_trace_extraction_synthetic_mask():
    # Create black image with bright yellow horizontal bar
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # Yellow is R=255, G=255, B=0
    img[50:53, 10:90] = [255, 255, 0]

    bounds = (10, 10, 90, 90)
    x_px, y_px, conf, mask = extract_trace_pixels(img, bounds, target_color="yellow")

    assert len(x_px) > 50
    assert conf > 0.5
    # y should be around 51
    assert np.allclose(y_px, 51.0, atol=2.0)
