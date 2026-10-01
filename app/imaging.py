"""Upload intake and the published image transform (CHAT_UI_PLAN.md P2-C).

The model must see exactly what every published number saw: Resize((224,224)) -> Grayscale(3) ->
ToTensor -> Normalize(CLIP mean/std), the transform in run_checkpoint_inspection. Uploads are not
MIMIC JPEGs, so two things happen first that MIMIC never needed: EXIF orientation is applied (the
model must see what the user sees) and 16-bit grayscale is rescaled to 8-bit (PIL's convert would
clip it to white). For an 8-bit, EXIF-free grayscale JPEG, both are no-ops.
"""
import io
from typing import Any, Callable, Dict, Tuple

import numpy as np
from PIL import Image, ImageOps

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MIN_SIDE = 64
MAX_PIXELS = 50_000_000
FORMATS_MSG = "Use a PNG, JPEG or WEBP image."


class UploadError(ValueError):
    """Shown to the user verbatim."""


def sniff_format(data: bytes) -> str:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "PNG"
    if data[:3] == b"\xff\xd8\xff":
        return "JPEG"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "WEBP"
    if data[128:132] == b"DICM":
        raise UploadError("DICOM is not supported. Export the image as PNG or JPEG first.")
    raise UploadError(FORMATS_MSG)


def _to_8bit(img: Image.Image) -> Image.Image:
    if img.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        arr = np.asarray(img, dtype=np.float64)
        lo, hi = float(arr.min()), float(arr.max())
        arr = (arr - lo) / max(hi - lo, 1.0) * 255.0
        return Image.fromarray(np.clip(np.rint(arr), 0, 255).astype(np.uint8))
    return img


def load_upload(data: bytes) -> Tuple[Image.Image, Dict[str, Any]]:
    """bytes -> the 8-bit RGB image the model and the viewer both use, plus facts for the stage detail."""
    if len(data) > MAX_UPLOAD_BYTES:
        raise UploadError("Image is larger than 20 MB.")
    fmt = sniff_format(data)
    try:
        img = Image.open(io.BytesIO(data))
        if img.width * img.height > MAX_PIXELS:
            raise UploadError("Image is larger than 50 megapixels.")
        img.seek(0)
        img.load()
    except UploadError:
        raise
    except Exception as exc:   # truncated or corrupt file, decompression bomb
        raise UploadError("Could not read the image ({}).".format(type(exc).__name__))
    mode_in = img.mode
    # exif_transpose returns a copy even when it does nothing, so only the tag says whether it acted
    # (2-8 are the orientations it handles; 1, no tag, or a junk value leaves the pixels alone)
    exif_transposed = img.getexif().get(0x0112, 1) in (2, 3, 4, 5, 6, 7, 8)
    img = _to_8bit(ImageOps.exif_transpose(img)).convert("RGB")
    if min(img.size) < MIN_SIDE:
        raise UploadError("Image is smaller than 64×64 pixels.")
    return img, {"format": fmt, "mode": mode_in, "input_px": list(img.size), "exif_transposed": bool(exif_transposed)}


def model_transform() -> Callable:
    import torchvision.transforms as T
    return T.Compose([T.Resize((224, 224)), T.Grayscale(num_output_channels=3), T.ToTensor(),
                      T.Normalize(mean=list(CLIP_MEAN), std=list(CLIP_STD))])


def model_input_image(img: Image.Image) -> Image.Image:
    """The 224×224 image the tower sees, before ToTensor/Normalize (lossless as PNG)."""
    import torchvision.transforms as T
    return T.Grayscale(num_output_channels=3)(T.Resize((224, 224))(img))


def thumbnail_jpeg(img: Image.Image, max_side: int = 512) -> bytes:
    thumb = img.copy()
    thumb.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    thumb.convert("RGB").save(buf, "JPEG", quality=88)
    return buf.getvalue()
