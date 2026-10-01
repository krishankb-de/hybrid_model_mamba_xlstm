"""Upload intake and the published image transform (CHAT_UI_PLAN.md P2-C).

The model must see exactly what every published number saw: Resize((224,224)) -> Grayscale(3) ->
ToTensor -> Normalize(CLIP mean/std), the transform in run_checkpoint_inspection. Uploads are not
MIMIC JPEGs, so two things happen first that MIMIC never needed: EXIF orientation is applied (the
model must see what the user sees) and 16-bit grayscale is rescaled to 8-bit (PIL's convert would
clip it to white). For an 8-bit, EXIF-free grayscale JPEG, both are no-ops.

Only UploadError leaves load_upload, and its message is plain text for the user: the bounds are checked
from the header before any pixel is decoded, and unreadable metadata never escapes as a Python exception.
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
TOO_LARGE_MSG = "Image is larger than {} MB.".format(MAX_UPLOAD_BYTES // (1024 * 1024))
TOO_MANY_PIXELS_MSG = "Image is larger than {} megapixels.".format(MAX_PIXELS // 1_000_000)
TOO_SMALL_MSG = "Each side must be at least {} pixels.".format(MIN_SIDE)
UNREADABLE_MSG = "Could not read the image. The file may be damaged or incomplete."
ORIENTATION_MSG = "Could not read the image's orientation data. Re-export it as PNG or JPEG."
EXIF_ORIENTATION = 0x0112
TRANSPOSING_ORIENTATIONS = (2, 3, 4, 5, 6, 7, 8)   # the values ImageOps.exif_transpose acts on


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
    """16-bit grayscale -> 8-bit by min-max rescale (PIL's own convert clips it to white).

    Known limit: Pillow decodes 16-bit RGB, RGBA and gray+alpha PNGs straight to 8-bit and keeps only the high
    byte, so those never reach this function and are not rescaled (a 12-bit RGB export comes out near black).
    """
    if img.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        arr = np.array(img, dtype=np.float32)          # float32 and in place: one float64 copy of 49 MP is 392 MB
        lo, hi = float(arr.min()), float(arr.max())
        arr -= lo
        arr *= 255.0 / max(hi - lo, 1.0)
        np.rint(arr, out=arr)
        return Image.fromarray(np.clip(arr, 0, 255, out=arr).astype(np.uint8))
    return img


def load_upload(data: bytes) -> Tuple[Image.Image, Dict[str, Any]]:
    """bytes -> the 8-bit RGB image the model and the viewer both use, plus facts for the stage detail."""
    if len(data) > MAX_UPLOAD_BYTES:
        raise UploadError(TOO_LARGE_MSG)
    fmt = sniff_format(data)
    try:
        img = Image.open(io.BytesIO(data))             # reads the header only
        img.seek(0)                                    # an animated file contributes its first frame
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise UploadError(TOO_MANY_PIXELS_MSG) from None
    except Exception:   # not a readable image of the sniffed format
        raise UploadError(UNREADABLE_MSG) from None
    # the bounds come from the header, before a pixel is decoded (EXIF orientation swaps the sides, not the minimum)
    if img.width * img.height > MAX_PIXELS:
        raise UploadError(TOO_MANY_PIXELS_MSG)
    if min(img.size) < MIN_SIDE:
        raise UploadError(TOO_SMALL_MSG)
    try:
        img.load()
    except Exception:   # truncated or corrupt pixel data
        raise UploadError(UNREADABLE_MSG) from None
    mode_in = img.mode
    try:
        orientation = img.getexif().get(EXIF_ORIENTATION, 1)
    except Exception:   # unreadable EXIF (bad TIFF header, bad hex in a PNG text chunk): treat it as upright
        orientation = 1
    # exif_transpose returns a copy even when it does nothing, so only the tag says whether it acted
    exif_transposed = orientation in TRANSPOSING_ORIENTATIONS
    if exif_transposed:
        try:
            img = ImageOps.exif_transpose(img)
        except Exception:   # it rewrites the metadata after rotating and trips on malformed entries
            raise UploadError(ORIENTATION_MSG) from None
    img = _to_8bit(img).convert("RGB")
    return img, {"format": fmt, "mode": mode_in, "input_px": list(img.size), "exif_transposed": exif_transposed}


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
