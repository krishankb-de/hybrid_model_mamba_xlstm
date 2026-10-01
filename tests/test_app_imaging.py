"""CHAT_UI_PLAN.md P2-C: uploads become exactly what the published transform expects."""
import io

import numpy as np
import pytest
import torch
import torchvision.transforms as T
from PIL import Image

from app.imaging import (CLIP_MEAN, CLIP_STD, UploadError, load_upload, model_input_image,
                         model_transform, thumbnail_jpeg)
from tests.app_helpers import jpeg_bytes, png_bytes

PUBLISHED = T.Compose([   # verbatim from scripts/evaluate_report_generation.py run_checkpoint_inspection
    T.Resize((224, 224)), T.Grayscale(num_output_channels=3), T.ToTensor(),
    T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
])


def _gray_rgb(w=320, h=320):
    return Image.open(io.BytesIO(png_bytes(w, h))).convert("RGB")


def test_16bit_png_is_rescaled_not_saturated():   # Review Focus 1
    ramp = np.linspace(0, 4095, 256 * 256).reshape(256, 256).astype(np.uint16)   # 12-bit data
    buf = io.BytesIO()
    Image.fromarray(ramp).save(buf, "PNG")
    img, facts = load_upload(buf.getvalue())
    arr = np.asarray(img.convert("L"))
    assert arr.min() <= 5 and arr.max() >= 250, (arr.min(), arr.max())
    assert facts["mode"].startswith("I")


def test_exif_rotation_is_applied_before_the_model_sees_it():   # Review Focus 5
    base = Image.new("RGB", (200, 100), (0, 0, 0))
    exif = Image.Exif()
    exif[0x0112] = 6                                   # display rotated 90° clockwise
    buf = io.BytesIO()
    base.save(buf, "JPEG", exif=exif.tobytes())
    img, facts = load_upload(buf.getvalue())
    assert img.size == (100, 200) and facts["exif_transposed"] is True


def test_unrotated_jpeg_reports_no_exif_transposition():   # exif_transpose returns a copy even when it does nothing
    img, facts = load_upload(jpeg_bytes(320, 240))
    assert img.size == (320, 240) and facts["exif_transposed"] is False


def _jpeg_with_orientation(tag, size=(200, 100)):
    base = Image.new("RGB", size, (0, 0, 0))
    base.paste((255, 255, 255), (0, 0, 40, 40))        # bright marker in the top-left corner
    exif = Image.Exif()
    exif[0x0112] = tag
    buf = io.BytesIO()
    base.save(buf, "JPEG", exif=exif.tobytes(), quality=95)
    return buf.getvalue()


@pytest.mark.parametrize("tag, expected", [(1, False), (0, False), (3, True)])
def test_exif_transposed_flag_follows_the_orientation_tag(tag, expected):
    # 1 = upright, 0 = junk (Pillow leaves it alone), 3 = 180 degrees (the size does not change)
    img, facts = load_upload(_jpeg_with_orientation(tag))
    assert facts["exif_transposed"] is expected
    gray = np.asarray(img.convert("L"))
    marker_corner = gray[-20:, -20:] if expected else gray[:20, :20]
    other_corner = gray[:20, :20] if expected else gray[-20:, -20:]
    assert marker_corner.mean() > 200 and other_corner.mean() < 50


@pytest.mark.parametrize("payload, needle", [
    (b"GIF89a" + b"\x00" * 200, "PNG, JPEG or WEBP"),
    (b"\x00" * 128 + b"DICM" + b"\x00" * 200, "DICOM"),
    (b"just some text", "PNG, JPEG or WEBP"),
])
def test_unsupported_bytes_get_a_readable_message(payload, needle):
    with pytest.raises(UploadError, match=needle):
        load_upload(payload)


def test_too_small_and_too_large_are_refused():
    with pytest.raises(UploadError, match="64"):
        load_upload(png_bytes(32, 32))
    with pytest.raises(UploadError, match="20 MB"):
        load_upload(b"\x89PNG\r\n\x1a\n" + b"\x00" * (20 * 1024 * 1024 + 1))


def test_model_transform_is_the_published_one():
    img = _gray_rgb()
    assert torch.equal(model_transform()(img), PUBLISHED(img))
    assert list(CLIP_MEAN) == [0.48145466, 0.4578275, 0.40821073]


def test_retrieval_transform_gives_identical_tensors_for_grayscale_origin_images():   # D4
    from scripts.evaluate_cxr_retrieval import _img_transform
    img = _gray_rgb(300, 340)
    assert torch.equal(model_transform()(img), _img_transform()(img))


def test_model_input_image_is_what_the_tower_sees():
    img = _gray_rgb()
    shown = model_input_image(img)
    assert shown.size == (224, 224)
    rebuilt = T.Normalize(mean=list(CLIP_MEAN), std=list(CLIP_STD))(T.ToTensor()(shown))
    assert torch.equal(rebuilt, model_transform()(img))


def test_thumbnail_is_a_bounded_jpeg():
    data = thumbnail_jpeg(_gray_rgb(1200, 900))
    thumb = Image.open(io.BytesIO(data))
    assert thumb.format == "JPEG" and max(thumb.size) == 512
