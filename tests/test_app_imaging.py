"""CHAT_UI_PLAN.md P2-C: uploads become exactly what the published transform expects."""
import io
import struct
import warnings
import zlib

import numpy as np
import pytest
import torch
import torchvision.transforms as T
from PIL import Image, PngImagePlugin, features

from app.imaging import (CLIP_MEAN, CLIP_STD, MIN_SIDE, UploadError, load_upload, model_input_image,
                         model_transform, thumbnail_jpeg)
from tests.app_helpers import jpeg_bytes, png_bytes

needs_webp = pytest.mark.skipif(not features.check("webp"), reason="Pillow built without WebP")

PUBLISHED = T.Compose([   # verbatim from scripts/evaluate_report_generation.py run_checkpoint_inspection
    T.Resize((224, 224)), T.Grayscale(num_output_channels=3), T.ToTensor(),
    T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
])


def _gray_rgb(w=320, h=320):
    return Image.open(io.BytesIO(png_bytes(w, h))).convert("RGB")


def _encode(img, fmt, **kwargs):
    buf = io.BytesIO()
    img.save(buf, fmt, **kwargs)
    return buf.getvalue()


def test_16bit_png_is_rescaled_not_saturated():   # Review Focus 1
    ramp = np.linspace(0, 4095, 256 * 256).reshape(256, 256).astype(np.uint16)   # 12-bit data
    img, facts = load_upload(_encode(Image.fromarray(ramp), "PNG"))
    arr = np.asarray(img.convert("L")).astype(np.float64)
    expected = np.rint(ramp / 4095.0 * 255.0)
    # PIL's own convert clips 12-bit data to white: mean ~247, ~94% of the pixels at 255 (and min/max still look fine)
    assert abs(arr.mean() - 127.5) < 3, arr.mean()
    assert (arr == 255).mean() < 0.02, (arr == 255).mean()
    assert np.abs(arr - expected).max() <= 1
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


# Malformed metadata (review I1): the contract is that only UploadError leaves load_upload.
SMALL = Image.new("RGB", (80, 80), (50, 60, 70))


def _png_with_exif_text_profile(payload):
    info = PngImagePlugin.PngInfo()
    info.add_text("Raw profile type exif", payload)
    return _encode(SMALL, "PNG", pnginfo=info)


def _jpeg_with_retyped_exif_entry(orientation, entry_type=11, tag=0x010F):
    """A JPEG whose EXIF carries `orientation` and one IFD0 entry (`tag`) with a wrong declared type."""
    exif = Image.Exif()
    exif[0x0112] = orientation
    exif[tag] = "Make"
    exif[0x0131] = "software-" + "x" * 40              # trailing data, so the re-typed entry still reads
    raw = bytearray(exif.tobytes())
    start = 6                                          # len(b"Exif\x00\x00")
    endian = "<" if raw[start:start + 2] == b"II" else ">"
    for i in range(struct.unpack(endian + "H", raw[start + 8:start + 10])[0]):
        entry = start + 10 + 12 * i
        if struct.unpack(endian + "H", raw[entry:entry + 2])[0] == tag:
            raw[entry + 2:entry + 4] = struct.pack(endian + "H", entry_type)
    return _encode(SMALL, "JPEG", exif=bytes(raw))


@pytest.mark.parametrize("make", [
    pytest.param(lambda: _encode(SMALL, "PNG", exif=b"Exif\x00\x00II"), id="png-truncated-tiff-header"),
    pytest.param(lambda: _encode(SMALL, "WEBP", exif=b"Exif\x00\x00II"), id="webp-truncated-tiff-header",
                 marks=needs_webp),
    pytest.param(lambda: _png_with_exif_text_profile("\nexif\n      10\nzzzz"), id="png-text-profile-not-hex"),
    pytest.param(lambda: _jpeg_with_retyped_exif_entry(1), id="jpeg-corrupt-entry-but-upright"),
])
def test_unreadable_exif_is_treated_as_no_orientation(make):
    img, facts = load_upload(make())
    assert img.size == (80, 80) and facts["exif_transposed"] is False


def test_corrupt_exif_entry_on_a_rotated_jpeg_is_a_readable_refusal():
    # exif_transpose re-serialises every tag once the orientation is gone, and trips on the bad entry
    with pytest.raises(UploadError, match="orientation data. Re-export it as PNG or JPEG") as err:
        load_upload(_jpeg_with_retyped_exif_entry(6))
    assert "Error" not in str(err.value)
    assert err.value.__cause__ is None and err.value.__suppress_context__      # raised `from None`


def test_only_upload_errors_escape_a_corrupt_exif_entry():
    leaks = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                # Pillow warns about corrupt EXIF
        for orientation in (1, 3, 6):
            for entry_type in range(14):
                try:
                    load_upload(_jpeg_with_retyped_exif_entry(orientation, entry_type))
                except UploadError:
                    pass
                except Exception as exc:
                    leaks.append((orientation, entry_type, type(exc).__name__))
    assert not leaks, leaks


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


def _png_with_header_size(w, h):
    """A valid 64x64 PNG whose IHDR claims w x h: Image.open accepts it, and decoding it would be a mistake."""
    data = bytearray(png_bytes(64, 64))
    data[16:24] = struct.pack(">II", w, h)
    data[29:33] = struct.pack(">I", zlib.crc32(bytes(data[12:29])))      # the IHDR CRC covers its type and 13 data bytes
    return bytes(data)


@pytest.mark.parametrize("w, h", [(8000, 7000), (10000, 10000), (20000, 20000)],
                         ids=["56MP-our-cap", "100MP-pillow-warns", "400MP-pillow-refuses"])
def test_more_than_50_megapixels_is_refused_from_the_header(w, h):
    with warnings.catch_warnings():
        warnings.simplefilter("error")                 # Pillow's decompression-bomb warning must not escape either
        with pytest.raises(UploadError, match="Image is larger than 50 megapixels."):
            load_upload(_png_with_header_size(w, h))


def test_a_thin_image_is_refused_before_any_pixel_is_decoded():   # review M1
    # 1 x 49M px is under the pixel cap, so only the side check stops it; its pixel data could not decode anyway
    with pytest.raises(UploadError, match="Each side must be at least {} pixels.".format(MIN_SIDE)):
        load_upload(_png_with_header_size(1, 49_000_000))


@pytest.mark.parametrize("w, h, accepted", [(MIN_SIDE, MIN_SIDE, True), (MIN_SIDE - 1, 200, False),
                                            (200, MIN_SIDE - 1, False)])
def test_each_side_must_be_at_least_the_minimum(w, h, accepted):
    if accepted:
        assert load_upload(png_bytes(w, h))[0].size == (w, h)
    else:
        with pytest.raises(UploadError, match="Each side must be at least {} pixels.".format(MIN_SIDE)):
            load_upload(png_bytes(w, h))


@pytest.mark.parametrize("make", [
    pytest.param(lambda: png_bytes(200, 200)[:20000], id="png-truncated"),
    pytest.param(lambda: jpeg_bytes(200, 200)[:2000], id="jpeg-truncated"),
    pytest.param(lambda: b"\x89PNG\r\n\x1a\n" + b"junk" * 50, id="png-magic-then-junk"),
    pytest.param(lambda: b"\xff\xd8\xff" + b"junk" * 50, id="jpeg-magic-then-junk"),
])
def test_damaged_files_get_a_plain_message_without_python_class_names(make):   # review M3
    with pytest.raises(UploadError) as err:
        load_upload(make())
    assert str(err.value).startswith("Could not read the image.") and "Error" not in str(err.value)
    assert err.value.__cause__ is None and err.value.__suppress_context__      # raised `from None`


def _animated(fmt):
    frames = [Image.new("RGB", (100, 100), c) for c in ((255, 0, 0), (0, 255, 0), (0, 0, 255))]
    kwargs = {"lossless": True} if fmt == "WEBP" else {}
    return _encode(frames[0], fmt, save_all=True, append_images=frames[1:], duration=50, **kwargs)


@pytest.mark.parametrize("fmt", ["PNG", pytest.param("WEBP", marks=needs_webp)])
def test_an_animated_upload_contributes_its_first_frame(fmt):   # APNG / animated WEBP
    img, facts = load_upload(_animated(fmt))
    assert facts["format"] == fmt and img.getpixel((50, 50)) == (255, 0, 0)


@pytest.mark.parametrize("value", [0, 777, 65535])
def test_a_flat_16bit_image_does_not_divide_by_zero(value):
    flat = np.full((100, 100), value, dtype=np.uint16)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)   # numpy reports 0/0 as a warning, not an exception
        img, _ = load_upload(_encode(Image.fromarray(flat), "PNG"))
    assert np.asarray(img.convert("L")).max() == 0


@pytest.mark.parametrize("make, mode", [
    pytest.param(lambda: png_bytes(120, 120, "P"), "P", id="palette-png"),
    pytest.param(lambda: png_bytes(120, 120, "LA"), "LA", id="gray-alpha-png"),
    pytest.param(lambda: _encode(Image.open(io.BytesIO(png_bytes(120, 120))).convert("CMYK"), "JPEG"), "CMYK",
                 id="cmyk-jpeg"),
])
def test_other_color_modes_reach_8bit_rgb(make, mode):
    img, facts = load_upload(make())
    assert img.mode == "RGB" and facts["mode"] == mode


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
