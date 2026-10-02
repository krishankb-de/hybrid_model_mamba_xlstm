import pytest
from pydantic import ValidationError

from app.ids import new_id
from app.schemas import Options, error_body


def test_defaults_are_the_published_protocol():
    o = Options()
    assert (o.decode, o.beam_size, o.max_new_tokens, o.cached_decode) == ("beam", 3, 100, True)
    assert (o.k_images, o.k_reports, o.label, o.display_repair, o.compile) == (4, 3, True, False, False)


@pytest.mark.parametrize("bad", [
    {"beam_size": 0}, {"beam_size": 9}, {"max_new_tokens": 15}, {"max_new_tokens": 201},
    {"k_images": 13}, {"k_reports": 11}, {"retrieval_k": 3}, {"decode": "sample"}, {"test_row": -1},
])
def test_out_of_bounds_or_unknown_options_are_rejected(bad):
    with pytest.raises(ValidationError):
        Options(**bad)


def test_ids_sort_by_creation_and_do_not_collide():
    ids = [new_id("m") for _ in range(500)]
    assert len(set(ids)) == 500 and ids == sorted(ids) and ids[0].startswith("m_")


def test_error_envelope_shape():
    assert error_body("overloaded_error", "busy") == {"type": "error", "error": {"type": "overloaded_error", "message": "busy"}}
