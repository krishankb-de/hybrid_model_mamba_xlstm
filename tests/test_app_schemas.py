import threading
import time

import pytest
from pydantic import ValidationError

from app import ids as ids_module
from app.ids import new_id
from app.schemas import Options, error_body


def test_defaults_are_the_published_protocol():
    o = Options()
    assert (o.decode, o.beam_size, o.max_new_tokens, o.cached_decode) == ("beam", 3, 100, True)
    assert (o.k_images, o.k_reports, o.label, o.display_repair, o.compile) == (4, 3, True, False, False)
    assert o.stop_on_repeat is False   # P4-G: the API keeps the published protocol, which always decodes the whole budget
    assert (o.model, o.reference, o.test_row) == (None, None, None)


def test_stop_on_repeat_is_a_plain_flag_that_follows_display_repair_in_the_field_order():
    assert Options(stop_on_repeat=True).stop_on_repeat is True
    assert list(Options.model_fields)[list(Options.model_fields).index("display_repair") + 1] == "stop_on_repeat"


@pytest.mark.parametrize("bad", [
    {"beam_size": 0}, {"beam_size": 9}, {"max_new_tokens": 15}, {"max_new_tokens": 201},
    {"k_images": 13}, {"k_reports": 11}, {"retrieval_k": 3}, {"decode": "sample"}, {"test_row": -1},
])
def test_out_of_bounds_or_unknown_options_are_rejected(bad):
    with pytest.raises(ValidationError):
        Options(**bad)


@pytest.mark.parametrize("good", [
    {"beam_size": 1}, {"beam_size": 8}, {"max_new_tokens": 16}, {"max_new_tokens": 200},
    {"k_images": 0}, {"k_images": 12}, {"k_reports": 0}, {"k_reports": 10}, {"test_row": 0},
])
def test_edge_case_options_are_accepted(good):
    o = Options(**good)
    assert o is not None


def test_ids_sort_by_creation_and_do_not_collide():
    ids = [new_id("m") for _ in range(500)]
    assert len(set(ids)) == 500 and ids == sorted(ids) and ids[0].startswith("m_")


def test_error_envelope_shape():
    assert error_body("overloaded_error", "busy") == {"type": "error", "error": {"type": "overloaded_error", "message": "busy"}}


def test_ids_monotonic_on_clock_stepback(monkeypatch):
    """Ids strictly increasing even when clock steps backward."""
    ids_module._id_state.clear()
    times = [100, 101, 102, 94]  # T0, T0+1, T0+2, T0-8
    time_iter = iter(times)

    def mock_time():
        return next(time_iter) / 1000.0

    monkeypatch.setattr(ids_module.time, "time", mock_time)
    ids = [new_id("stepback") for _ in range(4)]
    assert ids == sorted(ids), f"Not sorted: {ids}"
    assert len(set(ids)) == 4, f"Not unique: {ids}"


def test_ids_frozen_clock_70k(monkeypatch):
    """70,000 ids on frozen clock are unique, same length, and sorted."""
    ids_module._id_state.clear()
    frozen_time = 123456789.123  # Fixed ms = 123456789123

    def mock_time():
        return frozen_time

    monkeypatch.setattr(ids_module.time, "time", mock_time)
    ids = [new_id("frozen") for _ in range(70000)]
    assert len(set(ids)) == 70000, "Not all unique"
    assert ids == sorted(ids), "Not sorted"
    assert all(len(id_) == len(ids[0]) for id_ in ids), "Length mismatch"


def test_ids_threadsafe_4x5k():
    """4 threads × 5,000 ids each: all unique, each thread's ids strictly increasing."""
    ids_module._id_state.clear()

    thread_ids = [[] for _ in range(4)]

    def worker(thread_idx):
        for _ in range(5000):
            thread_ids[thread_idx].append(new_id("threads"))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    all_ids = [id_ for thread_list in thread_ids for id_ in thread_list]
    assert len(set(all_ids)) == 20000, "Not all unique across threads"
    for thread_list in thread_ids:
        assert thread_list == sorted(thread_list), f"Thread's ids not sorted: {thread_list}"
