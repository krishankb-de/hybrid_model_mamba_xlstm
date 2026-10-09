"""P9-G1 (CHAT_UI_PLAN.md): the end-of-report training target, behind dataset.report_eos_target.

Why it exists. The report model never learned to end a report. train_report_generation.py sets
tokenizer.pad_token = tokenizer.eos_token, ImageTextDataset tokenises with padding="max_length", and
ReportGenerationLightningModule._step masks every attention_mask == 0 position out of the loss, so an
EOS could only ever appear as masked padding and was never a target.

What the flag does. On, a report that fits with room for one more token ends in exactly one EOS whose
attention_mask is 1 (so it is supervised) and the padding after it stays masked; a report cut by
max_length, one that fills it exactly, and an empty one keep today's encoding. Off (the default, declared
in no yaml) every row is byte-identical to before. ImageTextDataset is shared with the closed retrieval
chapter and with every published report-gen run, so the off-parity tests are the point of this file.

CPU-only, offline, synthetic text only (R7). Every behaviour runs against a fake tokenizer that mimics
the slice of the HF call ImageTextDataset makes and, where the GPT-2 tokenizer is cached, against the
real one.
"""

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent

MAX_LEN = 16        # tiny on purpose: exact-fit and over-long reports are a handful of tokens
IMAGE_SIZE = 8
PREFIX_K = 4
SHORT = 9           # above GPT-2's 7-token "Findings: ... Impression:" scaffold, below MAX_LEN - 1
# One GPT-2 token each (" the", " of", ...), and a different id at every neighbouring position, so a
# row shifted by one slot cannot pass for the right one.
FILLER = ["the", "of", "and", "in", "to", "is", "was", "with", "for", "on"]


class _FakeTokenizer:
    """Whitespace tokenizer with the slice of the HF __call__ that ImageTextDataset makes.

    One token per word, ids from 8 up (so they fit the tiny decoder's vocab of 100 and never collide
    with eos/pad = 7). Right padding with pad_token_id, like train_report_generation.py's
    `tokenizer.pad_token = tokenizer.eos_token; tokenizer.padding_side = "right"`.
    """

    eos_token_id = 7
    pad_token_id = 7
    padding_side = "right"
    vocab_size = 100

    def __init__(self):
        self._ids = {}

    def __call__(self, text, max_length=None, truncation=False, padding=False, return_tensors=None):
        assert padding in (False, "max_length"), padding
        assert return_tensors in (None, "pt"), return_tensors
        ids = [self._ids.setdefault(w, 8 + len(self._ids)) for w in text.split()]
        if truncation:
            ids = ids[:max_length]
        mask = [1] * len(ids)
        if padding == "max_length":
            fill = max_length - len(ids)
            ids, mask = ids + [self.pad_token_id] * fill, mask + [0] * fill
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids], dtype=torch.long),
                    "attention_mask": torch.tensor([mask], dtype=torch.long)}
        return {"input_ids": ids, "attention_mask": mask}


@pytest.fixture(params=["fake", "gpt2"])
def tok(request):
    if request.param == "fake":
        return _FakeTokenizer()
    from transformers import AutoTokenizer
    try:
        t = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    except Exception as exc:                      # offline CI without the tokenizer cached
        pytest.skip("gpt2 tokenizer not cached: %s" % exc)
    t.pad_token = t.eos_token                     # scripts/train_report_generation.py
    t.padding_side = "right"
    return t


def _full_text(findings, impression=""):
    """The text ImageTextDataset builds when concatenate_sections is on (the report-gen setting)."""
    return "Findings: {} Impression: {}".format(findings, impression).strip()


def _n_tokens(tok, text):
    return len(tok(text, padding=False)["input_ids"])


def _findings_for(tok, n_tokens):
    """Synthetic findings so that the whole 'Findings: ... Impression:' text is exactly n_tokens long."""
    words = []
    while _n_tokens(tok, _full_text(" ".join(words))) < n_tokens:
        words.append(FILLER[len(words) % len(FILLER)])
    assert _n_tokens(tok, _full_text(" ".join(words))) == n_tokens, "the tokenizer jumped past the target"
    return " ".join(words)


def _todays_encoding(tok, text):
    """What ImageTextDataset.__getitem__ produced before P9-G1: its one padded tokenizer call, verbatim."""
    enc = tok(text, max_length=MAX_LEN, truncation=True, padding="max_length", return_tensors="pt")
    return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)


def _dataset(tok, findings, impression="", **dataset_cfg):
    from scripts.train_contrastive import ImageTextDataset

    rows = [{"findings": f, "impression": impression, "study_id": i, "image": Image.new("RGB", (8, 8))}
            for i, f in enumerate(findings)]
    cfg = OmegaConf.create({"dataset": dict({"max_length": MAX_LEN, "image_size": IMAGE_SIZE}, **dataset_cfg)})
    return ImageTextDataset(rows, tok, cfg)


def _supervised_eos(tok, input_ids, attention_mask):
    """How many EOS tokens the loss sees in each row: EOS ids whose attention_mask is 1."""
    return ((input_ids == tok.eos_token_id) & (attention_mask == 1)).sum(dim=-1)


LENGTHS = [
    pytest.param(SHORT, id="short"),
    pytest.param(MAX_LEN - 1, id="one_slot_left"),
    pytest.param(MAX_LEN, id="exact_fit_no_room"),
    pytest.param(MAX_LEN + 9, id="over_long"),
]


# ── the flag ─────────────────────────────────────────────────────────────────────────────────────────

def test_flag_defaults_off_and_never_touches_the_tokenizer_when_off():
    """Off is the default. load_mimic_cxr builds this dataset with tokenizer=None in other tests, so with
    the flag off __init__ must not look at the tokenizer at all."""
    from scripts.train_contrastive import ImageTextDataset

    for dataset_cfg in ({}, {"report_eos_target": False}, {"report_eos_target": None}):
        cfg = OmegaConf.create({"dataset": dict({"max_length": MAX_LEN}, **dataset_cfg)})
        assert ImageTextDataset([], None, cfg).report_eos_target is False, dataset_cfg
    on = OmegaConf.create({"dataset": {"max_length": MAX_LEN, "report_eos_target": True}})
    assert ImageTextDataset([], _FakeTokenizer(), on).report_eos_target is True


def test_flag_rejects_a_value_that_is_not_a_boolean():
    """A typo'd override ('no', 1) must fail at construction, not silently train the wrong target for hours."""
    from scripts.train_contrastive import ImageTextDataset

    for bad in ("no", "false", 1, 0):
        cfg = OmegaConf.create({"dataset": {"max_length": MAX_LEN, "report_eos_target": bad}})
        with pytest.raises(ValueError, match="report_eos_target"):
            ImageTextDataset([], _FakeTokenizer(), cfg)


def test_flag_on_needs_a_tokenizer_with_an_eos_and_a_pad_id():
    from scripts.train_contrastive import ImageTextDataset

    class _NoPad(_FakeTokenizer):
        pad_token_id = None

    class _NoEos(_FakeTokenizer):
        eos_token_id = None

    cfg = OmegaConf.create({"dataset": {"max_length": MAX_LEN, "report_eos_target": True}})
    for tokenizer in (_NoPad(), _NoEos(), None):
        with pytest.raises(ValueError, match="eos_token_id and pad_token_id"):
            ImageTextDataset([], tokenizer, cfg)


# ── flag off: byte-identical to today ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("dataset_cfg", [{}, {"report_eos_target": False}], ids=["key_absent", "key_false"])
@pytest.mark.parametrize("n_tokens", LENGTHS)
def test_flag_off_row_is_todays_encoding_byte_for_byte(tok, n_tokens, dataset_cfg):
    """key_absent is the cfg every contrastive and every published report-gen run has."""
    findings = _findings_for(tok, n_tokens)
    row = _dataset(tok, [findings], **dataset_cfg)[0]
    ids, mask = _todays_encoding(tok, _full_text(findings))

    assert set(row) == {"input_ids", "attention_mask", "pixel_values"}
    assert row["input_ids"].dtype == ids.dtype and row["attention_mask"].dtype == mask.dtype
    assert torch.equal(row["input_ids"], ids)
    assert torch.equal(row["attention_mask"], mask)
    assert int(row["attention_mask"].sum()) == min(n_tokens, MAX_LEN)
    assert int(_supervised_eos(tok, row["input_ids"], row["attention_mask"])) == 0, "today no EOS is supervised"


def test_flag_off_batch_through_the_default_collate_is_unchanged(tok):
    lengths = [SHORT, MAX_LEN - 1, MAX_LEN, MAX_LEN + 9, 12]
    texts = [_findings_for(tok, n) for n in lengths]
    batch = next(iter(DataLoader(_dataset(tok, texts), batch_size=len(texts), shuffle=False, num_workers=0)))
    today = [_todays_encoding(tok, _full_text(t)) for t in texts]

    assert set(batch) == {"input_ids", "attention_mask", "pixel_values"}
    assert batch["input_ids"].shape == batch["attention_mask"].shape == (len(texts), MAX_LEN)
    assert torch.equal(batch["input_ids"], torch.stack([i for i, _ in today]))
    assert torch.equal(batch["attention_mask"], torch.stack([m for _, m in today]))
    assert _supervised_eos(tok, batch["input_ids"], batch["attention_mask"]).tolist() == [0] * len(texts)


# ── flag on ───────────────────────────────────────────────────────────────────────────────────────────

def test_flag_on_short_report_gets_one_supervised_eos_then_masked_padding(tok):
    findings = _findings_for(tok, SHORT)
    row = _dataset(tok, [findings], report_eos_target=True)[0]
    today_ids, today_mask = _todays_encoding(tok, _full_text(findings))
    ids, mask = row["input_ids"], row["attention_mask"]
    n = SHORT

    assert set(row) == {"input_ids", "attention_mask", "pixel_values"}
    assert ids.shape == mask.shape == (MAX_LEN,)
    assert ids.dtype == today_ids.dtype and mask.dtype == today_mask.dtype
    assert torch.equal(ids[:n], today_ids[:n]), "the report text is untouched"
    assert ids[n].item() == tok.eos_token_id and mask[n].item() == 1, "one EOS right after the text, supervised"
    assert mask[:n + 1].tolist() == [1] * (n + 1)
    assert mask[n + 1:].tolist() == [0] * (MAX_LEN - n - 1), "every later position stays masked"
    assert ids[n + 1:].tolist() == [tok.pad_token_id] * (MAX_LEN - n - 1), "padding is the pad id, as before"
    assert int(_supervised_eos(tok, ids, mask)) == 1


def test_flag_on_report_that_leaves_exactly_one_slot_puts_the_eos_in_the_last_slot(tok):
    findings = _findings_for(tok, MAX_LEN - 1)
    row = _dataset(tok, [findings], report_eos_target=True)[0]
    today_ids, _ = _todays_encoding(tok, _full_text(findings))
    ids, mask = row["input_ids"], row["attention_mask"]

    assert torch.equal(ids[:-1], today_ids[:-1])
    assert ids[-1].item() == tok.eos_token_id and mask[-1].item() == 1
    assert mask.tolist() == [1] * MAX_LEN, "no padding left"
    assert int(_supervised_eos(tok, ids, mask)) == 1


@pytest.mark.parametrize("n_tokens", [MAX_LEN, MAX_LEN + 1, MAX_LEN + 9], ids=["exact_fit", "cut_by_1", "cut_by_9"])
def test_flag_on_report_with_no_room_for_the_eos_equals_flag_off(tok, n_tokens):
    """The budget, not the report, ended it: no EOS, and the row is today's row."""
    findings = _findings_for(tok, n_tokens)
    on = _dataset(tok, [findings], report_eos_target=True)[0]
    off = _dataset(tok, [findings], report_eos_target=False)[0]
    ids, mask = _todays_encoding(tok, _full_text(findings))

    for key in ("input_ids", "attention_mask"):
        assert torch.equal(on[key], off[key]), key
    assert torch.equal(on["input_ids"], ids) and torch.equal(on["attention_mask"], mask)
    assert int(on["attention_mask"].sum()) == MAX_LEN
    assert int(_supervised_eos(tok, on["input_ids"], on["attention_mask"])) == 0


def test_flag_on_empty_report_keeps_todays_encoding(tok):
    """An empty report is not a report: it must not teach the model to emit EOS as its very first token."""
    on = _dataset(tok, [""], concatenate_sections=False, report_eos_target=True)[0]
    ids, mask = _todays_encoding(tok, "")

    assert torch.equal(on["input_ids"], ids) and torch.equal(on["attention_mask"], mask)
    assert int(on["attention_mask"].sum()) == 0
    assert int(_supervised_eos(tok, on["input_ids"], on["attention_mask"])) == 0


def test_flag_on_changes_only_the_padding_slot_of_a_report_that_fits(tok):
    """From the bare scaffold up to MAX_LEN + 3 tokens, the flag-on row differs from today's in at most one
    position: the first pad slot, and only when the report leaves room for it."""
    for n_tokens in range(_n_tokens(tok, _full_text("")), MAX_LEN + 4):
        findings = _findings_for(tok, n_tokens)
        on = _dataset(tok, [findings], report_eos_target=True)[0]
        ids, mask = _todays_encoding(tok, _full_text(findings))
        changed_mask = (on["attention_mask"] != mask).nonzero().flatten().tolist()
        if n_tokens <= MAX_LEN - 1:
            assert changed_mask == [n_tokens], n_tokens
            assert int(_supervised_eos(tok, on["input_ids"], on["attention_mask"])) == 1, n_tokens
        else:
            assert changed_mask == [], n_tokens
            assert torch.equal(on["input_ids"], ids), n_tokens


def test_flag_on_batch_through_the_default_collate(tok):
    lengths = [SHORT, MAX_LEN - 1, MAX_LEN, MAX_LEN + 9, 12]
    texts = [_findings_for(tok, n) for n in lengths]
    batch = next(iter(DataLoader(_dataset(tok, texts, report_eos_target=True), batch_size=len(texts),
                                 shuffle=False, num_workers=0)))

    assert batch["input_ids"].shape == batch["attention_mask"].shape == (len(texts), MAX_LEN)
    assert batch["attention_mask"].sum(dim=1).tolist() == [SHORT + 1, MAX_LEN, MAX_LEN, MAX_LEN, 13]
    assert _supervised_eos(tok, batch["input_ids"], batch["attention_mask"]).tolist() == [1, 1, 0, 0, 1]


# ── end to end through the trainer's own masking ──────────────────────────────────────────────────────

def _labels_seen_by_the_decoder(monkeypatch, module, batch):
    """Run ReportGenerationLightningModule._step and return the labels it hands the decoder."""
    seen = {}
    real_forward = module.decoder.forward

    def spy(*args, **kwargs):
        seen["labels"] = kwargs["labels"].detach().clone()
        return real_forward(*args, **kwargs)

    monkeypatch.setattr(module.decoder, "forward", spy)
    module._step(batch, "val")
    return seen["labels"]


def test_step_supervises_the_eos_and_masks_every_pad(tok, monkeypatch):
    """The point of the whole change, through the real masking at lightning_module.py (_step): with the flag
    on the label at the EOS position is the EOS id, not -100, and every pad label is -100; with it off no
    EOS is ever a label."""
    from hybrid_xmamba.models.configuration_hybrid import HybridConfig
    from hybrid_xmamba.training.lightning_module import ReportGenerationLightningModule

    lengths = [SHORT, MAX_LEN - 1, MAX_LEN, MAX_LEN + 9]
    texts = [_findings_for(tok, n) for n in lengths]
    config = HybridConfig(
        vocab_size=tok.vocab_size, dim=64, num_layers=2, layer_pattern=["mamba", "mlstm"],
        max_position_embeddings=64, use_fast_path=False, use_tfla=False,
    )
    torch.manual_seed(0)
    module = ReportGenerationLightningModule(decoder_config=config, prefix_k=PREFIX_K)
    module.eval()

    for flag in (True, False):
        loader = DataLoader(_dataset(tok, texts, report_eos_target=flag), batch_size=len(texts), shuffle=False)
        batch = next(iter(loader))
        batch = {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"],
                 "patch_grid": torch.randn(len(texts), 16, 768)}
        labels = _labels_seen_by_the_decoder(monkeypatch, module, batch)

        assert labels.shape == (len(texts), PREFIX_K + MAX_LEN)
        assert (labels[:, :PREFIX_K] == -100).all(), "the image prefix is never a target"
        report = labels[:, PREFIX_K:]
        for row, n in enumerate(lengths):
            body = min(n, MAX_LEN)
            assert torch.equal(report[row, :body], batch["input_ids"][row, :body]), "the text is supervised"
            if flag and n <= MAX_LEN - 1:
                assert report[row, n].item() == tok.eos_token_id, "the label at the EOS position is the EOS id"
                assert (report[row, n + 1:] == -100).all(), "every pad label is -100"
            else:
                assert (report[row, body:] == -100).all(), "every pad label is -100"
        expected = [1, 1, 0, 0] if flag else [0, 0, 0, 0]
        assert (labels == tok.eos_token_id).sum(dim=1).tolist() == expected, "supervised EOS labels per row"


# ── the override G3 uses ──────────────────────────────────────────────────────────────────────────────

def test_plus_override_turns_the_flag_on_from_the_real_configs(tok):
    """The key is declared in no yaml, so Hydra's struct mode needs the leading '+' to add it. This composes
    the report-gen recipe's own model/dataset/trainer configs both ways and builds the dataset from each."""
    pytest.importorskip("hydra")
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from hydra.errors import ConfigCompositionException

    base = ["model=hybrid_150m_m3_rrg", "dataset=cxr_mimic_full", "trainer=h100_single_gpu"]
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base="1.3"):
        default_cfg = compose(config_name="config", overrides=base)
        on_cfg = compose(config_name="config", overrides=base + ["+dataset.report_eos_target=true"])
        with pytest.raises(ConfigCompositionException, match=r"\+dataset\.report_eos_target"):
            compose(config_name="config", overrides=base + ["dataset.report_eos_target=true"])

    from scripts.train_contrastive import ImageTextDataset

    findings = _findings_for(tok, SHORT)
    rows = [{"findings": findings, "impression": "", "study_id": 1, "image": Image.new("RGB", (8, 8))}]
    off = ImageTextDataset(rows, tok, default_cfg)
    on = ImageTextDataset(rows, tok, on_cfg)

    assert off.report_eos_target is False and on.report_eos_target is True
    assert default_cfg.dataset.max_length == 256 and on_cfg.dataset.max_length == 256
    assert int(off[0]["attention_mask"].sum()) == SHORT
    assert int(on[0]["attention_mask"].sum()) == SHORT + 1
    assert on[0]["input_ids"][SHORT].item() == tok.eos_token_id
