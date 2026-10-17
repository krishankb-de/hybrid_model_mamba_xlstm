#!/usr/bin/env python
"""ISBI B10: run PUBLISHED report generators on our MIMIC-CXR test split.

Reviewer: "no comparison with any published report-generation model (R2Gen,
CvT2DistilGPT2, R2Gen-Mamba, CXRMate), not even on a findings-only protocol".

Each released checkpoint runs with ITS OWN preprocessing, tokenizer and the
decoding settings of its released test script / example, on the SAME 2,663
studies and the SAME image per study (test.parquet row order, column `image`)
that our decoders read. Output: results/published_<name>_test_split/
{hyps.txt, refs.txt}, refs identical to our own refs ("Findings: .. Impression: ..").

Modes
-----
  vocab        Rebuild the R2Gen-family word vocabulary. R2Gen, R2GenCMN and
               R2Gen-Mamba build it at run time from the training reports of
               Chen et al.'s MIMIC annotation file (count >= 10, sorted, ids
               from 1); the checkpoints do not store it and the file itself is
               access-restricted. We rebuild the file's train entries from
               MIMIC (official split, one entry per image of a study with a
               FINDINGS section, report = findings section) and check the
               vocabulary size against each checkpoint's embedding rows.
  generate     Decode the test split with one published model.
  protocol_b   Findings-only protocol: studies whose reference has a non-empty
               findings section; refs = findings; hyps = generated findings.
               Both sides are normalised with R2Gen's clean_report_mimic_cxr
               (lower case, punctuation stripped), the field's convention.
  textmetrics  ROUGE-L / BLEU-1 / BLEU-4 with the project's own functions.

DUA (rule R6): nothing here prints report text; only counts and metrics.
"""
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXT = PROJECT_ROOT / "external"
DATA = Path(os.environ.get("MIMIC_DIR", "/sc/home/{}/dataset/mimic_full".format(os.environ.get("USER", ""))))

R2GEN_FAMILY = {
    # name: (repo dir, test-script arguments of the released repo, model module, class)
    "r2gen": ("R2Gen", ["--dataset_name", "mimic_cxr", "--max_seq_length", "100", "--threshold", "10",
                        "--batch_size", "16", "--seed", "42", "--beam_size", "2"],
              "models.r2gen", "R2GenModel"),
    "r2gencmn": ("R2GenCMN", ["--dataset_name", "mimic_cxr", "--max_seq_length", "100", "--threshold", "10",
                              "--batch_size", "16", "--num_layers", "3", "--topk", "32", "--cmm_size", "2048",
                              "--cmm_dim", "512", "--seed", "9153", "--beam_size", "3"],
                 "models.models", "BaseCMNModel"),
    "r2genmamba": ("R2Gen-Mamba", ["--dataset_name", "mimic_cxr", "--max_seq_length", "100", "--threshold", "10",
                                   "--batch_size", "16", "--seed", "42", "--beam_size", "2"],
                   "models.r2gen", "R2GenModel"),
}


def sanitize(s: str) -> str:
    return " ".join(str(s).split())


def our_reference(findings: str, impression: str) -> str:
    """Exactly the reference evaluate_report_generation.py builds (then sanitised on dump)."""
    return sanitize(f"Findings: {findings} Impression: {impression}".strip())


def load_chen_cleaner():
    """R2Gen's own clean_report_mimic_cxr, imported from the released repo (no re-implementation)."""
    path = EXT / "R2Gen" / "code" / "modules" / "tokenizers.py"
    spec = importlib.util.spec_from_file_location("r2gen_tokenizers_src", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    tok = mod.Tokenizer.__new__(mod.Tokenizer)  # no annotation file needed for the cleaner
    return tok.clean_report_mimic_cxr


def write_dir(out: Path, hyps: List[str], refs: List[str]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "hyps.txt").write_text("\n".join(sanitize(h) for h in hyps) + "\n")
    (out / "refs.txt").write_text("\n".join(sanitize(r) for r in refs) + "\n")
    print("wrote %d pairs to %s" % (len(hyps), out))


# --------------------------------------------------------------------------- vocab
def chen_train_reports(threshold_split: str = "train"):
    import pandas as pd
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from mimic_cxr_vendor.extract import extract_findings_impression

    split = pd.read_csv(DATA / "mimic-cxr-2.0.0-split.csv.gz")
    per_study = split.groupby(["subject_id", "study_id", "split"]).size().reset_index(name="n_img")
    counts, reports = {}, {}
    missing = 0
    for row in per_study.itertuples(index=False):
        s = "p{}".format(row.subject_id)
        p = DATA / "reports" / "files" / s[:3] / s / "s{}.txt".format(row.study_id)
        if not p.exists():
            missing += 1
            continue
        f, _ = extract_findings_impression(p.read_text(errors="ignore"), "s{}".format(row.study_id))
        if not f.strip():
            continue
        counts.setdefault(row.split, [0, 0])
        counts[row.split][0] += 1
        counts[row.split][1] += int(row.n_img)
        if row.split == threshold_split:
            reports[(row.subject_id, row.study_id)] = (f, int(row.n_img))
    return counts, reports, missing


def build_vocab(reports, clean, threshold: int, per_image: bool) -> List[str]:
    from collections import Counter
    c = Counter()
    for f, n in reports.values():
        toks = clean(f).split()
        for t in toks:
            c[t] += n if per_image else 1
    vocab = [k for k, v in c.items() if v >= threshold] + ["<unk>"]
    vocab.sort()
    return vocab


def ckpt_vocab_rows(path: Path) -> Dict[str, list]:
    import torch
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck)
    out = {k: list(v.shape) for k, v in sd.items()
           if any(s in k for s in ("tgt_embed", "logit.weight", "logit.bias", "lut"))}
    return out


def mode_vocab(args) -> None:
    clean = load_chen_cleaner()
    counts, reports, missing = chen_train_reports()
    print("missing report files:", missing)
    for sp, (n_st, n_img) in sorted(counts.items()):
        print("split %-9s studies with findings %7d  images %7d" % (sp, n_st, n_img))
    print("(Chen et al. annotation, per CvT2DistilGPT2 Table 2: train 270,790 / val 2,130 / test 3,858 images)")
    vocab_dir = EXT / "vocab"
    vocab_dir.mkdir(parents=True, exist_ok=True)
    for per_image in (True, False):
        for thr in (10, 3):
            v = build_vocab(reports, clean, thr, per_image)
            name = "chen_t%d_%s.json" % (thr, "img" if per_image else "study")
            (vocab_dir / name).write_text(json.dumps(v))
            print("vocab %-22s size %d" % (name, len(v)))
    for name, (repo, _, _, _) in R2GEN_FAMILY.items():
        p = EXT / repo / "weights" / "mimic_model.pth"
        try:
            rows = ckpt_vocab_rows(p)
            print("checkpoint %-11s %s" % (name, json.dumps(rows)))
        except Exception as exc:
            print("checkpoint %-11s FAILED to load: %s" % (name, exc))


# --------------------------------------------------------------------------- generate
def test_rows():
    import pandas as pd
    df = pd.read_parquet(args_global.parquet)
    return df


def check_ref_alignment(refs: List[str]) -> None:
    anchor = PROJECT_ROOT / "results" / "report_gen_m3_test_split_s42" / "refs.txt"
    ours = anchor.read_text().splitlines()
    mism = sum(1 for a, b in zip(ours, refs) if a != sanitize(b))
    print("row alignment vs %s: %d rows ours, %d rows here, %d mismatching refs"
          % (anchor.parent.name, len(ours), len(refs), mism))
    if len(ours) != len(refs) or mism:
        raise SystemExit("ERROR: reference rows do not match our published refs; refusing to write")


def r2gen_family(name: str, df, device, limit: Optional[int]):
    import numpy as np
    import torch
    from PIL import Image
    from torchvision import transforms

    repo, test_args, mod_name, cls_name = R2GEN_FAMILY[name]
    code = EXT / repo / "code"
    sys.path.insert(0, str(code))
    if name == "r2genmamba":
        # The vendored mamba_ssm imports the compiled `selective_scan_cuda` extension at module level.
        # It is only used through selective_scan_fn; we route that to mamba_ssm's own pure-PyTorch
        # reference implementation (selective_scan_ref, same file), so nothing is compiled.
        import types
        # Appended, not prepended: code/modules also holds a `tokenizers.py` that would shadow the HF
        # `tokenizers` package; only the vendored `mamba_ssm` (not installed) should resolve from here.
        sys.path.append(str(code / "modules"))
        sys.modules["selective_scan_cuda"] = types.ModuleType("selective_scan_cuda")
    import importlib
    if name == "r2gencmn":
        try:
            import spacy  # noqa: F401  (R2GenCMN's tester.py imports it; the tester is never used here)
        except ImportError:
            import types
            sys.modules["spacy"] = types.ModuleType("spacy")
    main_test = importlib.import_module("main_test")
    sys.argv = ["main_test.py", "--image_dir", "/nonexistent", "--ann_path", "/nonexistent",
                "--visual_extractor_pretrained", ""] + test_args  # bool("") is False: no ImageNet download
    a = main_test.parse_agrs()
    print("decoding args:", {k: getattr(a, k) for k in ("beam_size", "max_seq_length", "sample_method",
                                                          "block_trigrams", "temperature", "group_size",
                                                          "decoding_constraint", "threshold")})
    vocab = json.loads((EXT / "vocab" / "chen_t10_img.json").read_text())

    tok_mod = importlib.import_module("modules.tokenizers")
    tok = tok_mod.Tokenizer.__new__(tok_mod.Tokenizer)
    tok.ann_path, tok.threshold, tok.dataset_name = None, a.threshold, "mimic_cxr"
    tok.clean_report = tok.clean_report_mimic_cxr
    tok.token2idx = {t: i + 1 for i, t in enumerate(vocab)}
    tok.idx2token = {i + 1: t for i, t in enumerate(vocab)}

    model_mod = importlib.import_module(mod_name)
    model = getattr(model_mod, cls_name)(a, tok)
    if name == "r2genmamba":
        ms = sys.modules.get("modules.mamba_ssm.modules.mamba_simple")
        ops = importlib.import_module("mamba_ssm.ops.selective_scan_interface")
        patched = 0
        for m in (ms, sys.modules.get("mamba_ssm.modules.mamba_simple")):
            if m is not None and hasattr(m, "selective_scan_fn"):
                m.selective_scan_fn = ops.selective_scan_ref
                patched += 1
        print("selective_scan_fn -> selective_scan_ref in %d module(s)" % patched)
        if not patched:
            raise SystemExit("ERROR: could not patch the Mamba scan")
    ck = torch.load(str(EXT / repo / "weights" / "mimic_model.pth"), map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ck["state_dict"], strict=True), None
    print("checkpoint loaded strictly; epoch", ck.get("epoch"), "monitor_best", ck.get("monitor_best"))
    model.to(device).eval()

    # The released test transform (modules/dataloaders.py, split != train).
    tf = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    hyps = []
    n = len(df) if limit is None else min(limit, len(df))
    bs = a.batch_size
    with torch.no_grad():
        for i in range(0, n, bs):
            imgs = torch.stack([tf(Image.open(p).convert("RGB")) for p in df["image"].iloc[i:min(i + bs, n)]])
            out = model(imgs.to(device), mode="sample")
            if isinstance(out, tuple):
                out = out[0]
            hyps.extend(model.tokenizer.decode_batch(out.cpu().numpy()))
            if (i // bs) % 20 == 0:
                print("  decoded %d/%d" % (min(i + bs, n), n), flush=True)
    return hyps, {"image_transform": "Resize((224,224)), ToTensor, ImageNet mean/std (released test transform)",
                  "beam_size": a.beam_size, "max_seq_length": a.max_seq_length,
                  "block_trigrams": a.block_trigrams, "vocab": "chen_t10_img (rebuilt, %d tokens)" % len(vocab),
                  "target": "findings only (Chen et al. formatting)"}


def cvt2distilgpt2(df, device, limit: Optional[int]):
    import torch
    import transformers
    from PIL import Image
    from torchvision import transforms

    code = EXT / "cvt2distilgpt2" / "code"
    os.chdir(str(code))  # CvT reads its yaml through a relative path
    sys.path.insert(0, str(code))
    from tools.cvt import CvT
    from tools.encoder_projection import EncoderPermuteProject
    from transformers.configuration_utils import PretrainedConfig

    class Net(torch.nn.Module):
        """Same modules, names and construction as CvT2DistilGPT2MIMICXRChen.__init__ (no metrics, no data)."""

        def __init__(self):
            super().__init__()
            self.encoder = CvT(warm_start=False, model_config="cvt-21-384x384", ckpt_name="CvT-21-384x384-IN-22k",
                               ckpt_dir=None, is_encoder=True)
            self.encoder_projection = EncoderPermuteProject(permute_encoder_last_hidden_state=[0, 2, 1],
                                                            encoder_last_hidden_state_size=384,
                                                            decoder_hidden_state_size=768)
            config = transformers.GPT2Config.from_pretrained("distilgpt2")
            config.add_cross_attention = True
            config.is_decoder = True
            decoder = transformers.GPT2LMHeadModel(config=config)
            decoder.resize_token_embeddings(config.vocab_size + 2)

            class DummyEncoder:
                main_input_name = "dummy"

                class DummyConfig(PretrainedConfig):
                    model_type = "bert"

                config = DummyConfig()

                def __init__(self, hidden_size):
                    self.config.hidden_size = hidden_size

                def get_output_embeddings(cls):
                    return None

                def forward(self):
                    return None

            dummy = DummyEncoder(hidden_size=decoder.config.hidden_size)

            class Decoder(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.encoder_decoder = transformers.EncoderDecoderModel(encoder=dummy, decoder=decoder)

            self.decoder = Decoder()

    net = Net()
    ck = torch.load(str(EXT / "cvt2distilgpt2" / "weights" / "mimic_epoch8.ckpt"), map_location="cpu",
                    weights_only=False)
    sd = {k: v for k, v in ck["state_dict"].items()
          if k.split(".")[0] in ("encoder", "encoder_projection", "decoder")}
    res = net.load_state_dict(sd, strict=False)
    print("cvt2distilgpt2: kept %d/%d tensors; missing %d %s; unexpected %d %s"
          % (len(sd), len(ck["state_dict"]), len(res.missing_keys), res.missing_keys[:5],
             len(res.unexpected_keys), res.unexpected_keys[:5]))
    # GPT-2's causal-mask buffers (attn.bias / masked_bias) were persistent in the transformers version the
    # checkpoint was saved with and are non-persistent now; they are constants, not weights.
    mask_buf = ("attn.bias", "attn.masked_bias", "crossattention.bias", "crossattention.masked_bias")
    real_missing = [k for k in res.missing_keys if not k.endswith(mask_buf)]
    real_unexpected = [k for k in res.unexpected_keys if not k.endswith(mask_buf)]
    if real_missing or real_unexpected:
        raise SystemExit("ERROR: checkpoint does not match the CvT2DistilGPT2 module")
    net.to(device).eval()

    tokenizer = transformers.GPT2TokenizerFast.from_pretrained("distilgpt2")
    tokenizer.add_special_tokens({"bos_token": "[BOS]", "pad_token": "[PAD]"})
    tf = transforms.Compose([  # the released test_transforms
        transforms.Resize(size=384 + 64), transforms.CenterCrop(size=[384, 384]), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    hyps = []
    n = len(df) if limit is None else min(limit, len(df))
    bs = 8
    with torch.no_grad():
        for i in range(0, n, bs):
            imgs = torch.stack([tf(Image.open(p).convert("RGB")) for p in df["image"].iloc[i:min(i + bs, n)]])
            imgs = imgs.to(device)
            feats = net.encoder(imgs)["last_hidden_state"]
            feats = net.encoder_projection(feats)["projected_encoder_last_hidden_state"]
            enc = transformers.modeling_outputs.BaseModelOutput(last_hidden_state=feats)
            out = net.decoder.encoder_decoder.generate(
                max_length=128, bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id, num_beams=4, return_dict_in_generate=True,
                use_cache=True, encoder_outputs=enc)
            hyps.extend(tokenizer.batch_decode(out["sequences"], skip_special_tokens=True))
            if (i // bs) % 40 == 0:
                print("  decoded %d/%d" % (min(i + bs, n), n), flush=True)
    return hyps, {"image_transform": "Resize(448) shortest side, CenterCrop(384), ImageNet mean/std (released)",
                  "num_beams": 4, "max_length": 128, "precision": "fp32 (released test config: 16-mixed)",
                  "target": "findings only (Chen et al. formatting)"}


def cxrmate_single(df, device, limit: Optional[int]):
    import torch
    import transformers
    from PIL import Image
    from torchvision import transforms

    rid = "aehrc/cxrmate-single-tf"
    model = transformers.AutoModel.from_pretrained(rid, trust_remote_code=True).to(device).eval()
    tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(rid)
    ip = transformers.AutoFeatureExtractor.from_pretrained(rid)
    e = ip.size["shortest_edge"]
    tf = transforms.Compose([  # the model card's example notebook
        transforms.Resize(size=e), transforms.CenterCrop(size=[e, e]), transforms.ToTensor(),
        transforms.Normalize(mean=ip.image_mean, std=ip.image_std)])
    hyps, finds = [], []
    n = len(df) if limit is None else min(limit, len(df))
    bs = 8
    with torch.no_grad():
        for i in range(0, n, bs):
            imgs = torch.stack([tf(Image.open(p).convert("RGB")) for p in df["image"].iloc[i:min(i + bs, n)]])
            out = model.generate(pixel_values=imgs.to(device), special_token_ids=[tokenizer.sep_token_id],
                                 bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
                                 pad_token_id=tokenizer.pad_token_id, return_dict_in_generate=True,
                                 use_cache=True, max_length=256, num_beams=4)
            f, imp = model.split_and_decode_sections(out.sequences, [tokenizer.sep_token_id, tokenizer.eos_token_id],
                                                     tokenizer)
            for a, b in zip(f, imp):
                hyps.append(sanitize("Findings: %s Impression: %s" % (a, b)))
                finds.append(sanitize(a))
            if (i // bs) % 40 == 0:
                print("  decoded %d/%d" % (min(i + bs, n), n), flush=True)
    return (hyps, finds), {"image_transform": "Resize(384) shortest side, CenterCrop(384), ImageNet mean/std",
                           "num_beams": 4, "max_length": 256, "target": "findings [SEP] impression",
                           "hyp_format": "'Findings: {findings} Impression: {impression}' (our format)"}


def mode_generate(args) -> None:
    import torch
    args.out = str(Path(args.out).resolve())  # the CvT adapter chdirs into its repo
    df = test_rows()
    refs_all = [our_reference(f, i) for f, i in zip(df["findings"].fillna(""), df["impression"].fillna(""))]
    if args.limit is None:
        check_ref_alignment(refs_all)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", device, "rows", len(df))
    finds = None
    if args.model in R2GEN_FAMILY:
        hyps, meta = r2gen_family(args.model, df, device, args.limit)
    elif args.model == "cvt2distilgpt2":
        hyps, meta = cvt2distilgpt2(df, device, args.limit)
    elif args.model == "cxrmate_single_tf":
        (hyps, finds), meta = cxrmate_single(df, device, args.limit)
    else:
        raise SystemExit("unknown model " + args.model)
    out = Path(args.out)
    write_dir(out, hyps, refs_all[:len(hyps)])
    if finds is not None:
        (out / "hyps_findings.txt").write_text("\n".join(finds) + "\n")
    meta.update({"model": args.model, "n": len(hyps), "image": "test.parquet column `image` (our 320x320 frontal)",
                 "empty_hyps": sum(1 for h in hyps if not h.strip()),
                 "mean_hyp_words": sum(len(h.split()) for h in hyps) / max(len(hyps), 1)})
    (out / "generation_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


# --------------------------------------------------------------------------- protocol (b)
def findings_part(h: str) -> str:
    """Text between 'Findings:' and 'Impression:' of one of OUR hypotheses."""
    if "Findings:" not in h:
        return ""
    rest = h.split("Findings:", 1)[1]
    return rest.split("Impression:", 1)[0].strip()


def mode_protocol_b(args) -> None:
    clean = load_chen_cleaner()
    df = test_rows()
    keep = [i for i, f in enumerate(df["findings"].fillna("")) if str(f).strip()]
    refs = [clean(sanitize(df["findings"].iloc[i])) for i in keep]
    print("protocol (b): %d of %d studies have a non-empty findings section" % (len(keep), len(df)))
    for spec in args.systems:
        name, src = spec.split("=", 1)
        src = Path(src)
        hyps_all = src.joinpath("hyps.txt").read_text().splitlines()
        assert len(hyps_all) == len(df), (src, len(hyps_all))
        if args.hyp_mode == "full":
            # Sensitivity variant: the whole hypothesis (findings + impression), section labels removed.
            fh = [sanitize(h.replace("Findings:", " ").replace("Impression:", " ")) for h in hyps_all]
            how = "whole hypothesis, 'Findings:'/'Impression:' labels removed"
        elif src.joinpath("hyps_findings.txt").exists():
            fh = src.joinpath("hyps_findings.txt").read_text().splitlines()
            how = "hyps_findings.txt (model's own findings section)"
        elif sum(h.startswith("Findings:") for h in hyps_all) > 0.5 * len(hyps_all):
            fh = [findings_part(h) for h in hyps_all]
            how = "text between 'Findings:' and 'Impression:'"
        else:
            fh = hyps_all
            how = "whole hypothesis (findings-only generator)"
        hyps = [clean(fh[i]) if fh[i].strip() else "" for i in keep]
        suffix = "_findings_only" if args.hyp_mode == "findings" else "_findings_only_fullhyp"
        out = PROJECT_ROOT / "results" / (name + suffix)
        write_dir(out, hyps, refs)
        (out / "protocol_meta.json").write_text(json.dumps({
            "source": str(src), "n": len(keep), "hyp": how, "hyp_mode": args.hyp_mode,
            "normalisation": "R2Gen clean_report_mimic_cxr on hyps and refs",
            "empty_hyps": sum(1 for h in hyps if not h.strip())}, indent=2))
        print("  %-40s n=%d hyp=%s empty=%d" % (name, len(keep), how, sum(1 for h in hyps if not h.strip())))


# --------------------------------------------------------------------------- impression only
def impression_part(h: str) -> str:
    """Text after 'Impression:' of a hypothesis that has sections (ours, CXRMate); '' if absent."""
    if "Impression:" not in h:
        return ""
    return h.split("Impression:", 1)[1].strip()


def mode_protocol_imp(args) -> None:
    """Impression-only protocol: studies whose reference has a non-empty impression; references =
    impression section; hypotheses = the system's own impression section. A findings-only generator
    (R2Gen family, CvT2DistilGPT2) writes no impression, so its whole output is scored instead and
    the row is marked as a sensitivity check, not a like-for-like comparison. Same R2Gen cleaner on
    both sides as protocol (b)."""
    clean = load_chen_cleaner()
    df = test_rows()
    keep = [i for i, v in enumerate(df["impression"].fillna("")) if str(v).strip()]
    refs = [clean(sanitize(df["impression"].iloc[i])) for i in keep]
    print("impression-only: %d of %d studies have a non-empty impression" % (len(keep), len(df)))
    for spec in args.systems:
        name, src = spec.split("=", 1)
        src = Path(src)
        hyps_all = src.joinpath("hyps.txt").read_text().splitlines()
        assert len(hyps_all) == len(df), (src, len(hyps_all))
        if sum("Impression:" in h for h in hyps_all) > 0.5 * len(hyps_all):
            fh = [impression_part(h) for h in hyps_all]
            how, like_for_like = "text after 'Impression:'", True
        else:
            fh = [sanitize(h) for h in hyps_all]
            how, like_for_like = "whole output (findings-only generator; sensitivity only)", False
        hyps = [clean(fh[i]) if fh[i].strip() else "" for i in keep]
        out = PROJECT_ROOT / "results" / (name + "_impression_only")
        write_dir(out, hyps, refs)
        (out / "protocol_meta.json").write_text(json.dumps({
            "source": str(src), "n": len(keep), "hyp": how, "like_for_like": like_for_like,
            "normalisation": "R2Gen clean_report_mimic_cxr on hyps and refs",
            "empty_hyps": sum(1 for h in hyps if not h.strip())}, indent=2))
        print("  %-40s n=%d hyp=%s empty=%d" % (name, len(keep), how, sum(1 for h in hyps if not h.strip())))


def mode_protocol_combined(args) -> None:
    """Combined protocol (= our protocol (a)): copy a result dir's hyps/refs (and CheXbert files, if
    scored) unchanged into results/<name>_combined, so text metrics never write into a published dir."""
    import shutil
    for spec in args.systems:
        name, src = spec.split("=", 1)
        src, out = Path(src), PROJECT_ROOT / "results" / (name + "_combined")
        out.mkdir(parents=True, exist_ok=True)
        for f in ("hyps.txt", "refs.txt", "chexbert_labels.json", "chexbert_metrics.json"):
            if src.joinpath(f).exists():
                shutil.copy2(src / f, out / f)
        print("  %-40s <- %s" % (name, src))


# --------------------------------------------------------------------------- text metrics
def mode_textmetrics(args) -> None:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from evaluate_report_generation import corpus_bleu, rouge_l_score
    rows = {}
    for d in args.dirs:
        d = Path(d)
        hyps = d.joinpath("hyps.txt").read_text().splitlines()
        refs = d.joinpath("refs.txt").read_text().splitlines()
        assert len(hyps) == len(refs), d
        ht, rt = [h.split() for h in hyps], [r.split() for r in refs]
        m = {"n": len(hyps),
             "rouge_l": sum(rouge_l_score(h, r) for h, r in zip(ht, rt)) / max(len(hyps), 1),
             "bleu_1": corpus_bleu(ht, rt, max_n=1), "bleu_4": corpus_bleu(ht, rt, max_n=4)}
        d.joinpath("text_metrics.json").write_text(json.dumps(m, indent=2))
        rows[d.name] = m
        print("%-55s n=%5d ROUGE-L %.4f BLEU-1 %.4f BLEU-4 %.4f" % (d.name, m["n"], m["rouge_l"], m["bleu_1"],
                                                                   m["bleu_4"]))


def main(argv=None):
    global args_global
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["vocab", "generate", "protocol_b", "protocol_imp",
                                    "protocol_combined", "textmetrics"])
    ap.add_argument("--parquet", default=str(DATA / "test.parquet"))
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--systems", nargs="*", default=[], help="name=result_dir for protocol_b")
    ap.add_argument("--hyp-mode", choices=["findings", "full"], default="findings",
                    help="protocol_b hypothesis: generated findings (default) or whole report (sensitivity)")
    ap.add_argument("--dirs", nargs="*", default=[], help="result dirs for textmetrics")
    args_global = ap.parse_args(argv)
    {"vocab": mode_vocab, "generate": mode_generate, "protocol_b": mode_protocol_b,
     "protocol_imp": mode_protocol_imp, "protocol_combined": mode_protocol_combined,
     "textmetrics": mode_textmetrics}[args_global.mode](args_global)


args_global = None

if __name__ == "__main__":
    main()
