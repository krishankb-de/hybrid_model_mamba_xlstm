"""ISBI_BASELINES_PLAN.md B9: does a model hallucinate findings or omit them?

From the CheXbert label dumps (chexbert_labels.json: y_true = reference, y_pred = generated,
already binarised: present/uncertain = 1, absent/blank = 0):

* hallucination rate of finding c = FP / (studies whose reference is negative for c)
* omission rate of finding c      = FN / (studies whose reference is positive for c)
* positive findings per report (generated vs reference), "No Finding" excluded

AGGREGATE ONLY goes to --output (rule R6). With --qualitative, a second file with study ids and
report texts for a few cases per acute finding is written; it is MIMIC-derived and must stay on
the cluster under results/.

    python scripts/isbi_error_analysis.py --system mLMamba_s42=results/report_gen_m3_test_split_s42 \
        --system floor=results/retrieval_floor_test_split --output analysis/isbi_error_analysis.json
"""
import argparse
import json
from pathlib import Path

ACUTE = ["Pneumothorax", "Consolidation", "Pneumonia", "Edema", "Pleural Effusion"]


def load(dirpath: str):
    d = json.loads((Path(dirpath) / "chexbert_labels.json").read_text())
    return d["y_true"], d["y_pred"], d["label_names"]


def rates(y_true, y_pred, names):
    per = {}
    for j, name in enumerate(names):
        tp = fp = fn = tn = 0
        for t, p in zip(y_true, y_pred):
            if t[j] and p[j]:
                tp += 1
            elif p[j]:
                fp += 1
            elif t[j]:
                fn += 1
            else:
                tn += 1
        per[name] = {
            "ref_pos": tp + fn, "ref_neg": fp + tn, "tp": tp, "fp": fp, "fn": fn,
            "hallucination_rate": fp / max(fp + tn, 1),
            "omission_rate": fn / max(tp + fn, 1),
            "precision": tp / max(tp + fp, 1),
            "recall": tp / max(tp + fn, 1),
        }
    nf = names.index("No Finding") if "No Finding" in names else None
    keep = [j for j in range(len(names)) if j != nf]
    gen_pos = [sum(p[j] for j in keep) for p in y_pred]
    ref_pos = [sum(t[j] for j in keep) for t in y_true]
    acute_idx = [names.index(a) for a in ACUTE if a in names]
    any_fp_acute = sum(any(p[j] and not t[j] for j in acute_idx) for t, p in zip(y_true, y_pred))
    any_fn_acute = sum(any(t[j] and not p[j] for j in acute_idx) for t, p in zip(y_true, y_pred))
    tot = {k: sum(per[n][k] for n in names if n != "No Finding") for k in ("tp", "fp", "fn")}
    neg = sum(per[n]["ref_neg"] for n in names if n != "No Finding")
    pos = sum(per[n]["ref_pos"] for n in names if n != "No Finding")
    return {
        "per_finding": per,
        "n_studies": len(y_true),
        "mean_positive_findings_generated": sum(gen_pos) / len(gen_pos),
        "mean_positive_findings_reference": sum(ref_pos) / len(ref_pos),
        "overall_hallucination_rate": tot["fp"] / max(neg, 1),
        "overall_omission_rate": tot["fn"] / max(pos, 1),
        "studies_with_hallucinated_acute_finding": any_fp_acute / len(y_true),
        "studies_with_omitted_acute_finding": any_fn_acute / len(y_true),
        "acute_findings": ACUTE,
    }


def qualitative(dirpath, y_true, y_pred, names, per_case=3):
    hyps = (Path(dirpath) / "hyps.txt").read_text().splitlines()
    refs = (Path(dirpath) / "refs.txt").read_text().splitlines()
    lines = [f"# Qualitative cases from {dirpath} (MIMIC-derived: do not copy off the cluster)\n"]
    for a in ACUTE:
        j = names.index(a)
        for kind, cond in (("HALLUCINATED", lambda t, p: p[j] and not t[j]),
                           ("OMITTED", lambda t, p: t[j] and not p[j])):
            idx = [i for i, (t, p) in enumerate(zip(y_true, y_pred)) if cond(t, p)][:per_case]
            lines.append(f"\n## {a}: {kind} ({len(idx)} shown)\n")
            for i in idx:
                lines.append(f"- test row {i}\n  - generated: {hyps[i]}\n  - reference: {refs[i]}\n")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", action="append", required=True, help="name=result_dir")
    ap.add_argument("--output", required=True)
    ap.add_argument("--qualitative", default=None, help="name=out_path (cluster-only file)")
    args = ap.parse_args()

    result = {}
    loaded = {}
    for spec in args.system:
        name, d = spec.split("=", 1)
        y_true, y_pred, names = load(d)
        loaded[name] = (d, y_true, y_pred, names)
        result[name] = rates(y_true, y_pred, names)
        r = result[name]
        print(f"{name:24s} halluc {r['overall_hallucination_rate']:.3f}  omit {r['overall_omission_rate']:.3f}  "
              f"pos/report gen {r['mean_positive_findings_generated']:.2f} ref {r['mean_positive_findings_reference']:.2f}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2))
    if args.qualitative:
        name, out = args.qualitative.split("=", 1)
        d, y_true, y_pred, names = loaded[name]
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(qualitative(d, y_true, y_pred, names))
        print(f"qualitative cases written to {out} (cluster only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
