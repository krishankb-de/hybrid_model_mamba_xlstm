"""Per-turn scores with the thesis tables' own functions (CHAT_UI_PLAN.md P5-E)."""
import sys
from typing import Any, Dict, List, Optional

_PATH = list(sys.path)   # scripts.bootstrap_compare puts scripts/ first on sys.path when it loads; the server's imports keep theirs
from scripts.bootstrap_compare import chexbert_f1  # noqa: E402
from scripts.evaluate_report_generation import corpus_bleu, rouge_l_score  # noqa: E402
sys.path[:] = _PATH


def score_pair(hyp: str, ref: str, y_hyp: Optional[List[int]] = None,
               y_ref: Optional[List[int]] = None) -> Dict[str, Any]:
    h, r = hyp.split(), ref.split()
    out = {"rouge_l": rouge_l_score(h, r), "bleu_1": corpus_bleu([h], [r], 1), "bleu_4": corpus_bleu([h], [r], 4)}
    if y_hyp is not None and y_ref is not None:
        out["chexbert_14_micro_f1"] = chexbert_f1([y_ref], [y_hyp], "micro")
        out["exact_match_14"] = list(y_ref) == list(y_hyp)
    return out
