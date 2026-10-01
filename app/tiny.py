"""Tiny random-init stand-ins for laptop work (CHAT_UI_PLAN.md P2-B, P2-D). No weights, no data.

The decoder config is the one tests/test_mamba3_numerics.py::_cached_lm proves token-identical
between the cached and uncached beam search (M6-D), so the tiny engine exercises the real cached
path. Ids decode to radiology words so the UI shows something report-shaped.
"""
from typing import List, Sequence

import torch
import torch.nn as nn

TINY_VOCAB = [
    "Findings:", "Impression:", "The", "the", "heart", "size", "is", "normal", "mediastinal", "contours",
    "are", "within", "limits", "lungs", "clear", "no", "focal", "consolidation", "pleural", "effusion",
    "or", "pneumothorax", "seen", "there", "mild", "moderate", "small", "large", "left", "right",
    "bilateral", "basilar", "atelectasis", "opacity", "opacities", "pulmonary", "edema", "vascular",
    "congestion", "cardiomegaly", "enlarged", "cardiac", "silhouette", "stable", "unchanged", "compared",
    "to", "prior", "study", "of", "and", "with", "without", "evidence", "acute", "cardiopulmonary",
    "process", "abnormality", "lobe", "upper", "lower", "middle", "chest", "tube", "endotracheal",
    "nasogastric", "line", "catheter", "tip", "terminates", "in", "pacemaker", "leads", "sternotomy",
    "wires", "fracture", "rib", "degenerative", "changes", "spine", "hilar", "aortic", "knob",
    "calcified", "granuloma", "nodule", "lesion", "pneumonia", "infection", "may", "be", "present",
    "likely", "low", "volumes", ".", ",",
]


def tiny_decoder_config():
    from hybrid_xmamba.models.configuration_hybrid import HybridConfig
    return HybridConfig(vocab_size=len(TINY_VOCAB), dim=64, num_layers=4, layer_pattern=["mamba3", "mlstm"],
                        head_dim=16, num_heads=4, max_position_embeddings=512, tfla_impl="exact",
                        mamba3_d_state=32, mamba3_head_dim=16, mamba3_chunk_size=8)


def tiny_decoder(seed: int = 0):
    from hybrid_xmamba.models.hybrid_lm import HybridLanguageModel
    torch.manual_seed(seed)
    return HybridLanguageModel(tiny_decoder_config()).eval()


class TinyTokenizer:
    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        return " ".join(TINY_VOCAB[i] for i in ids).replace(" .", ".").replace(" ,", ",")
