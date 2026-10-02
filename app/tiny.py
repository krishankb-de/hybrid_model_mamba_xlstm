"""Tiny random-init stand-ins for laptop work (CHAT_UI_PLAN.md P2-B, P2-D). No weights, no data.

The decoder config is the one tests/test_mamba3_numerics.py::_cached_lm proves token-identical
between the cached and uncached beam search (M6-D), so the tiny engine exercises the real cached
path. Ids decode to radiology words so the UI shows something report-shaped.

Every builder seeds inside torch.random.fork_rng, so making one never moves the caller's random state.
"""
from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterator, Sequence

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from hybrid_xmamba.models.configuration_hybrid import HybridConfig
    from hybrid_xmamba.models.hybrid_lm import HybridLanguageModel
    from hybrid_xmamba.models.prefix_mapper import ImagePrefixMapper

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

TINY_DIM = 64            # decoder width
TINY_PATCH_DIM = 32      # tower patch width (768 for BiomedCLIP)
TINY_POOLED_DIM = 16     # tower pooled width (512 for BiomedCLIP)
TINY_PREFIX_K = 4        # image-prefix tokens (32 in the published models)


@contextmanager
def _seeded(seed: int) -> Iterator[None]:
    """Build under a fixed seed, then hand the caller's random state back untouched.

    default_generator.manual_seed seeds the CPU stream only; torch.manual_seed would also reseed the CUDA
    and MPS generators, which fork_rng(devices=[]) does not restore.
    """
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        yield


def tiny_decoder_config() -> "HybridConfig":
    from hybrid_xmamba.models.configuration_hybrid import HybridConfig
    return HybridConfig(vocab_size=len(TINY_VOCAB), dim=TINY_DIM, num_layers=4, layer_pattern=["mamba3", "mlstm"],
                        head_dim=16, num_heads=4, max_position_embeddings=512, tfla_impl="exact",
                        mamba3_d_state=32, mamba3_head_dim=16, mamba3_chunk_size=8)


def tiny_decoder(seed: int = 0) -> "HybridLanguageModel":
    from hybrid_xmamba.models.hybrid_lm import HybridLanguageModel
    with _seeded(seed):
        return HybridLanguageModel(tiny_decoder_config()).eval()


def tiny_prefix_mapper(seed: int = 0) -> "ImagePrefixMapper":
    from hybrid_xmamba.models.prefix_mapper import ImagePrefixMapper
    with _seeded(seed):
        return ImagePrefixMapper(patch_dim=TINY_PATCH_DIM, decoder_dim=TINY_DIM, k=TINY_PREFIX_K).eval()


class _TinyTrunk(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch = nn.Conv2d(3, TINY_PATCH_DIM, kernel_size=16, stride=16)
        self.cls = nn.Parameter(torch.zeros(1, 1, TINY_PATCH_DIM))

    def forward_features(self, x):
        p = self.patch(x).flatten(2).transpose(1, 2)                       # (B, 196, 32)
        return torch.cat([self.cls.expand(x.shape[0], -1, -1), p], dim=1)  # (B, 197, 32)

    def forward_head(self, feats):
        # No attention blocks here to pull the patches into the CLS token, so pool every token: the real
        # trunk's CLS output depends on the image, and a constant one would give every query the same vector.
        return feats.mean(dim=1)

    def forward(self, x):
        return self.forward_head(self.forward_features(x))


class TinyTower(nn.Module):
    """Stand-in for open_clip's TimmModel: same trunk/head surface, 32-d patches, 16-d pooled output."""

    def __init__(self, seed: int = 0):
        super().__init__()
        with _seeded(seed):
            self.trunk = _TinyTrunk()
            self.head = nn.Linear(TINY_PATCH_DIM, TINY_POOLED_DIM)
        self.eval()

    def forward(self, x):
        return self.head(self.trunk(x))


class TinyTokenizer:
    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        return " ".join(TINY_VOCAB[i] for i in ids).replace(" .", ".").replace(" ,", ",")
