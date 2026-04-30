"""Contrastive training eval callbacks for Stage 1 SimCSE.

Runs BIOSSES and STS-B Spearman evals, cosine-similarity stats, and
alignment/uniformity (Wang & Isola 2020) at configurable step intervals.
All metrics logged via pl_module.log() — no W&B dependency.

Also includes AnomalyDetectionCallback to enable torch.autograd anomaly
detection for the first N steps (debugging aid, disable after first clean run).
"""

import math
from typing import Any, List, Optional, Tuple

import torch
import torch.nn.functional as F
import pytorch_lightning as pl


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _spearman_rho(x: List[float], y: List[float]) -> float:
    """Compute Spearman rank correlation without scipy dependency."""
    n = len(x)
    if n < 2:
        return 0.0

    def _rank(vals: List[float]) -> List[float]:
        sorted_idx = sorted(range(n), key=lambda i: vals[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and vals[sorted_idx[j + 1]] == vals[sorted_idx[j]]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[sorted_idx[k]] = avg_rank
            i = j + 1
        return ranks

    rx = _rank(x)
    ry = _rank(y)
    d2 = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    return 1.0 - 6.0 * d2 / (n * (n * n - 1))


def _cosine_sim_batch(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Per-pair cosine similarity: (B,)"""
    return (z1 * z2).sum(dim=-1)  # already L2-normalised


def _alignment(z1: torch.Tensor, z2: torch.Tensor) -> float:
    """Alignment: mean squared L2 distance between positive pairs (lower = better alignment)."""
    return (z1 - z2).pow(2).sum(dim=-1).mean().item()


def _uniformity(z: torch.Tensor) -> float:
    """Uniformity: log mean exp(-2 * pairwise squared distances) (more negative = more uniform)."""
    sq_dists = torch.pdist(z, p=2).pow(2)
    return sq_dists.mul(-2.0).exp().mean().log().item()


# ---------------------------------------------------------------------------
# Main eval callback
# ---------------------------------------------------------------------------

class ContrastiveEvalCallback(pl.Callback):
    """In-training biomedical eval for Stage 1 SimCSE.

    Runs:
      - BIOSSES Spearman ρ every ``eval_every_n_steps`` steps
      - STS-B dev Spearman ρ every ``eval_every_n_steps`` steps
      - Cosine-sim mean + std, mean embedding norm every ``eval_every_n_steps`` steps
      - Alignment + uniformity (Wang & Isola 2020) every ``align_unif_every_n_steps`` steps

    All metrics logged via pl_module.log() so they appear in TensorBoard.
    """

    def __init__(
        self,
        tokenizer: Any,
        eval_every_n_steps: int = 500,
        align_unif_every_n_steps: int = 1000,
        max_eval_samples: int = 512,
        device_override: Optional[str] = None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.eval_every = eval_every_n_steps
        self.align_unif_every = align_unif_every_n_steps
        self.max_eval_samples = max_eval_samples
        self.device_override = device_override

        # Populated in on_fit_start
        self.biosses_pairs: List[Tuple[str, str, float]] = []
        self.stsb_pairs: List[Tuple[str, str, float]] = []
        self._datasets_loaded = False

    # ------------------------------------------------------------------
    # Dataset loading (deferred to avoid import cost at module load)
    # ------------------------------------------------------------------

    def _load_datasets(self) -> None:
        if self._datasets_loaded:
            return
        self._datasets_loaded = True

        try:
            from datasets import load_dataset as _ld
        except ImportError:
            print("[ContrastiveEvalCallback] 'datasets' not installed — skipping eval.")
            return

        # BIOSSES — 100 biomedical sentence pairs, scores 0–4.
        # Uses mteb/biosses (Parquet, no loading script) — bigbio/biosses requires
        # a loading script that HuggingFace no longer supports.
        try:
            ds = _ld("mteb/biosses", split="test")
            for row in ds:
                t1 = str(row.get("sentence1") or row.get("text_1") or "")
                t2 = str(row.get("sentence2") or row.get("text_2") or "")
                sc = float(row.get("score") or row.get("label") or 0.0)
                if t1 and t2:
                    self.biosses_pairs.append((t1, t2, sc))
            print(f"[ContrastiveEvalCallback] BIOSSES loaded: {len(self.biosses_pairs)} pairs")
        except Exception as exc:
            print(f"[ContrastiveEvalCallback] BIOSSES load failed ({exc}); skipping BIOSSES eval.")

        # STS-B dev — ~1500 general sentence pairs, scores 0–5
        try:
            ds = _ld("glue", "stsb", split="validation")
            for row in ds:
                t1 = str(row.get("sentence1") or "")
                t2 = str(row.get("sentence2") or "")
                sc = float(row.get("label") or 0.0)
                if t1 and t2:
                    self.stsb_pairs.append((t1, t2, sc))
            # Cap to max_eval_samples for speed
            self.stsb_pairs = self.stsb_pairs[: self.max_eval_samples]
            print(f"[ContrastiveEvalCallback] STS-B loaded: {len(self.stsb_pairs)} pairs")
        except Exception as exc:
            print(f"[ContrastiveEvalCallback] STS-B load failed ({exc}); skipping STS-B eval.")

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _encode_pairs(
        self,
        pl_module: pl.LightningModule,
        pairs: List[Tuple[str, str, float]],
        max_length: int = 512,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[float]]:
        """Encode all sentence pairs, return (z1, z2, scores)."""
        device = self.device_override or next(pl_module.model.parameters()).device

        sents1 = [p[0] for p in pairs]
        sents2 = [p[1] for p in pairs]
        scores = [p[2] for p in pairs]

        was_training = pl_module.model.training
        pl_module.model.eval()
        z1_parts: List[torch.Tensor] = []
        z2_parts: List[torch.Tensor] = []

        try:
            batch_size = 32
            for i in range(0, len(sents1), batch_size):
                enc1 = self.tokenizer(
                    sents1[i : i + batch_size],
                    max_length=max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                enc2 = self.tokenizer(
                    sents2[i : i + batch_size],
                    max_length=max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                ids1 = enc1["input_ids"].to(device)
                mask1 = enc1["attention_mask"].to(device)
                ids2 = enc2["input_ids"].to(device)
                mask2 = enc2["attention_mask"].to(device)

                z1_parts.append(pl_module.model.encode(ids1, attention_mask=mask1).cpu())
                z2_parts.append(pl_module.model.encode(ids2, attention_mask=mask2).cpu())
        finally:
            pl_module.model.train(was_training)

        z1 = torch.cat(z1_parts, dim=0)
        z2 = torch.cat(z2_parts, dim=0)
        return z1, z2, scores

    # ------------------------------------------------------------------
    # Metric computations
    # ------------------------------------------------------------------

    def _eval_biosses(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self.biosses_pairs:
            return
        try:
            z1, z2, scores = self._encode_pairs(pl_module, self.biosses_pairs)
            cos_sims = _cosine_sim_batch(z1, z2).tolist()
            rho = _spearman_rho(cos_sims, scores)
            # on_step=True required when logging from on_train_batch_end
            pl_module.log("train/biosses_spearman", rho, prog_bar=True, on_step=True, on_epoch=False)
            print(f"[step={trainer.global_step}] BIOSSES Spearman ρ = {rho:.4f}")
        except Exception as exc:
            print(f"[ContrastiveEvalCallback] BIOSSES eval error: {exc}")

    def _eval_stsb(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self.stsb_pairs:
            return
        try:
            z1, z2, scores = self._encode_pairs(pl_module, self.stsb_pairs)
            cos_sims = _cosine_sim_batch(z1, z2).tolist()
            rho = _spearman_rho(cos_sims, scores)
            # on_step=True required when logging from on_train_batch_end
            pl_module.log("train/stsb_spearman", rho, prog_bar=True, on_step=True, on_epoch=False)
            print(f"[step={trainer.global_step}] STS-B Spearman ρ = {rho:.4f}")
        except Exception as exc:
            print(f"[ContrastiveEvalCallback] STS-B eval error: {exc}")

    def _log_cosine_stats(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
    ) -> None:
        """Log off-diagonal cosine sim stats and mean embedding norm from training batch."""
        try:
            device = next(pl_module.model.parameters()).device
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch.get("attention_mask")
            if attn_mask is not None:
                attn_mask = attn_mask.to(device)

            was_training = pl_module.model.training
            pl_module.model.eval()
            try:
                with torch.no_grad():
                    z = pl_module.model.encode(input_ids, attention_mask=attn_mask)
            finally:
                pl_module.model.train(was_training)

            B = z.size(0)
            if B < 2:
                return

            # Off-diagonal cosine similarities
            sim_mat = z @ z.T
            mask = ~torch.eye(B, dtype=torch.bool, device=z.device)
            off_diag = sim_mat[mask]
            mean_cos = off_diag.mean().item()
            std_cos = off_diag.std().item()
            mean_norm = z.norm(dim=-1).mean().item()

            pl_module.log("train/offdiag_cosine_mean", mean_cos, on_step=True, on_epoch=False)
            pl_module.log("train/offdiag_cosine_std", std_cos, on_step=True, on_epoch=False)
            pl_module.log("train/embed_norm_mean", mean_norm, on_step=True, on_epoch=False)
        except Exception as exc:
            print(f"[ContrastiveEvalCallback] cosine stats error: {exc}")

    def _log_alignment_uniformity(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
    ) -> None:
        """Log Wang & Isola 2020 alignment + uniformity from training batch."""
        try:
            device = next(pl_module.model.parameters()).device
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch.get("attention_mask")
            if attn_mask is not None:
                attn_mask = attn_mask.to(device)

            was_training = pl_module.model.training
            pl_module.model.eval()
            try:
                with torch.no_grad():
                    z1 = pl_module.model.encode(input_ids, attention_mask=attn_mask)
                    z2 = pl_module.model.encode(input_ids, attention_mask=attn_mask)
            finally:
                pl_module.model.train(was_training)

            align = _alignment(z1, z2)
            unif = _uniformity(z1)
            pl_module.log("train/alignment", align, on_step=True, on_epoch=False)
            pl_module.log("train/uniformity", unif, on_step=True, on_epoch=False)
        except Exception as exc:
            print(f"[ContrastiveEvalCallback] alignment/uniformity error: {exc}")

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._load_datasets()

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        step = trainer.global_step
        if step > 0 and step % self.eval_every == 0:
            self._eval_biosses(trainer, pl_module)
            self._eval_stsb(trainer, pl_module)
            self._log_cosine_stats(trainer, pl_module, batch)

        if step > 0 and step % self.align_unif_every == 0:
            self._log_alignment_uniformity(trainer, pl_module, batch)


# ---------------------------------------------------------------------------
# Anomaly detection helper (disable after first clean run)
# ---------------------------------------------------------------------------

class AnomalyDetectionCallback(pl.Callback):
    """Enable torch.autograd anomaly detection for the first N training steps.

    Disable once the run is confirmed clean — detection adds ~10x overhead.
    """

    def __init__(self, max_steps: int = 200):
        super().__init__()
        self.max_steps = max_steps

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        torch.autograd.set_detect_anomaly(trainer.global_step < self.max_steps)

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        torch.autograd.set_detect_anomaly(False)


# ---------------------------------------------------------------------------
# Stage 2 retrieval evaluation: R@1, R@5, R@10 (image→text and text→image)
# ---------------------------------------------------------------------------

class CLIPRetrievalCallback(pl.Callback):
    """Compute image-text retrieval metrics on the validation dataloader.

    At the end of each validation epoch, embeds the full validation set and
    computes R@1, R@5, R@10 in both directions (image→text and text→image).

    Requires ``pl_module.image_encoder``, ``pl_module.img_proj``, and
    ``pl_module.model.encode()`` to exist (i.e. CLIP mode).

    Args:
        eval_every_n_epochs: Run retrieval eval every N epochs (default 1).
        max_samples:         Cap validation set at this many samples to avoid
                             OOM on large datasets. 0 = no cap.
    """

    def __init__(
        self,
        eval_every_n_epochs: int = 1,
        max_samples: int = 0,
    ):
        super().__init__()
        self.eval_every_n_epochs = eval_every_n_epochs
        self.max_samples = max_samples

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        if trainer.current_epoch % self.eval_every_n_epochs != 0:
            return
        if not hasattr(pl_module, 'image_encoder') or pl_module.image_encoder is None:
            return

        val_loader = trainer.val_dataloaders
        if val_loader is None:
            return
        # PyTorch Lightning wraps in a list for single loader
        if isinstance(val_loader, list):
            val_loader = val_loader[0]

        device = pl_module.device
        all_z_img = []
        all_z_txt = []

        pl_module.eval()
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if self.max_samples > 0 and i * len(batch["input_ids"]) >= self.max_samples:
                    break
                input_ids = batch["input_ids"].to(device)
                attn_mask = batch.get("attention_mask")
                if attn_mask is not None:
                    attn_mask = attn_mask.to(device)
                pixel_values = batch["pixel_values"].to(device)

                z_txt = pl_module.model.encode(input_ids, attention_mask=attn_mask)
                z_img_raw = pl_module.image_encoder(pixel_values)
                z_img = F.normalize(
                    pl_module.img_proj(z_img_raw.float()), dim=-1
                )
                all_z_img.append(z_img.cpu())
                all_z_txt.append(z_txt.cpu())

        if not all_z_img:
            return

        Z_img = torch.cat(all_z_img, dim=0)
        Z_txt = torch.cat(all_z_txt, dim=0)
        N = Z_img.shape[0]

        # Pairwise cosine similarity matrix (N, N)
        sim = Z_img @ Z_txt.T  # already L2-normalised

        ks = [1, 5, 10]
        metrics = {}
        for direction, rows, cols in [("i2t", sim, sim), ("t2i", sim.T, sim.T)]:
            mat = rows if direction == "i2t" else cols
            for k in ks:
                k_eff = min(k, N)
                top_k = mat.topk(k_eff, dim=1).indices  # (N, k)
                labels = torch.arange(N).unsqueeze(1)   # (N, 1)
                hits = (top_k == labels).any(dim=1).float().mean().item()
                key = f"val/retrieval_{direction}_R@{k}"
                metrics[key] = hits

        for key, val in metrics.items():
            pl_module.log(key, val, prog_bar=True, on_epoch=True, sync_dist=True)

        r1 = (metrics.get("val/retrieval_i2t_R@1", 0) +
              metrics.get("val/retrieval_t2i_R@1", 0)) / 2
        print(
            f"[CLIPRetrieval epoch={trainer.current_epoch}] "
            f"i2t R@1={metrics.get('val/retrieval_i2t_R@1', 0):.3f} "
            f"R@5={metrics.get('val/retrieval_i2t_R@5', 0):.3f} "
            f"R@10={metrics.get('val/retrieval_i2t_R@10', 0):.3f} | "
            f"t2i R@1={metrics.get('val/retrieval_t2i_R@1', 0):.3f} "
            f"R@5={metrics.get('val/retrieval_t2i_R@5', 0):.3f} "
            f"R@10={metrics.get('val/retrieval_t2i_R@10', 0):.3f} | "
            f"mean_R@1={r1:.3f} | N={N}"
        )
