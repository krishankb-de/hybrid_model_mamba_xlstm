# ISBI floors: encoder vetting (B1, 2026-10-04)

| key | HF id | loader | input | embed | licence | pretraining data | MIMIC-CXR test overlap (R2) |
|---|---|---|---|---|---|---|---|
| biomedclip | microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 | open_clip | 224 | 512 | MIT | PMC-15M figure-caption pairs | none (no MIMIC) |
| clip | openai/clip-vit-base-patch16 | transformers CLIPModel | 224, centre crop, CLIP mean/std | 512 | MIT | web image-text pairs | none (no MIMIC) |
| pubmedclip | flaviagiammarino/pubmed-clip-vit-base-patch32 | transformers CLIPModel | 224, centre crop, CLIP mean/std | 512 | MIT | ROCO (PubMed figures), model card | none (no MIMIC) |
| xrayclip | StanfordAIMI/XrayCLIP__vit-b-16__laion2b-s34b-b88k | transformers CLIPModel | 512, no crop, CLIP mean/std | 512 | CC-BY-NC-4.0 | CheXagent project (CheXinstruct, incl. MIMIC-CXR) | none expected: CheXagent "strictly followed the official or traditional dataset splits ... to prevent data leakage" (arXiv 2401.12208 v2) |
| medsiglip | google/medsiglip-448 | transformers SiglipModel (gated, HAI-DEF terms) | 448, normalised to (-1, 1) | 1152 | HAI-DEF | medical image-text pairs incl. MIMIC-CXR (model card) | probably none: MedGemma report evaluates on the official MIMIC-CXR test split (MAIRA test set, 2,461 frontal) |

Dropped: **BioViL-T** (microsoft/BiomedVLP-BioViL-T). It pretrains on MIMIC-CXR v2 with its own
"disjoint held-out test set ... 2971" (Bannur et al., CVPR 2023), not the official split, so it may
have seen our test studies (R2). Its image loader also needs the extra `hi-ml-multimodal` package.

Each HF encoder is embedded with its own AutoImageProcessor. MedSigLIP's card recommends TF bilinear
resize for exact reproduction. We use the HF processor, which is the standard PyTorch path.
Embedding = vision tower pooled output, through visual_projection for CLIP models (none for SigLIP).
