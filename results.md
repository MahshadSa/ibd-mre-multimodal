# Results

> Runs on a laptop (GTX 1650, PyTorch 2.4, Python 3.12). This repo is intentionally runnable without protected data.

## Segmentation pretrain (synthetic MRE-style)
- Image size: **256** | Batch: **4** | Epochs: **4**
- **Best Val Dice:** <paste from console, e.g., 0.67>
- Checkpoints: `artifacts/mre_seg_pretrain/seg_best.pt`, `artifacts/mre_seg_pretrain/resnet18_mre_encoder.pt`
- Figure: `docs/figures/seg_val_examples.png`

## CLIP dummy pairs (smoke test)
- Epochs: **2** | Batch: **32** | Text model: **distilbert-base-uncased**
- Retrieval (val): **R@1 <…> | R@5 <…> | R@10 <…>**
- Checkpoint: `artifacts/clip_pretrain/clip_pretrain.pt`

## Scope for this submission
- Code runs end-to-end on synthetic data.
- Image branch adapted via segmentation pretrain (Stage 2A).
- Remaining pieces (paired late-fusion, cross-attention, ordinal loss, calibration, decision curves, agent) are outlined in `docs/ROADMAP.md`.
