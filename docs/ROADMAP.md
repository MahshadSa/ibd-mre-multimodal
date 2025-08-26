# Roadmap — IBD Multimodal Blueprint

## Phase 0 — MVP scaffolding (done)
- ✅ Synthetic multimodal baseline + ablations (repo-ready)
- ✅ CLIP-style pretraining smoke test on dummy pairs
- ✅ Segmentation pretrain on synthetic MRE-style slices → export encoder

## Phase 1 — Image branch (1–2 weeks)
- Re-run seg pretrain on a public proxy or approved internal sample
- Save checkpoint + exported encoder; add qualitative figure
- Unit test for ImageBranch forward pass

**Deliverables:** `seg_best.pt`, `resnet18_mre_encoder.pt`, figure of val examples

## Phase 2 — Text + Tabular branches (1–1.5 weeks)
- (Optional) Light domain tuning for DistilBERT/ClinicalBERT on radiology text
- Finalize tabular schema + normalizer (missingness handling)
- Per-branch smoke tests/metrics

## Phase 3 — Late-fusion training (when paired data exists)
- Train fused classifier for IBD activity (binary + optional ordinal)
- Calibrate (temperature scaling); reliability plots
- Subgroup analysis; ablations (image-only / text-only / tab-only)

## Phase 4 — Extensions (2–3 weeks)
- Cross-attention fusion
- Ordinal loss and weak localization (CAMs/attention)
- Agent to draft structured summaries clinicians can verify

## Governance & reproducibility (ongoing)
- Seeds/configs; pinned `requirements.txt`
- No PHI in repo; CC-BY datasets only
- Bootstrap CIs; decision curves where labels allow
