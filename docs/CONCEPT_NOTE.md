# Concept Note (2 pages)

## Title
Scalable Multimodal Modeling of IBD Activity from MRE, Clinical Data, and Radiology Reports

## Summary
- Decisions rely on images + labs + narrative.
- MVP here: runnable late-fusion baseline + image encoder adapted via segmentation pretrain; CLIP smoke test for future image–text alignment.
- Extensions: cross-attention fusion, ordinal objectives, uncertainty calibration, and a clinician-readable summary agent.

## Why this matters
- Clinically meaningful: mirrors real decision flow.
- Technically aligned with multimodal clinical AI work.
- Feasible: MVP → iterative upgrades → potential internal validation.

## Data plan
- Primary: approved IBD MRE cohort with segment-level labels.
- Interim/demo: synthetic + public proxies to exercise code paths.

## Model plan
- Image: ResNet-18 encoder adapted via seg pretrain (MRE-style); later ViT/Swin.
- Text: DistilBERT/ClinicalBERT; lightweight adapters later.
- Tabular: MLP with normalizer.
- Fusion: MVP late fusion → cross-attention.

## Tasks & outputs
- Patient/segment-level activity (binary/ordinal).
- Calibrated probabilities; weak localization (attention/CAMs).
- Clinician-readable structured output from model + report cues.

## Evaluation
- AUROC/AUPRC; ECE/Brier; Dice (seg); decision curves; ablations.

## Risks & mitigations
- Data access/heterogeneity → synthetic demo + pluggable loaders.
- Label noise/ordinal → ordinal losses; bootstrap CIs.
- Generalization → robust aug; site/time splits; uncertainty reporting.

## Milestones (6–8 weeks once embedded)
- W1–2: EDA + baselines (image-only, text-only).
- W3–4: Late fusion + calibration; reporting agent draft.
- W5–6: Cross-attention + ablations; error analysis.
- W7–8: Internal note + preprint draft; plan external validation.
