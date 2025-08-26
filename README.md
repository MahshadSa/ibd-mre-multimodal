# IBD Multimodal Blueprint

**Scalable multimodal modeling of IBD activity from MR Enterography, clinical data, and radiology reports.**

> Status: Done: synthetic MVP + image segmentation pretrain + CLIP dummy pretrain.  
> Next: calibration & reliability plots, cross-attention fusion, report→labels extractor.

## Why this matters
Clinical decisions combine **images + labs + narrative**. This baseline can grow into a clinically meaningful model.

## Repo structure

ibd-mre-multimodal/
├─ data/ # local data (ignored by git)
├─ src/ # training code, datasets, branches, pretrain
├─ artifacts/ # checkpoints (ignored by git)
├─ .vscode/launch.json # VS Code run configs
└─ README.md


## Quickstart (Windows, Python 3.12)


py -3.12 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt