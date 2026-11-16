# Glass Crack Segmentation: Model Implementation Plan (v4)

Date: 2026-11-05. Supersedes v3 (2026-10-26). Aligned with `agent_plan.md` (v4: dataset acquisition, rationale, ELI5, contingency plan); this file is the short operational version.
Scope: improve the crack model with the datasets on disk plus open proxy datasets. The end state is a curtain-wall inspection pipeline (high-resolution tiles, coarse search, pixel segmentation, false-positive filtering, human review). Licensing is deferred (Section 9).

**Changes from v3:** NCUGD is dropped (out of domain, 0/32 recall for every model). The curtain-wall authors did not reply, so CUBIT-InSeg and CarDD become proxy pretraining arms, and the Brandon-Johns curtain-wall set (no crack labels) supplies hard negatives. Track B (semantic segmentation) is **retained** for academic cross-architecture comparisons against Track A.

---

## 1. Goal

**Now:** make E1 more reliable on the glass images we have; measure whether more pixels (tiles, 1024/1280 input) beat a single 640 px resize; test augmentation, hard negatives and proxy pretraining one at a time.

**Later:** a facade photo goes in; the tool returns a crack mask, location, confidence and the source crop for a person to accept or reject.

A change counts only if it beats the seed spread and the bootstrap interval on at least one of:
- test recall at precision 0.85 and PR area on `glass_merged_seg` (threshold-free, `compare_runs`)
- false positives per 100 images
- AGDD recall, `images/` and `image/` reported separately

---

## 2. Where the project stands (TIDE, NVIDIA L40, threshold frozen at max F1 on valid)

| Run | Test recall | Precision | FP / 100 | Mask IoU | Recall @ P≥0.85 | PR area |
|---|---|---|---|---|---|---|
| E1 YOLO-seg, 4 classes, 3 seeds | 0.767 ± 0.041 | 0.865 ± 0.008 | 25.0 ± 1.2 | 0.603 ± 0.028 | 0.825 ± 0.010 | 0.828 ± 0.002 |
| E2 YOLO-seg, `linear_defect`, 3 seeds | 0.772 ± 0.051 | 0.834 ± 0.032 | 32.9 ± 7.9 | 0.587 ± 0.014 | 0.706 ± 0.105 | 0.818 ± 0.010 |
| E3 U-Net ResNet34, 1 seed | 0.634 | 0.643 | 74.3 | 0.448 | 0.000 | 0.427 |
| E4old (scale/colour only), 1 seed | 0.814 | 0.819 | 36.5 | 0.627 | - | - |

- E1 is the Track A model to build on. E3's AGDD number is inflated by box scoring of blobby masks.
- AGDD `image/` responds far more weakly than `images/` with the same labels (model robustness, not data). Report views separately.
- The frozen threshold swings 0.30–0.50 on 148 valid images; rank with threshold-free metrics.

---

## 3. Decisions

| Decision | Choice | Why |
|---|---|---|
| Track A base | E1, `yolo26s-seg`, 4 classes, scored as crack OR scratch | Lowest FP, stable precision |
| Track B | Kept: `train_semseg.py` with any segmentation-models-pytorch `arch` / `encoder` (default `unet` + `resnet34` = E3) | Academic comparison of semantic-segmentation architectures against YOLO under the same protocol |
| Headline class | `linear_defect` at evaluation | Crack/scratch labels disagree; bubble/inclusion stay hard negatives |
| Training data | `glass_merged_seg`; proxies only as isolated arms | Pooled crack sets did not beat per-dataset training |
| Never trained on | AGDD | External test |
| Dropped | NCUGD | Out of domain |
| Input size | R0–R2 ablation before more long runs | Pixels across a crack may matter more than architecture |
| Seeds | 3 per retrained config compared with E1 | Seed spread (±0.04) exceeds the E1–E2 gap |
| Parked | SSGD, CrackSeg9k, deblur GAN, LoRA-SAM, clDice loss | After M1–M4 |

---

## 4. Compute

TIDE (https://jupyterhub-research.sdsu.edu, PyTorch Notebook, L40, 8 CPU, 32 GB, 75 GiB `/home/jovyan`). Server stops after 12 h without a browser; `scripts/tide/job.sh` keeps one job alive across closed tabs; re-running a job name resumes. Session notebook: `notebooks/tide_v4.ipynb`. A 150-epoch E1-style run takes about 2.5 h; the full v4 sweep about 25–30 GPU hours.

Laptop (`env .venv/bin/python`, CPU): unit tests, `compare_runs`, building `data/extras.zip` (`scripts/build_extras.sh`), `predict.py` on a few images.

---

## 5. Data

| Set | Role |
|---|---|
| `processed/glass_merged_seg` | Train 980 / valid 148 / test 148, 4 classes. Fixed group-aware splits |
| `processed/glass_merged_linear` | Built by `relabel_classes`; E2 only |
| `datasets/agdd_aircraft_glass/data` | External test, per view; V1 two-view filter |
| `extras/cubit_tiles` | 58,844 CUBIT CR1 tiles (1024 px windows -> 640), `linear_defect`: E5 pretraining |
| `extras/cardd` | CarDD crack + glass shatter, 1,077 / 183: E6 pretraining |
| `extras/hardneg` | 1,441 crack-free tiles (curtain-wall 840, CUBIT 600, facade 1), half with synthetic glare: E4b |
| GSDNet CGD | Blocked (Baidu account); `agent_plan.md` §1 |

Rules: thresholds and checkpoints from valid only; test scored once per config; augmentation in train only; external sets never pick anything.

---

## 6. Evaluation protocol (fixed; `protocol.py`)

- **Metrics** (`linear_defect`): recall, precision, mask IoU, FP/100 (all / unannotated / bubble-inclusion slices), mask mAP50 (Track A); threshold-free recall at precision 0.80/0.85/0.90 and PR area.
- **Matching:** mask IoU ≥ 0.3, or overlap after a 3 px dilation of the prediction; strict IoU 0.5 also reported; one prediction may cover touching GTs via union IoU; regions < 24 px² dropped. Track B instances are 8-connected components.
- **Operating point:** max F1 on valid, grid 0.05–0.95.
- **Uncertainty:** cluster bootstrap over near-duplicate groups, 2,000 resamples, 95% CI.
- **AGDD:** scratches + crack are GT, contusion/spot are confusions; hit = box IoU ≥ 0.3 or ≥ 50% of predicted pixels in the oriented box; `images/` and `image/` separately.

---

## 7. Work order (`notebooks/tide_v4.ipynb`)

| Step | Run | How | Done when |
|---|---|---|---|
| M0 | Baselines | `compare_runs`, `agdd_box_scores` | Done |
| M1 | R0 whole@640, R1 tiles over long side 1280, R2 whole@1024/1280 on frozen E1_s0 | `run_eval --imgsz N [--whole] --exp-id R*` (Track A and B checkpoints) | Decides E4's scale strategy |
| M2 | E4 = E1 + scale jitter, colour, motion blur, JPEG; 3 seeds | `e4_yolo_robust.yaml` | Beats E1 threshold-free, or not |
| M3 | E8 on audited labels (100 crack + 100 scratch crops); E4b = E4 + 30% hard negatives | `e4b_yolo_hardneg.yaml` | FP/100 falls without losing recall @ P 0.85 |
| M4 A | E5pre (CUBIT, 30 epochs) -> E5 fine-tune, 3 seeds | `e5_pre_cubit.yaml`, `e5_yolo_cubit_ft.yaml` | Glass test holds or improves |
| M4 B | E6pre (CarDD) -> E6, only if E5 does not help | `e6_pre_cardd.yaml`, `e6_yolo_cardd_ft.yaml` | Same |
| Track B | Extra `arch` / `encoder` configs (e.g. FPN, DeepLabV3+, SegFormer `mit_b2`) under the same protocol | `train_semseg.py` | Cross-architecture table vs E1 |
| V1 | AGDD two-view consistency filter | cached maps | Reported as aircraft glass only |

---

## 8. Experiment register

| ID | What changes | Compared with |
|---|---|---|
| E1 | 4-class YOLO-seg, 3 seeds (done) | Reference |
| E2 | `linear_defect` head (done) | E1 |
| E3 | U-Net ResNet34 (done); Track B baseline | E1 and other Track B architectures |
| R0–R2 | Inference resolution only | E1 seed 0 |
| E4 | E1 + scale, colour, motion blur, JPEG | E1 |
| E8 | E1 on audited labels | E1 |
| E4b | E4 + 30% hard negatives | E4 |
| V1 | AGDD two-view filter | Filter off |
| E5 | CUBIT pretrain -> glass fine-tune | E4b / E1 |
| E6 | CarDD pretrain -> glass fine-tune | E5 / E4b |

---

## 9. Deferred

Licenses (Ultralytics AGPL-3.0, SSGD academic-only, CarDD and AGDD non-commercial, Roboflow upstream rights), company agreements, privacy, drone hardware. Curtain-wall micro test set (`agent_plan.md` §5) once images are scraped and annotated.

---

## 10. Definition of done

1. E1, E4 and R0–R2 in one threshold-free table with per-view AGDD.
2. E4b and E8 compared with E1 on FP/100.
3. E5 (and E6 if run) evaluated: does proxy pretraining help the glass test?
4. At least one extra Track B architecture compared with E3 and E1 under the same protocol.
5. `predict.py` runs on the winning checkpoint for a new facade photo.
