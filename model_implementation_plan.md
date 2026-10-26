# Glass Crack Segmentation: Model Implementation Plan (v3)

Date: 2026-10-26
Scope: improve the crack model on the datasets already on disk. The end state is a curtain-wall inspection pipeline (high-resolution tiles, coarse search, pixel segmentation, false-positive filtering, human review). This revision does not chase new datasets. Curtain-wall imagery, company photos, and author requests stay parked until that data is in hand.

Licensing is still deferred (Section 10).

---

## 1. Goal

**Now:** make E1 more reliable on the glass images we already have, and measure whether keeping more pixels (tiling, higher input size) beats a single 640 px resize. Every comparison uses the same splits, the same matching rule, and more than one seed when a model is retrained.

**Later, unchanged:** a photo of a facade goes in, and the tool returns a crack mask, location, a confidence, and the source crop for a person to accept or reject. Multi-view confirmation and a deblurring stage are experiments, not the default path.

A change counts only if it moves at least one of these, by more than the seed spread or the bootstrap interval:

- test recall and precision on `glass_merged_seg`
- false positives per 100 images
- NCUGD recall (today 0/32)
- AGDD recall reported separately for `images/` and `image/`

---

## 2. Where the project stands

Trained on SDSU TIDE (NVIDIA L40), new matching rule, threshold frozen at max F1 on valid.

| Run | Test recall | Precision | FP / 100 | Mask IoU | Frozen thresholds |
|---|---|---|---|---|---|
| E1 YOLO-seg, 4 classes, 3 seeds | 0.767 ± 0.041 | 0.865 ± 0.008 | 25.0 ± 1.2 | 0.603 ± 0.028 | 0.30, 0.50, 0.40 |
| E2 YOLO-seg, `linear_defect`, 3 seeds | 0.772 ± 0.051 | 0.834 ± 0.032 | 32.9 ± 7.9 | 0.587 ± 0.014 | 0.35, 0.45, 0.45 |
| E3 U-Net ResNet34, 1 seed | 0.634 | 0.643 | 74.3 | 0.448 | 0.80 |

E1 is the model to build on. Recall ties E2. False positives are lower in every seed, and precision barely moves across seeds. E3 is behind on the glass test. Its AGDD pair number (0.833) is not evidence of a better model: box scoring of a blobby mask inflates it, and the two views must not be averaged.

Other facts that constrain the next experiments:

- NCUGD is a real miss. Labels, loading, and matching were checked. Cached E1 maps are zero near the cracks. The cracks are faint 1–2 px hairlines. One GPU check remains: the real checkpoints at 640, 1024, and 1280, to separate "blind" from "below the threshold."
- AGDD `image/` responds much more weakly than `images/` with the same labels. Shift, brightness, contrast, and crack polarity were ruled out. Report the views separately. `images/` is the external number to quote.
- The frozen threshold swings from 0.30 to 0.50 because valid has only 148 images. Rank models with recall at a fixed precision and with the area under the precision-recall curve (`glass_crack_models.eval.compare_runs`), not with one threshold.
- `predict.py` already runs overlapping 640 px tiles. It has not been compared, on the test set, with a single resized image.
- Huang and Liu, *Sensors* 2024 (10.3390/s24237713), use a GAN plus a Swin U-shape to deblur UAV glass. They report PSNR 29.49 dB on restored images. They do not report crack recall. Do not add that network unless a blur ablation shows that blur, not appearance, is what the model misses.
- An agent or MCP wrapper around the checkpoint does not change these numbers. Leave agents out of training and inference.

Code for the measurement steps below is written and unit-tested (34 tests) and still needs a commit and a push: `compare_runs.py`, `agdd_box_scores.py`, `e4_yolo_robust.yaml` based on E1, `notebooks/tide_phase1.ipynb`.

---

## 3. Decisions

| Decision | Choice | Why |
|---|---|---|
| Model to extend | E1, YOLO-seg `yolo26s-seg`, 4 classes, scored as crack or scratch | Won on false positives and precision stability. |
| Headline class | `linear_defect` at evaluation time | Crack and scratch labels disagree. Keep the 4-class head so bubble and inclusion stay hard negatives. |
| What we train on | `glass_merged_seg` first. CUBIT, FCD, SSGD, Ultralytics crack-seg, CarDD, and CrackSeg9k only as later one-at-a-time ablations | A 2025 review found pooled crack sets did not beat per-dataset training. |
| What we never train on | NCUGD, AGDD | External tests. |
| New data | Not in this revision | No author email, no GCC / Zhou download, no company-photo request in the work queue. |
| Input size | Measure 640, 1024, and 1280, and tiled vs whole-image, before another long training run | NCUGD says pixels across the crack may matter more than a new architecture. |
| Deblur GAN | Parked | Restoration metric only, until a blur ablation says otherwise. |
| SegFormer, LoRA-SAM | Parked | After the resolution, augmentation, and hard-negative results are in. |
| Compute | TIDE for training and GPU eval. Laptop for tests, profiling, and cached re-scores | Colab Stage A is finished. Do not mix its old matching numbers with TIDE rows. |
| Seeds | 3 for any retrained config that will be compared with E1 | Seed spread (±0.04 recall) is larger than the E1–E2 gap. |

---

## 4. Compute

Training and GPU evaluation run on TIDE: https://jupyterhub-research.sdsu.edu, PyTorch Notebook image, NVIDIA L40, 8 CPU, 32 GB RAM. Persistent storage is only `/home/jovyan` (75 GiB), laid out by `scripts/tide/env.sh` under `~/glass_crack/`. The server stops after 12 hours with no browser connected. `scripts/tide/job.sh` keeps one training job alive across a closed tab. Reconnect within 12 hours. Start the same job name again to resume.

```sh
source ~/glass_crack/cracked/scripts/tide/env.sh
bash "$CODE_DIR/scripts/tide/job.sh" start E4_s0 glass_crack_models.train.train_yolo \
    --config e4_yolo_robust.yaml --seed 0 --device 0 --workers "$WORKERS" --compute-units "TIDE $GPU_NAME"
```

One GPU job at a time. Re-scoring a finished run replaces its `results.csv` row when `exp_id`, `seed`, and `checkpoint_sha256` match. After a push, notebook cells 1–2 of `notebooks/tide_setup.ipynb` pull and run `setup.sh`. The Phase 1 measurement notebook is `notebooks/tide_phase1.ipynb`.

The laptop (`.venv`, CPU) is for unit tests, `compare_runs`, dataset profiles, and `predict.py` on a few images. Run Python as `env .venv/bin/python`.

---

## 5. Data already on disk

| Set | Role in this revision |
|---|---|
| `processed/glass_merged_seg` | Train 980 / valid 148 / test 148. 4 classes: bubble, crack, inclusion, scratch. Splits stay fixed. |
| `processed/glass_merged_linear` | Built from the same images by `relabel_classes`. E2 only. New runs use the 4-class set. |
| `processed/ncugd_external_test` | 19 images, 32 cracks. Score only. |
| `datasets/agdd_aircraft_glass/data` | 219 pairs, two views, box labels. Score only, per view. Also the stand-in for a two-view consistency check. |
| CUBIT-InSeg | Drone facade crack and spalling, not glass. Pretrain ablation only, after E4. Tile on the laptop first (pilot numbers in the old Section 2b notes: ~58 tiles per photo). |
| FCD-108 | Facade masks, localization only. One ablation after CUBIT, and only if CUBIT did not hurt the glass test. |
| SSGD | Glass, COCO polygons (crack, scratch, and others). Profile before any use. One ablation, not the default mix. |
| Ultralytics crack-seg, CrackSeg9k, CarDD | Generic or off-domain. CrackSeg9k is parquet and must be materialized first. Do not add them until the glass-only and CUBIT arms are scored. |
| `data/processed_bundle_v1` | Alternate single-class bundle (7,022 / 1,207). Not the training set for E4. Revisit only if the glass-core runs plateau. |

Rules:

- Thresholds and checkpoints are chosen on valid only. Test is scored once per configuration.
- Group-aware splits stay fixed.
- External sets are never used to pick a checkpoint or a threshold.
- Augmentation stays in train. Valid and test stay unaugmented.

---

## 6. Evaluation protocol (fixed)

**Metrics** (crack or scratch, scored as `linear_defect`): recall, precision, mask IoU, false positives per 100 images, and mask mAP50 for Track A. Also report threshold-free numbers from the stored sweeps: recall at precision 0.85, and precision-recall area.

**Matching rule:** a prediction matches a ground-truth instance if mask IoU ≥ 0.3, or if the masks overlap after dilating the prediction by 3 px. Also report strict IoU 0.5. One connected prediction may cover several touching ground-truth instances when its IoU with their union meets the threshold. Recall counts matched ground-truth instances. Precision counts matched predictions. Regions under 24 px² at model-input resolution are dropped before matching. Track B instances are 8-connected components of the thresholded map.

**False alarms in three slices:** all images, images with no annotations, and bubble/inclusion-only images. Valid has 67 images without linear defects (40 unannotated + 27 bubble/inclusion-only). Test has 54 (36 + 18).

**Operating point:** max F1 on valid, thresholds 0.05–0.95 step 0.05. Written to `operating_point.json`. Because that threshold moves between seeds, do not rank E1 against a new run on frozen-threshold recall alone.

**Uncertainty:** cluster bootstrap over near-duplicate groups, 2,000 resamples, 95% intervals. Three seeds for any model compared with E1. A gap smaller than the interval is not a result.

**AGDD:** never trained on. Class map: scratches (1) and crack (2) are ground truth; contusion (0) and spot (3) are confusions, not ground truth. A prediction hits a box if box IoU ≥ 0.3 or at least half of its pixels lie inside the oriented box. Quote `images/` and `image/` separately. Pair recall (found in either view) is logged and is not the headline. Crack-only AGDD has 29 boxes, so treat it as indicative. The linear-defect count is 246 boxes in 136 pairs.

---

## 7. Work order

Do these in order. Stop adding sources that do not beat E1 on the glass test.

### M0. Lock the measurements we already paid for

No new training.

1. Commit and push the Phase 1 code. On TIDE, pull and run `setup.sh`.
2. `python -m glass_crack_models.eval.compare_runs` on E1 seeds 0–2, E2 seeds 0–2, and E3. Record recall at precision 0.85 and PR area.
3. `python -m glass_crack_models.eval.agdd_box_scores` on those runs. Record in-box max score per view and look at the crack-box montages.
4. `scripts/check_ncugd_pipeline.py --backend model` on the E1, E2, and E3 checkpoints, sizes 640 1024 1280.

**Done when:** a short note says whether E1 still leads without a frozen threshold, whether `image/` is dim inside the crack boxes, and whether NCUGD is blind or only under the threshold.

### M1. Resolution, before the next training run

Use the frozen E1 seed-0 checkpoint. Same images, three inference modes:

| ID | Inference | Question |
|---|---|---|
| R0 | Whole image resized to 640 | Current reference |
| R1 | Overlapping 640 tiles, 25% overlap, as `predict.py` already does | Does tiling change test FP or NCUGD recall? |
| R2 | Same checkpoint, `imgsz` 1024 and 1280 (letterbox) | Does a larger input recover hairline cracks? |

Score test, NCUGD, and both AGDD views. No retune of the threshold on the external sets. On valid, record the threshold that maxes F1 at each input size, and also the threshold-free curve.

**Done when:** we know if extra pixels move NCUGD or only cost time. That result decides how aggressive E4's downscale augmentation should be.

### M2. E4, robustness augmentation on E1

`configs/e4_yolo_robust.yaml` already bases on E1 (4-class, `dataset: merged`) and adds scale jitter and stronger color shift. Extend that config with motion blur and mild JPEG, which Roboflow did not apply. Do not stack more Gaussian noise on the Roboflow copies.

Train seeds 0, 1, and 2 on TIDE, one job at a time. Compare to E1 with `compare_runs`, not with a single seed.

**Done when:** three seeds are in `results.csv`, and the note says whether recall at precision 0.85 and FP/100 beat E1.

### M3. Labels, then hard negatives

1. Export about 100 crack crops and 100 scratch crops from the glass core. Tag each as correct, wrong class, wrong extent, or not a defect. Fix or drop clear errors. Keep the valid and test split membership fixed. Retrain one E1-setup run on the corrected labels (E8) and compare to E1.
2. From the valid and test failure galleries, list what the false positives actually are (frame edge, reflection, sealant, dirt, bubble).
3. Build a hard-negative set from images we already have: crack-free glass in the merged set, bubble/inclusion images, and crack-free CUBIT tiles. Composite straight bright lines onto crack and non-crack images. Target about 25–35% negatives. Retrain as E4b from the E4 recipe plus these negatives.

**Done when:** we know how much of the error was label noise, and whether FP/100 falls without giving up recall at precision 0.85.

### M4. Two-stage search, only if M1 says full-frame resize loses cracks

If R1 or R2 beats R0 on NCUGD or on thin cracks, the deployment path is tiles, not a second network. Add a two-stage prototype only to save compute: a coarse pass (E1 at 640 on a downscaled frame) chooses tiles, and those tiles are segmented at the input size M1 picked. Score the same three sets. Keep it if recall holds and the tile count drops.

### M5. Two-view filter on AGDD

AGDD is the only paired data we have. As a post-process on E1 (and on E4 if M2 wins), try one rule: keep a detection only when a nearby detection exists in the other view. Report recall and false positives for each view, and the same numbers after the filter. This tests the reflection hypothesis on aircraft glass. It does not prove anything about a facade. Do not train on AGDD.

### M6. One extra source at a time

Only after M2 and M3. Pretrain on tiled CUBIT crack (drop spalling), then fine-tune on the glass core. If the glass test does not drop and NCUGD or AGDD `images/` rises, try FCD with stroke width normalized toward the glass-core width (about 5 px at 640). SSGD, Ultralytics crack-seg, CrackSeg9k, and CarDD each get a later arm only if the profile says the labels are usable. Stop when an arm does not beat the glass-only model.

---

## 8. Experiment register

| ID | What changes | Compared with |
|---|---|---|
| E1 | Done. 4-class YOLO-seg, 3 seeds | Reference |
| E2 | Done. Merged crack+scratch head | E1. E1 wins on FP. No further E2 runs. |
| E3 | Done. U-Net, 1 seed | E1. No further U-Net runs until a thin-structure loss is justified by M1. |
| R0–R2 | Inference only. Resize vs tiles vs 1024/1280 | E1 seed 0 |
| E4 | E1 recipe + scale, color, motion blur, JPEG. 3 seeds | E1, threshold-free |
| E8 | E1 recipe on audited labels | E1 |
| E4b | E4 + hard negatives from current images | E4 |
| V1 | Two-view consistency filter on AGDD | Same checkpoint, filter off |
| E5 | CUBIT pretrain, then glass fine-tune | E4 or E4b, whichever won |
| E6 | E5 + FCD, only if E5 helped | E5 |

Parked: SegFormer, LoRA-SAM, clDice as a training loss, the 2024 deblur GAN, company fine-tuning, GCC, Zhou.

---

## 9. Order of work

| Step | Where | GPU? |
|---|---|---|
| M0 measurements | TIDE notebook `tide_phase1.ipynb` | NCUGD check only |
| M1 resolution | TIDE, existing E1 checkpoint | Yes, short |
| M2 E4 × 3 seeds | TIDE `job.sh` | Yes, the long jobs |
| M3 audit and hard negatives | Laptop for the sheets, TIDE to retrain | Yes, after the audit |
| M4 two-stage | Only if M1 says tiling wins | Short |
| M5 AGDD view filter | TIDE, cached maps | No |
| M6 CUBIT arm | Laptop to tile, TIDE to train | Yes, last |

---

## 10. Deferred

Not in this revision:

- Asking authors or the company for data. The inspection pipeline still needs curtain-wall images later. That wait is not a task in the queue above.
- Agents and MCP servers. They do not change detector accuracy.
- The deblur GAN, SegFormer, and SAM, until M1–M3 say the current head and resolution are the limit.
- Drone hardware, flight approvals, and multi-view capture. M5 is the only multi-view test we can run on current files.
- Licenses (Ultralytics AGPL-3.0, SSGD academic-only, CarDD and AGDD non-commercial, Roboflow upstream rights), company agreements, and privacy. Revisit before publishing or sharing weights.

---

## 11. Risks

| Risk | What we do |
|---|---|
| Seed noise looks like a gain | 3 seeds, and recall at precision 0.85 |
| Threshold on 148 valid images jumps around | Threshold-free metrics in every comparison |
| Higher resolution still scores 0 on NCUGD | Accept it as out of domain. Do not tune the threshold on NCUGD. |
| CUBIT or FCD helps facades and hurts glass | Require the glass test to hold before keeping the arm |
| AGDD two-view filter looks good and we treat it as a facade result | Report it as aircraft glass only |
| Deblur PSNR is mistaken for crack recall | Do not add the GAN on PSNR alone |
| Roboflow stretch to 640 hides real crack width | Letterbox at inference. Do not claim a physical width. |

---

## 12. Definition of done

**This revision is done when:**

- E1, E4, and the resolution runs are in one table with threshold-free metrics and per-view AGDD.
- NCUGD is either still 0/32 at 1280, with that written down, or a higher input size has a measured recall.
- A hard-negative or label-audit run has been compared with E1 on false positives per 100 images.
- `predict.py` on the winning checkpoint is the command we would run on a new glass photo.

Curtain-wall data, when it arrives, is a new fine-tune and a new external test. It does not reopen the model choice until that test exists.
