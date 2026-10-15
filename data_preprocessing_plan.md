# Data Pre-processing Plan: Glass Crack Segmentation (v1)

Goal: raise accuracy of the crack / linear-defect model on real curtain-wall glass **before the next retrain**, using the datasets we now hold. Research scope; licensing is deferred (see Section 10 for what to record now so it is not painful later).

---

## 1. What the current results tell us (baseline diagnosis)

| Observation | What it suggests | Pre-processing response |
|---|---|---|
| E1 (4-class) and E2 (linear_defect) recall tie (0.767 vs 0.772); seed spread 0.72 to 0.82 dwarfs the gap. E1 has fewer FPs in every seed (25.0 vs 32.9 per 100) | Valid/test (148 images, ~480 crack instances total) is too small to rank configs, and the frozen threshold (0.30 to 0.50) drives much of the variance | Add data; judge changes with group k-fold CV and threshold-free metrics (recall at fixed precision, PR area), not a single split (Section 9) |
| NCUGD recall = 0.0 for every model. **Verified real, not a bug** (labels, loading, matching checked): cracks are faint 1-2 px hairlines starting at bright dots; the long dark lines are unlabeled; max score within ~7 px of any crack is 0.00 to 0.03 | Out-of-domain failure at hairline scale. The model does not respond at all, so this is not calibration | Scale/appearance work (Sections 5, 6); keep reporting NCUGD as 0/32 rather than tuning for it |
| E3 U-Net: recall 0.634, precision 0.643, FP/100 = 74.3 at threshold 0.80 | Over-predicts; too few hard negatives | Hard negatives and reflection confusers (Section 7) |
| E3's AGDD pair recall 0.833 vs E1 0.581 / E2 0.507 | Probably inflated by box-level scoring of blobby masks | Do not read it as a real E3 advantage until the view gap below is explained |
| AGDD `image/` view recall about 0.05 to 0.13 vs `images/` 0.50 to 0.85 with the same labels | Not geometry, global lighting, visibility or polarity (Section 8, step 0). A real model-robustness gap to that illumination | Report per view; consider lighting/texture augmentation. AGDD stays external (never trained on) |
| Crack vs scratch labels inconsistent | Label noise caps accuracy | Label audit (Section 4) |

Working hypothesis (updated with the TIDE results): accuracy is limited by (a) too few real crack examples, (b) scale/domain mismatch between training images and drone/facade images, (c) too few hard negatives, and (d) noisy crack/scratch labels. The plan below targets each in that order.

---

## 2. Dataset triage and roles

Rule: **only the original glass sets (and later, elicc data) are used for validation and test.** External sets are used for training or pre-training, or kept as held-out tests. Nothing external enters valid/test.

| Dataset | Images | What it is | Role | Notes |
|---|---|---|---|---|
| glass_defect_detection2 (Roboflow v2) | 578 | Glass defects, 4 classes | **Train / valid / test core** (merged) | Roboflow export stretched images to 640x640; deduplicate against glass_segmentation_2026 |
| glass_segmentation_2026 (Roboflow v10) | 1,176 | Glass defects | **Train / valid / test core** (merged) | Already merged to 980/148/148; keep that split fixed |
| ncugd_glass_crack | 26 files (19 images, 32 polygons) | External glass cracks | **Held-out test only** | Never train on it. Add a second external test later from elicc data |
| agdd_aircraft_glass | 438 (219 forward/backward pairs) | Aircraft glass, boxes, 29 crack instances | **Held-out robustness test**, optionally lighting-augmentation reference | CC BY-NC-SA; boxes only, so evaluate with box-hit rules, not mask IoU |
| cubit_inseg | 6,996 (4500x3000 / 4800x3200) | Drone facade images; class 0 crack (51,865 polygons), class 1 spalling (10,322) | **Main pre-training / extra training source for cracks** | **Not a glass dataset** (the label "Glass segmentation" in the updated list is incorrect). Needs tiling. Use crack class only; drop spalling or use as negative context |
| fcd_108_facade | 281 (270 usable, `fcd_108_grouped`) | Facade cracks, 512x512 masks | Localization pre-training, low weight | Thick-stroke masks; skeletonize then re-dilate to a consistent width (Section 5) |
| SSGD | 2,504 | Glass surface defects | Pre-training only after format check | Annotations believed to be boxes (mmdetection); no masks. Verify before use |
| CarDD | 4,000 | Car damage, COCO instance masks | Optional: only `glass shatter` / `crack` / `scratch` classes after visual review | Car-body domain; likely low value. Test as an ablation, not a default |
| Ultralytics crack-seg | 4,029 | Road/wall crack segmentation, YOLO-seg | Generic crack pre-training | Overlaps with CrackSeg9k; dedupe across them |
| CrackSeg9k | 9,159 | Aggregated crack masks | Generic crack pre-training | Mixed sources and mask thickness; sample, do not use wholesale |

**Verify first (nothing here has been inspected yet):** actual file formats, class IDs, and label quality of SSGD, CarDD, CrackSeg9k, and Ultralytics crack-seg. Run the profiling script in Section 11 before building any mixture.

**Already built (per your progress notes):** `phase_0/prepare_dataset_bundle.py` produces `data/processed_bundle_v1/` (single class `crack`, group-aware split, 7,022 train / 1,207 val / 26 NCUGD external, 0 groups leaking; AGDD excluded so it stays external). It is not yet used by the training code, which still uses `bundle_colab.zip`. Treat it as the starting point for the mixtures in Section 9 instead of rebuilding: profile it, check its crack/scratch handling against Section 4, and add the missing pieces (stroke-width normalization, hard-negative library, cross-source dedupe against valid/test/NCUGD/AGDD). Its CUBIT negatives are capped at 600 tiles, so check the negative fraction against Section 7.

---

## 3. Pipeline overview

```
raw/  ->  profile  ->  harmonize labels  ->  convert to common format
      ->  tile/rescale  ->  dedupe + leakage check  ->  QC  ->  build mixtures
      ->  manifest.csv (+ provenance)  ->  train
```

Rules that apply throughout:
1. Raw data is never modified. All outputs go to `processed/<dataset>_<version>/`.
2. Every processed image has a row in one `manifest.csv`: `image_id, source_dataset, source_path, split, group, tile_origin, label_type, license_note, sha256`.
3. **Split before tiling or augmenting.** Tiles of one source image always share its split and group.
4. Every step is a script with a version string; re-running with the same inputs gives identical outputs (fixed seeds).

---

## 4. Label harmonization

**Target schema.** Two label sets, produced from the same annotations:
- `linear_defect` (crack + scratch merged), the primary training target.
- 4-class (bubble, crack, inclusion, scratch) for the glass core only, kept for analysis.

**Steps**
1. Map each external dataset to `linear_defect` or `ignore`:
   - CUBIT crack -> `linear_defect`; CUBIT spalling -> drop (or mask as ignore region if it overlaps tiles).
   - Ultralytics / CrackSeg9k / FCD crack -> `linear_defect`.
   - CarDD: only the classes kept after review.
2. **Crack/scratch audit on the glass core.** Export ~100 random crack and ~100 random scratch instances as crops with their labels into a contact sheet. Review (and ideally have a second annotator review) and tag each as: correct, wrong class, wrong extent, not a defect. Record agreement. Fix or drop clear errors. This is the cheapest accuracy gain if label noise is the problem.
3. Remove degenerate polygons: fewer than 3 points, area below a minimum, self-intersections (repair with `buffer(0)`), and coordinates outside the image (clip).
4. Keep `ignore` regions (blurry/ambiguous cracks) as masks that are excluded from the loss instead of treating them as background.

---

## 5. Format conversion, scale, and stroke-width normalization

**Format.** Canonical store = per-image polygons (COCO-style JSON or YOLO-seg txt) plus on-demand rasterization. Convert:
- Masks -> polygons with `cv2.findContours` + simplification (epsilon about 0.5 px); keep a raster copy for the U-Net.
- Boxes (SSGD, AGDD) -> keep as boxes; never present them as masks. If used for pre-training, treat them as a separate weakly supervised stage, or generate masks with a box-prompted segmenter as an optional experiment and treat those as noisy labels.
- Reuse the existing `fix_yolo_lines` for malformed YOLO lines.

**Stroke width.** Crack masks differ in thickness by source (FCD thick strokes; CUBIT tight polygons; CrackSeg9k mixed). For a thin-structure target this inconsistency hurts.
- For thick-stroke sources (FCD, parts of CrackSeg9k): skeletonize, then dilate to a fixed width (start at 3 px at 640 px scale; tune as an ablation at 2/3/5 px).
- Check the glass-core masks' typical width distribution (skeleton-based distance transform) and match the external sets to it.
- Evaluation always uses the original, un-normalized masks.

**Aspect ratio and resizing.** Roboflow stretched images to 640x640, which distorts crack geometry. For all training and inference: letterbox (preserve aspect ratio, pad). Re-export the glass core from original images if available, otherwise note that the stretched version is the only one and keep inference consistent with it.

---

## 6. Tiling and scale matching (largest expected gain for the drone domain)

The target deployment is drone images of facades, where a crack is a few pixels wide in a 4500x3000 frame, while the glass core consists of close-up 640 px images.

**CUBIT tiling (split first, then tile)**
- Split CUBIT by source image (and by building/scene if identifiable) into train/val; no tile leakage.
- Tile at 1024 px with 25% overlap, resize to 640 (zoom factor 0.625), plus a native 640 px pass (zoom 1.0). Two scales so the model sees cracks at multiple apparent widths.
- Keep every tile that contains a crack; add ~30% crack-free tiles as negatives (Section 7). Clip polygons at tile borders; drop fragments smaller than a minimum area or length.
- Cap tiles per source image so a few large images do not dominate.

**Scale augmentation.** Random scale 0.5x to 1.5x on all sources, so the model is not tied to one crack width.

**Inference-side consistency.** `predict.py` uses the same tile size, overlap, and letterboxing as training; overlapping-tile predictions are merged by max or mean of probabilities, not by naive concatenation.

**AGDD.** Resize to match glass-core scale only for evaluation consistency; keep its forward/backward pairs together (same group) and report the recall of each lighting separately.

---

## 7. Hard negatives

False positives (FP/100 = 112.8 for the U-Net) are the second biggest accuracy problem.

1. **Crack-free glass:** every glass-core image with no `linear_defect` is retained as a negative (check how many exist; do not filter them out).
2. **Facade negatives from CUBIT:** crack-free tiles, plus spalling-only tiles (spalling is not a crack and is a natural confuser).
3. **Reflection / edge confusers:** mullions, frame edges, sealant lines, bright reflections, power lines, and window grids. Build a small library (~100 to 300 patches) from crack-free tiles and the glass core.
4. **Reflection compositing (augmentation):** alpha-blend reflection patches or gradient overlays onto crack and non-crack images with random strength, so the model learns that straight bright lines are not cracks.
5. Target negative fraction per training mixture: start at 25 to 35% of images, then tune (Section 9).

---

## 8. Deduplication, leakage control, and sanity checks

**Step 0 (done, mostly): the AGDD `image/` vs `images/` gap.** Same labels, recall 0.05 to 0.13 on `image/` vs 0.50 to 0.85 on `images/`; cached E1 peak score averages 0.19 on `image/` vs 0.60 on `images/` (219 pairs). Checked on the server (40 pairs / 29 crack boxes):
- Geometry: the views are pixel-aligned (median shift about 0.01 px, phase-correlation response 0.435). Not a label-geometry bug.
- Global lighting: brightness 186.8 vs 187.4, contrast (std) 7.0 vs 6.8. Not a global lighting difference.
- Visibility: crack boxes stand out slightly more in `image/` (edge-contrast 0.383 vs 0.255).
- Polarity: cracks are darker than their surround in 100% of boxes in both views (signed contrast -0.255 vs -0.383). Not a polarity flip.
- Conclusion: the labels are valid on both views and the cracks are at least as visible in `image/`, so this is a genuine model-robustness gap to that illumination, not a data bug. Report AGDD per view, with `images/` as the main external number; do not average the views or use E3's 0.833 as evidence of a better model. Remaining cheap check: compare the cached max score inside the 29 crack boxes per view, and look at `agdd_views_crops.png` for texture differences.

**Step 0b: NCUGD is closed as a pipeline question.** The existing checks showed labels, loading and matching are correct and the model scores 0.00 to 0.03 near the cracks. `check_ncugd_pipeline.py` can still be re-run on any new external set (or on NCUGD at 1024/1280 input) to separate "labels misplaced" from "model blind" from "threshold too high".

**Step 0c: threshold-free comparison (no GPU needed).** `evals/<run>/summary.json` already holds full precision/recall sweeps. Compare E1 vs E2 by recall at fixed precision (e.g., 0.85) or PR-curve area, averaged over seeds. For E3, check the loss weighting (BCE+Dice on tiny positive areas often needs a higher BCE weight or a focal term) and log it here.

**Near-duplicate and leakage removal**
- Compute perceptual hashes (pHash/dHash) plus a deep-feature similarity pass (e.g., a small CNN embedding) for all images.
- Remove from every **training** source any image that is a near-duplicate of a valid/test image, NCUGD, or AGDD image (Hamming distance threshold tuned by looking at examples).
- Dedupe between Ultralytics crack-seg and CrackSeg9k (they likely share source images).
- Reuse `near_duplicate_pairs` and the union-find grouping; extend the `group` column to the new sources.
- Log counts removed per source in `dedupe_report.csv`.

**QC checks (automated; fail the build on violation)**
- Image/mask size mismatch; unreadable files; empty or all-ones masks; mask area outside the expected range.
- Polygons outside the image; empty label files for images that should have cracks.
- Duplicate `image_id`; split contamination (a group in more than one split).
- Class distribution per split and per source, written to `qc_summary.json`.
- Visual check: a random contact sheet of 50 images with overlays per source, reviewed by a person before each mixture is built.

---

## 9. Training mixtures and ablations

Because E1/E2 cannot be separated on one split, compare configurations with **group k-fold CV (k = 5) on the glass core** and 3 seeds for the finalists. Additional data is added only to the training folds.

**Stage structure**
1. *Pre-train* (optional, on external sets): CUBIT crack tiles + sampled Ultralytics/CrackSeg9k + FCD (normalized stroke width).
2. *Fine-tune* on the glass core (train folds), with negatives and augmentation.
3. Evaluate on valid (threshold selection), test, NCUGD, and AGDD.

**Ablation order (one change at a time; stop adding sources that do not help)**

| ID | Change | Question |
|---|---|---|
| P0 | Glass core only, current preprocessing (baseline, re-run under CV) | Reference |
| P1 | P0 + corrected labels (Section 4) | How much is label noise? |
| P2 | P1 + hard negatives and reflection compositing | Do FPs fall without losing recall? |
| P3 | P2 + scale augmentation | Does NCUGD/AGDD recall improve? |
| P4 | P3 + CUBIT crack tiles as pre-training | Does the drone domain help? |
| P5 | P4 + FCD (normalized width) | Is it worth the complexity? |
| P6 | P4 + Ultralytics/CrackSeg9k sample (e.g., 2,000 images) | Do generic cracks help or hurt? |
| P7 | Best of P4-P6 + SSGD / CarDD | Only if profiling shows usable labels |

**Sampling weights.** Glass core weight 1.0; external sets down-weighted (start 0.1 to 0.3 per sample) or used only in pre-training. Tune on valid; never on test, NCUGD, or AGDD.

**Metrics reported for every row** (same protocol as before): mask IoU / Dice, instance recall and precision under the matching rule (IoU >= 0.3 or overlap after 3 px dilation; also strict 0.5), FP/100 images, clDice (thin-structure), per-source breakdown, NCUGD and AGDD recall, bootstrap CIs over near-duplicate groups.

**Augmentation policy (all stages)**
- Geometric: flips, rotations (0 to 360 for cracks, since orientation is arbitrary), small perspective, scale 0.5 to 1.5.
- Photometric: brightness/contrast, gamma, color jitter, simulated glare/gradients, JPEG compression, motion/defocus blur (drone shake).
- Avoid: aggressive mosaic or cut-and-paste that creates cracks crossing artificial boundaries, and elastic distortion that bends cracks into unrealistic shapes. Test mosaic on/off if using YOLO.

---

## 10. Provenance (record now, resolve later)

- Keep `source_url`, `license_note`, and `sha256` per file in `manifest.csv` (the downloader scripts already record these).
- Do not redistribute Roboflow images; models trained on them can be described in the paper with the dataset citation.
- SSGD (academic only), CarDD (non-commercial + Flickr/Shutterstock terms), AGDD (CC BY-NC-SA): note them in the table of datasets used. Confirm Roboflow upstream rights before publication.
- When elicc imagery arrives, store it separately with its own manifest and keep it out of any shared Drive folder until agreements are in place.

---

## 11. Scripts to build (in order)

| Script | Purpose |
|---|---|
| `profile_datasets.py` | Per dataset: file counts, formats, image sizes, label types and class IDs, mask-width stats, empty-label rate; writes `profile/<dataset>.json` |
| `check_ncugd_pipeline.py` | Overlays GT on NCUGD as the evaluator loads it; runs predictions at multiple sizes and saves probability maps |
| `harmonize_labels.py` | Class mapping and polygon cleanup to the common format |
| `label_audit_sheet.py` | Contact sheets for the crack/scratch audit |
| `normalize_stroke.py` | Skeletonize + re-dilate for thick-stroke masks |
| `tile_cubit.py` | Split-aware tiling with negatives and polygon clipping |
| `dedupe_leakage.py` | pHash + embedding dedupe and leakage report |
| `build_mixture.py` | Builds a training mixture from a config (sources, weights, negative fraction) and writes the manifest |
| `qc_checks.py` | Automated checks and contact sheets |

All scripts should run from a config file, work on local disk or Drive archives (for Colab), and write outputs that can be zipped for the cluster later.

---

## 12. Order of work and effort

1. **Day 1:** Step 0 (AGDD view gap), Step 0c (threshold-free E1 vs E2), dataset profiling of every source. Cheapest, and makes the later comparisons trustworthy.
2. **Days 2 to 3:** label audit, label harmonization, dedupe/leakage on all sources.
3. **Days 3 to 5:** CUBIT tiling and mixture builder; QC and contact-sheet review.
4. **Week 2:** P0 to P3 under group k-fold (fast, glass core only).
5. **Week 2 to 3:** P4 to P7 (needs more GPU time; Colab now, cluster/DGX when available).
6. Freeze the best mixture, retrain final models with 3 seeds, and report with CIs.

---

## 13. Risks and open items

- **Domain gap may remain** even after tiling (close-up glass vs. drone facade concrete cracks). If P4 does not help NCUGD/AGDD, the fix is elicc drone imagery, not more public data.
- **Label noise in external sets** (CrackSeg9k mixture, SSGD boxes) may add as much error as signal; hence the audit, sampling, and low weights.
- **Overfitting to a tiny valid/test set**: keep test untouched until the final run; use CV for decisions.
- **Compute**: CUBIT tiling is CPU/disk heavy; do it once locally (16 cores) and upload the tiled result, not the 4500x3000 originals, to Colab.
- **Data to confirm with you:** whether original-resolution (non-stretched) images exist for the Roboflow sets; whether a second annotator is available for the crack/scratch audit; whether NCUGD's 19 images should ever be split for fine-tuning (default: no).
