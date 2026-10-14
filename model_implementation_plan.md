# Glass Crack Segmentation: Model Implementation Plan (v2, usable-model-first)

Date: 2026-10-13
Scope: build a working crack-segmentation model on the existing bundle (`temp/glass_crack_dataset_bundle/`), get it running on real photos, then improve it. This is research work with an industry partner (curtain-wall glass company). Licensing, agreements and permissions are deliberately deferred; see Section 9.

---

## 1. Goal

**A usable model first:** a command-line tool that takes a photo of any size (phone shot, drone frame, pane close-up) and returns crack masks, an overlay image, and a JSON of detections, with a trained checkpoint behind it. Everything else (pretraining, label audit, fusion) improves that tool.

"Usable" means:
- It runs on full-resolution photos through overlapping-tile inference, not just 640 px crops.
- It has a chosen operating threshold and a stated false-alarm rate on held-out data.
- You can hand it photos from the company and see sensible output the same day.
- It saves per-pixel probability maps, so the later multi-view fusion step can build on it.

**Not in scope yet:** physical crack width, multi-view fusion itself, claims about field performance.

---

## 2. Decisions (defaults, change freely)

| Decision | Default | Reason |
|---|---|---|
| Task type | Run **both** YOLO-seg (Track A) and U-Net/SegFormer (Track B); ship whichever wins on valid | Format is ready for A; B is typically better on thin cracks. |
| Classes | Single `linear_defect` (crack + scratch) as headline; keep the 4-class run as a reference | Crack/scratch labels are inconsistent. |
| Bubble / inclusion under `linear_defect` | Remove their annotations (pixels become background) and **keep the images** as hard negatives | 379 train / 27 valid / 18 test images contain only bubbles or inclusions, and none also contain a crack or scratch. Keeping them teaches the model that round and point defects are not linear defects, which should raise precision; dropping them would discard 39% of train images. |
| Pretraining | After a working baseline exists (CUBIT tiles, optional FCD) | Gives something to compare against and ship early. |
| Training location | **Stage A: Google Colab (student plan)** for training, with Kaggle as a backup; **laptop (CPU only)** for code, tests, data prep and small inference runs. **Stage B: SDSU GPU cluster / DGX** once access is granted (Section 2a). | Colab upload is about 200 MB. CUBIT is tiled on the laptop, then uploaded (Section 2b). |
| Real-world data | Ask the company for pane photos now; send the GCC-385 request too | Biggest gap in the bundle. |
| Tools | Use whatever works best (Ultralytics, smp, transformers) | Speed to a usable model. |

---

## 2a. Compute staging

Access to the SDSU cluster / DGX has been requested and may take a while. The plan therefore runs in two stages, and nothing in Stage A should block on the answer.

| | Stage A (now) | Stage B (when cluster access arrives) |
|---|---|---|
| Where | Colab Pro for training; laptop (16 CPU cores, 30 GB RAM, no CUDA GPU) for code, unit tests, CUBIT tiling and small `predict.py` checks; Kaggle as overflow | SDSU cluster / DGX |
| What | Phases 0-3 and E1-E3: environment, support code, baselines, first `predict.py`; a small CUBIT tiling pilot | Full CUBIT pretraining (E5-E6), multi-seed reruns, SegFormer and other larger models, ablations, company-data fine-tuning (E7) |
| Seeds | 1 per config, then 3 for the top two | 3+ for everything that is compared |
| CUBIT budget | Pilot only (a few thousand tiles) to measure speed and storage | Large tile budget; try more than one tile size |

**Make Stage A portable so the move to Stage B is quick**
- All paths come from environment variables (`DATA_ROOT`, `OUT_ROOT`), never hard-coded Drive paths.
- Everything is driven by a config file and a script (`python train_*.py --config ...`), and every run is resumable.
- Keep package pins in `requirements.md` (uv commands for the laptop, a pip line for Colab). If the DGX uses containers, a short Dockerfile based on the same pins is enough. A Slurm job template can be written once you know the scheduler.
- Results append to `results/results.csv` in the same format on both stages, so runs from Colab and the cluster land in one table.

**What not to do on Colab**
- Do not spend compute units on full CUBIT pretraining, large hyperparameter sweeps or the bigger models. Hold these for Stage B.
- Do not tune hard on the baselines. A reasonable, reproducible reference is the Stage A goal.

**If the cluster has not answered after about two weeks**
- Follow up by email, and ask your advisor to contact the system owner directly.
- Start the NSF ACCESS Explore request in parallel (advisor letter of collaboration needed).
- Decide then whether to run a reduced CUBIT pretraining on Colab or Kaggle (capped tile subset, split across sessions with resumable checkpoints).

---

## 2b. Colab workflow

I don't know exactly what your student plan includes (GPU types, compute units, maximum runtime). Check this in Colab before planning run lengths, and adjust the budget below.

**Setup that survives disconnects**
- Keep all code in a git repo (GitHub). Each session starts with a short setup notebook: mount Drive, clone or pull the repo, run the Colab pip line from `requirements.md` (Colab's own `torch` is kept), then copy the data from Drive to local disk.
- Reading thousands of small files straight from Drive is slow. Store each dataset as a single archive on Drive, copy it to `/content`, and unzip it there every session.
- Keep the notebook thin. Training and evaluation logic live in repo scripts (`python train_yolo.py --config configs/e2.yaml`), so a run is reproducible from a config and not from notebook state.
- Write checkpoints, logs and `results/results.csv` to Drive, not to the VM. The VM disk is wiped when the session ends.

**Plan for the session to die**
- Make every training script resumable: Ultralytics `resume=True`, and for Track B save model, optimizer, scheduler and epoch each epoch. Append result rows to the CSV on Drive as soon as an evaluation finishes.
- Keep each run short enough to finish in one session (a few hours at most), and run one experiment per session.
- Check whether your plan supports background execution. If not, keep the tab open during long runs.
- Use experiment tracking (W&B or MLflow) so loss curves survive a disconnect.

**Budget the compute**
- Use mixed precision (AMP) and small models (a small YOLO-seg checkpoint, a ResNet34 U-Net) at 640 px.
- Run 1 seed per config first. Rerun only the top two configs with 3 seeds.
- Track compute-unit use per run in the results table, so you know how many experiments you can still afford.
- Kaggle's weekly free GPU quota is a backup. The same scripts work there with only the paths changed.

**CUBIT: tile locally, not on Colab**
- CUBIT is 37 GB extracted (18 GB of zips), which almost certainly won't fit in your Drive storage, and Colab VMs have only a few CPU cores, so tiling there would be slow.
- Tile on your own 16-core machine, where the data already sits and you have about 530 GB free. Keep a subset (for example 40-60k tiles at 640 px).
- **Pilot, measured 2026-10-13** (`python -m glass_crack_models.data.tile_cubit --pilot 500`, 500 random CR1 train photos, 14 workers): 30 s wall time (0.83 s per photo per worker); 28,823 tiles = 57.6 per photo (23,184 with cracks, 5,639 kept negatives after dropping 70%); 43.6 KB per JPEG tile (quality 90); 1.26 GB in total. The native 640 px pass makes up two-thirds of the tiles (19,457 vs 9,366 from the 1024 px pass). 2,688 clipped fragments under 24 px² were dropped and 165 fragment-only tiles skipped.
- **Extrapolated to all 3,154 CR1 photos** (train/val/test): about 182k tiles, 7.9 GB, about 3 minutes of tiling. That is too much for Colab (well over an hour per epoch). For Stage A, use the 1024 px pass only (`--windows 1024`: about 59k tiles, about 2.6 GB) or cap the native pass; keep the full set for Stage B.
- Pack the tiles into a few archives, upload them to Drive, and copy them to `/content` each session like any other dataset.

**What goes to Drive (about 200 MB)**
- Only `processed/` and AGDD's `data/` folder are needed for Stage A. The raw Roboflow folders are already merged into `processed/`, and AGDD's `.git` folder (123 MB) is never uploaded:
  ```sh
  cd temp/glass_crack_dataset_bundle
  zip -rq ../bundle_colab.zip processed datasets/agdd_aircraft_glass/data -x '*/.git/*' \
    'processed/glass_merged_linear/*' '*.cache' 'processed/*/data_*.yaml' 'processed/*/train_*.txt'
  ```
  The excluded files are rebuilt in each session by `python -m glass_crack_models.data.relabel_classes` (they hold machine-specific paths). Built 2026-10-13: 185 MB. Session steps are in `notebooks/colab_setup.ipynb`.

**Inference on large photos**
- `predict.py` should also run on the laptop CPU for small batches, so you can try new company photos without spending Colab compute. The CPU build of PyTorch and all model packages are already installed in `.venv` (see `requirements.md`). Tiled inference on 16 cores is slow but fine for a few images.

---

## 3. Data roles

| Set | Path | Role |
|---|---|---|
| Merged glass seg | `processed/glass_merged_seg/` | Train / valid / test (980 / 148 / 148 images; group-aware splits) |
| NCUGD | `processed/ncugd_external_test/` | External test (19 images, 32 crack polygons) |
| AGDD | `datasets/agdd_aircraft_glass/data/` | External box-level test (mapping in Section 4a); optional lighting experiment |
| CUBIT-InSeg | `datasets/cubit_inseg/` | Pretraining (needs tiling) |
| FCD-108 | `processed/fcd_108_grouped/` | Optional pretraining (localization only) |
| Company photos | to be requested | Qualitative check first, then fine-tuning and test data |

Rules that keep results honest, whatever happens with permissions:
- Choose thresholds and checkpoints on valid only; evaluate on test once per configuration.
- Keep the existing group-aware splits fixed across experiments.
- External sets are never used for model selection.
- Every file keeps its source in a `manifest.csv`, extended to any new data (company photos included), so provenance can be reconstructed later.

---

## 4. Evaluation protocol (lightweight but fixed)

**Metrics** (crack / `linear_defect`): recall, precision, mask IoU, false positives per 100 images, and mask mAP50 for Track A.

**Matching rule (fixed in advance):** a predicted instance matches a GT instance if mask IoU >= 0.3, or if the masks overlap after dilating both by 3 px. Also report strict IoU 0.5. Thin cracks score low on pixel IoU even when correct. For Track B, derive instances from 8-connected components of the thresholded probability map.

**Minimum region size (fixed): 24 px², measured at model-input resolution (a 640 px image or tile).** Predicted regions smaller than this are discarded before matching, for both tracks. Measured on the 1,600 crack and scratch polygons in `glass_merged_seg`: median area 608 px² (median length 103 px, median width about 5 px), 1st percentile 26 px², and only 0.6% are below 25 px². So 24 px² removes specks and JPEG-noise blobs while giving up almost no real defects. In `predict.py` the filter is applied per tile at model-input scale, before stitching, so it means the same thing on a 640 px crop and a full-resolution drone frame. It is not re-tuned per experiment.

**False alarms are reported in three slices**, because the negatives differ in kind: FP/100 over all images (headline), over images with no annotations at all, and over bubble/inclusion-only images (Section 2). Valid has 67 images without linear defects (40 unannotated + 27 bubble/inclusion-only); test has 54 (36 + 18).

**Operating point:** pick the threshold on valid, freeze it, then evaluate on test and the external sets. **Decided: max F1 under the lenient matching rule**, over thresholds 0.05-0.95 in steps of 0.05 (`run_eval.py` writes it to `operating_point.json` next to the checkpoint, and `predict.py` reads it from there). Revisit once the company states a required recall or false-alarm rate (Phase 6).

**Uncertainty:** cluster bootstrap over near-duplicate groups (`group` column in `manifest.csv`), 2,000 resamples, 95% CIs; at least 3 seeds for the models you compare seriously. Early exploration can use 1 seed. With about 480 crack instances in total, treat differences smaller than the CI as unestablished.

**Results table:** one row per experiment in `results/results.csv`:
`exp_id | task | classes | pretrain | seed | recall [CI] | precision [CI] | mask IoU [CI] | FP/100 [CI] | NCUGD recall | AGDD box recall`

## 4a. AGDD mapping (external box-level test)

AGDD is never trained on, so all 219 image pairs (its own train and val) are test data. Images are 640x640 and are fed without resizing.

**Classes.** AGDD ids: `0 contusion, 1 scratches, 2 crack, 3 spot`.

| AGDD class | `linear_defect` models | 4-class reference (E1) | Role in scoring |
|---|---|---|---|
| `crack` (2) | `linear_defect` | `crack` | Ground truth |
| `scratches` (1) | `linear_defect` | `scratch` | Ground truth |
| `contusion` (0) | not mapped | not mapped | Not ground truth; predictions landing on these boxes are false positives, also counted separately as "non-linear confusions" |
| `spot` (3) | not mapped | not mapped | Same as `contusion` |

That gives 246 `linear_defect` boxes in 136 pairs. The other 83 pairs contain only contusions and spots (353 boxes) and serve as negatives; AGDD has no defect-free images. For E1, only its `crack` and `scratch` predictions are scored; its `bubble` and `inclusion` predictions are ignored on AGDD. Crack alone has 29 boxes, so crack-only numbers are indicative; the headline is `linear_defect`.

**Geometry.** Ground truth comes from `labels_rect/` (axis-aligned boxes). `labels/` holds the same objects in the same order as oriented boxes and is used only for the "hit" rule below. Each predicted instance (a Track A mask, or a Track B connected component after the 24 px² filter) becomes its axis-aligned bounding box.

**Matching.** A prediction matches a GT box if box IoU >= 0.3, **or** if at least 50% of the predicted region's pixels fall inside the GT oriented box. The second rule exists because a thin diagonal crack fills little of its box. Matching is greedy and one-to-one by descending score. Also report strict box IoU >= 0.5.

**Two illumination views.** `data/images/` and `data/image/` hold the two lighting setups for the same scene, with identical filenames and one shared label file. The repo does not say which folder is forward and which is backward lighting, so results are reported per folder (two rows), plus **pair recall**: a GT box counts as found if it is matched in either view. Pair recall is an early, cheap preview of the multi-view idea in Phase 7.

**Operating point.** Use the threshold frozen on merged valid, and also report the precision-recall curve, since thresholds will not transfer across domains.

---

## 5. Phases

### Phase 0: Environment (Day 1-2)
- Set up the Colab workflow in Section 2b, and check which GPU and runtime limits your plan gives you. Package pins are in `requirements.md`; the laptop `.venv` is already installed with them (CPU PyTorch). On Colab, keep its preinstalled `torch` and install the rest with the pip line in `requirements.md`.
- Upload `bundle_colab.zip` (about 200 MB; `processed/` plus AGDD `data/`, no `.git`; command in Section 2b) to Drive.
- Verify `glass_merged_seg/data.yaml` loads, and render GT overlays on ~20 images per split, including empty-label ones.
- Pin versions; log seed, git commit and dataset hashes (`files_sha256.csv`) per run.
- **Done when:** overlays look correct and a 1-epoch smoke run finishes.

### Phase 1: Minimal support code (Day 2-4)
Only what is needed to train, evaluate and predict:

```
glass_crack_models/
  paths.py, protocol.py     # DATA_ROOT/OUT_ROOT; every fixed evaluation constant (Sections 4, 4a)
  data/relabel_classes.py   # 4-class -> linear_defect; drops bubble/inclusion rows, keeps the images;
                            # writes data_{all,balanced,no_augpairs}.yaml weighting variants
  data/rasterize.py         # YOLO polygons <-> masks (Track B)
  data/tile_cubit.py        # Phase 4 CUBIT tiling (+ --pilot)
  train/train_yolo.py, train_semseg.py, semseg_common.py
  eval/regions.py, match.py, metrics.py, bootstrap.py, agdd.py, run_eval.py
  infer/predictors.py, tiling.py, predict.py
  viz/overlay.py
  configs/                  # e1, e2, e3 (+ tversky, cldice, segformer), e4 starting points
  tests/
notebooks/colab_setup.ipynb
OUT_ROOT/{runs, evals, results/results.csv, overlays}
```

Reuse `fix_yolo_lines`, the union-find grouping and the `manifest.csv` format from `clean_datasets.py`. Keep the evaluator model-agnostic (predicted instance masks plus scores in, metrics out), with the 24 px² minimum region size (Section 4) and the AGDD mapping (Section 4a) as fixed constants in one config. Unit-test `match.py` and `bootstrap.py` on toy cases.

### Phase 2: Baselines (Week 1-2)
- **E1:** YOLO-seg (small checkpoint: `yolo26s-seg.pt`; `yolo11s-seg.pt` is a drop-in alternative), 4 classes, `imgsz=640`, COCO-pretrained.
- **Source weighting (set once, here):** `balanced`. Each Roboflow pair copy is sampled at half weight (YOLO: single-copy images listed twice in `train_balanced.txt`; U-Net: `WeightedRandomSampler`). Kept fixed for all comparisons; `no_augpairs` is the Phase 4 ablation.
- **E2:** YOLO-seg, `linear_defect`.
- **E3:** U-Net (ResNet34 or EfficientNet encoder) or SegFormer-B2, `linear_defect`, BCE + Dice, with a Tversky/clDice ablation for thin structures.
- 1 seed to start; rerun the top two with 3 seeds.
- **Output:** E1/E2/E3 on valid and test; failure galleries (top 20 false negatives and positives).
- **Done when:** you have a reference checkpoint and a decision on which track continues.

### Phase 3: Make it usable (Week 2-3). Main milestone
Build `predict.py` around the best Phase 2 checkpoint:
- **Tile-aware inference:** sliding window with overlap (for example, 640 px tiles, 25% overlap), probability blending, and an optional multi-scale pass, since drone-distance cracks are far smaller in pixels than close-up cracks.
- **Aspect-preserving resize:** the training data was stretched to 640x640, so avoid stretching real photos.
- **Outputs per image:** overlay PNG, binary mask, float16 probability map, and a JSON of instances (polygon, score, area, length in pixels), plus a sidecar recording model, checkpoint and threshold.
- **Batch mode** over a folder, and ONNX or TorchScript export if deployment is wanted later.
- **Try it on real photos immediately:** NCUGD, AGDD, any facade or glass photos you can take on campus, and company photos once you have them. Review the output by eye and write down failure modes (reflections, mullions, frame edges, scratches, dirt).
- **Done when:** a new photo goes in and a sensible overlay comes out in one command, with the false-alarm rate on valid stated.

### Phase 4: Improve robustness (Week 3-6)
Ordered by expected payoff; each step is compared against the Phase 3 model on identical splits.

**Pre-augmented copies already in train.** 582 of the 980 train images (59%) are Roboflow augmented pairs: 2 copies each of 291 source photos from `glass_segmentation_2026`, marked `copy_kind = original_or_augmented` in `manifest.csv`. One copy of each pair is probably the original, but which one cannot be told. The augmented copy already has: horizontal/vertical flips, a random 90° rotation, ±10° rotation, ±10° shear, 0-5% crop, ±15% brightness, ±10% exposure, up to 1 px Gaussian blur and 0.5% salt-and-pepper noise. Implications for this phase:
- **Weight by source, not by file.** Give each copy in a pair half weight (or sample one copy per pair per epoch), so these 291 photos do not count double. Make this a config switch, set it once in Phase 2, and keep it fixed across all Phase 4 comparisons.
- **Do not stack the same augmentations.** Online flips and 90° rotations are harmless, but extra blur and noise on top of Roboflow's would push these copies further from real photos. Prefer augmentations Roboflow did not apply (step 1 below).
- **Ablation:** run E4 once with the 291 augmented-pair sources removed from train (keeps 398 single-copy train images), to see whether the pre-augmented copies help or hurt.
- Valid and test contain no augmented copies; keep it that way for any data added later.

1. **Scale and appearance augmentation** of the kinds Roboflow did not apply: scale jitter that downsizes cracks toward drone-distance pixel sizes, JPEG compression, motion blur, and colour shifts beyond ±15%, so close-up training photos better match drone imagery. Keep extra Gaussian blur and noise mild on the pre-augmented copies (see above).
2. **Reflection hard negatives:** composite reflection-like clutter (wires, branches, frames, blinds) onto crack-free glass, and add real crack-free reflective photos from the company when available.
3. **CUBIT pretraining:**
   - Tile images (4500x3000 or 4800x3200) into 1024 px tiles with 25% overlap, resized to 640, plus a native-resolution 640 pass; clip polygons, drop fragments under 24 px² at the 640 px training scale (same rule as Section 4), simplify (about 0.5 px tolerance).
   - Keep all crack tiles plus about 30% negative tiles; skip spalling at first.
   - Tile after splitting; keep CUBIT's original train/valid/test split. Run tiling on the laptop (16 cores) and upload the tile archives to Drive (Section 2b).
   - Pretrain, then fine-tune on merged `linear_defect`. Optionally add FCD as an extra stage (localization only).
   - Watch for negative transfer: if gains appear only on NCUGD and not on the merged test, do not assume it helps.
4. **Company-photo fine-tuning:** as soon as you have labeled pane photos, fine-tune and test building-disjoint (see Phase 6). Expect this to matter more than any of the steps above.

### Phase 5: Deeper evaluation (parallel, lower priority)
- **External tests:** NCUGD (segmentation) and AGDD (box-level crack/scratch; 29 crack instances, indicative only). Report frozen-threshold numbers plus precision-recall curves, since thresholds will not transfer across domains.
- **Label audit:** audit crack versus scratch on valid and test (about 296 images), prioritizing images where the model confidently disagrees with the label. A second annotator on a ~20% sample (about 60 images) would give agreement (Cohen's kappa, mask IoU); a labmate, an undergraduate assistant, or someone at the company would do. Re-score saved checkpoints on corrected labels without retraining to see how much label noise was costing.
- **Thin-structure metric:** add clDice or a skeleton-based score alongside IoU.
- **Group k-fold CV** for the key comparison between the baseline and the best improved model.

### Phase 6: Company data and capture (start Week 1, in parallel)
This is the research plan's top priority and the biggest gap in the bundle. Ask the company for:
- Pane photo bursts (several views per pane, varied angle and distance).
- Known pane dimensions (gives pixel scale from a reference), or a reference target in frame.
- Cracked panes with known crack locations, and crack-free panes with strong reflections for hard negatives.
- Pane and building identifiers, so the split can be **building-disjoint**.
- Glass type (tempered, laminated, coated, low-e) and what they care about most (factory, installation, in-service).
- What the model has to deliver (for example, recall at a fixed false-alarm rate), which sets the operating point.

Annotation spec: polygon cracks, one `crack` class, a "reflection" tag on negatives. If a labeled pilot of 5-10 panes across 2-3 buildings is possible, use it first to test the pipeline. Keep sending the GCC-385 / Zhou dataset requests. Drone flights need FAA, SDSU and building-owner approval; start that early because it can take weeks.

### Phase 7: Multi-view fusion interface (Week 6+)
Phase 3 already saves probability maps, masks and scores per frame. For the fusion idea (keep cracks that hold position across views, reject reflections that move), a synthetic prototype is the cheapest first test: composite a reflection layer that shifts differently from the crack across simulated views, and check the fusion logic before real bursts arrive.

---

## 5b. Colab-first checklist (first two weeks)

**Days 1-2: set up**
- [ ] In Colab, check that Pro is active and when it expires; note the GPU type (Runtime menu) and your compute-unit balance.
- [ ] Create the GitHub repo and the layout from Phase 1. Create a Drive folder for datasets, checkpoints and results.
- [x] Build `bundle_colab.zip` with the command in Section 2b (built: `temp/bundle_colab.zip`, 185 MB). [ ] Upload it to Drive.
- [x] Write the setup notebook: `notebooks/colab_setup.ipynb` (mount Drive, clone, pip line, copy and unzip, relabel, one training cell per experiment).
- [x] Verify `glass_merged_seg/data.yaml` loads; render GT overlays for about 20 images per split (`outputs/overlays/`); run a 1-epoch smoke test (YOLO and U-Net on the laptop CPU). [ ] Repeat the smoke test on Colab GPU.

**Days 2-4: support code (write and test on the laptop, not on Colab; `.venv` already has everything in `requirements.md`)**
- [x] `relabel_classes.py` (4-class to `linear_defect`) and `rasterize.py` (polygons to masks, with a round-trip check).
- [x] `match.py`, `metrics.py`, `bootstrap.py`, `run_eval.py`, with unit tests on toy cases (23 tests in `glass_crack_models/tests/`).
- [x] Training scripts with config files and resume support (`train_yolo.py`, `train_semseg.py`, `configs/`).

**Days 4-7: first baselines (1 seed each, log to `results.csv`)**
- [ ] E2: YOLO-seg, `linear_defect`.
- [ ] E1: YOLO-seg, 4-class.
- [ ] E3: U-Net (ResNet34 encoder), `linear_defect`, BCE + Dice.

**Week 2: pick a reference and make it usable**
- [ ] Choose the operating threshold on valid; evaluate once on test; rerun the top two configs with 3 seeds.
- [ ] Write failure galleries (top 20 false negatives and false positives per model). Generated automatically by `run_eval.py` on valid; review them per model.
- [x] Start `predict.py` (tiled multi-scale inference, overlay, mask, probability map, JSON, `run.json` sidecar, folder batch mode, TorchScript/ONNX export). Tested on a 4500x3000 CUBIT photo (about 30 s on the laptop CPU) and on NCUGD. [ ] Run it with a trained checkpoint on NCUGD, AGDD and real glass photos.
- [x] Optional pilot: tile 500 CUBIT images locally and measure tile size, count and storage (numbers in Section 2b).

**In parallel, from day 1**
- [ ] Follow up on the cluster / DGX request.
- [ ] Ask the company for pane photos (Phase 6) and send the GCC-385 / Zhou dataset requests.
- [ ] Ask your advisor about an ACCESS Explore letter of collaboration as a backup.
- [ ] Decide who could be a second annotator (Phase 5).

**Stage A is done when:** a reference checkpoint exists, `predict.py` runs on a full-size photo and produces sensible output, the results table has E1-E3 with CIs, and you have written down the CUBIT pilot numbers.

---

## 6. Experiment register

| ID | Track | Classes | Pretraining | Purpose |
|---|---|---|---|---|
| E1 | A (YOLO-seg) | 4-class | COCO | Reference |
| E2 | A | linear_defect | COCO | Effect of merging crack and scratch |
| E3 | B (U-Net/SegFormer) | linear_defect | ImageNet | Semantic vs. instance; thin-structure loss |
| E4 | best of A/B | linear_defect | + scale/reflection augmentation | Robustness |
| E5 | best of A/B | linear_defect | CUBIT tiles | Does facade pretraining help? |
| E6 | best of A/B | linear_defect | CUBIT then FCD | Does FCD add anything? |
| E7 | best | linear_defect | + company data | Fine-tuned model |
| E8 | best | linear_defect | corrected labels | Retrain after audit |

---

## 7. Timeline (rough; Phases 0-3 are the committed scope)

| Week | Work |
|---|---|
| 1 | Phase 0-1; contact company about data; send GCC-385 / Zhou requests; start flight approvals |
| 2 | Phase 2 baselines (E1-E3) |
| 3 | Phase 3: `predict.py`, first run on real photos |
| 4-6 | Phase 4: augmentation, hard negatives, CUBIT, first company data |
| 5-6+ | Phase 5 evaluation depth; Phase 7 prototype |

With coursework and waiting on data and compute, treat Phase 4 and beyond as stretch.

---

## 8. Risks

| Risk | Mitigation |
|---|---|
| Domain gap (shards and phone close-ups vs. drone facades) | Phase 3 real-photo check; scale augmentation; company data |
| Small eval sets (~90 cracks per split) | Cluster bootstrap, seeds, group CV; claim only differences larger than the CI |
| Crack/scratch label inconsistency | `linear_defect` headline; Phase 5 audit |
| Reflections trigger false alarms | Reflection hard negatives; track FP/100 on crack-free images |
| Negative transfer from CUBIT | Pilot epoch; compare to baseline before scaling |
| Threshold does not carry across domains | PR curves; re-pick the threshold on company validation data |
| Colab sessions disconnect and compute units run out | Resumable scripts, checkpoints and results on Drive, 1 seed first, track units per run (Section 2b); Kaggle as backup |
| Timeline slips (data, compute, approvals) | Phases 0-3 are self-contained |

---

## 9. Deferred items (not blocking)

Intentionally postponed so they do not slow down getting a model working. Revisit before sharing the model or data outside the project, publishing, or deploying:
- Licenses and usage terms: AGDD is non-commercial; upstream image rights for the Roboflow sets are unverified; Ultralytics is AGPL; check the pretrained SegFormer weights' terms.
- Agreements with the company and SDSU: IP ownership of models and captured data, publication review, NDA scope.
- Privacy: blur or drop frames showing people or interiors before sharing captured data.
- Framing: describe the model as a screening aid, not a structural-safety judgment.

Provenance is already tracked via `manifest.csv` and dataset hashes, so this can be sorted out later without redoing any experiments.

---

## 10. Definition of done

**Milestone 1 (usable model):** `predict.py` runs on full-resolution photos and produces overlays, masks, probability maps and JSON; the chosen checkpoint has a stated threshold and FP/100 with CIs on valid and test; a first qualitative pass on real photos with written failure modes.

**Milestone 2 (improved model):** a results table (Section 4) for E1-E6 with CIs, showing what augmentation, hard negatives and CUBIT pretraining each add.

**Milestone 3 (company model):** a fine-tuned model evaluated on building-disjoint company data.

All runs are reproducible from a config file, a seed and a dataset hash.
