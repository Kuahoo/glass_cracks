# Dataset profile summary

Snapshot from `scripts/profile_datasets.py` on 2026-10-15 (laptop `data/` + `temp/glass_crack_dataset_bundle/processed/`). Full per-dataset JSON lives under `outputs/profile/` (git-ignored). Re-run:

```sh
env .venv/bin/python scripts/profile_datasets.py --children data temp/glass_crack_dataset_bundle/processed --out outputs/profile
```

**Takeaways for Phase 3 mixtures**
- **SSGD** has real COCO polygons (flat `[x1,y1,...]` lists), not boxes. Crack / scratch / broken are usable; crack stroke ~26-35 px at 640 scale (thicker than the glass core ~5-6 px).
- **CrackSeg9k** is parquet only (`train.parquet`, `test.parquet`); materialize with `phase_0/prepare_dataset_bundle.py`. The file profiler correctly reports 0 standalone images.
- **CUBIT** is drone facade crack/spalling (classes 0/1), not glass. Tile before use.
- **CarDD** PNG masks are thick brush strokes (~256 px); localization only.
- **glass core** stroke widths ~4-6 px @640; NCUGD labels are wider (~9.5 px) than the 1-2 px hairlines they mark.
- **processed_bundle_v1** has 13 degenerate polygons and a few empty-label quirks; clean before training on it.

| dataset | images | disk | most common size (#sizes) | label type | empty-label rate | stroke width p50 @640 | warnings |
|---|---|---|---|---|---|---|---|
| CarDD_release | 12000 | 6GB | 1000x667 (286) | COCO json; PNG masks |  | 255.74 | 3 |
| SSGD | 2504 | 1GB | 1500x1000 (1) | COCO json |  |  | 0 |
| agdd_aircraft_glass | 439 | 122MB | 640x640 (2) | YOLO box; quad/OBB (9 tokens) | 0.0 |  | 5 |
| crackSegUltralytics | 4029 | 105MB | 416x416 (1) | YOLO polygon | 0.002 | 8.22 | 1 |
| crackseg9k | 0 | 4GB |  (0) | none found |  |  | 2 |
| cubit_inseg | 6996 | 37GB | 4500x3000 (3) | YOLO polygon; quad/OBB (9 tokens) | 0.0 | 4.81 | 6 |
| fcd_108_facade | 281 | 12MB | 512x512 (1) | PNG masks |  | 3.89 | 1 |
| glass_defect_detection2 | 578 | 23MB | 640x640 (1) | YOLO polygon; YOLO box; quad/OBB (9 tokens) | 0.269 | 4.82 | 2 |
| glass_segmentation_2026 | 1176 | 51MB | 640x640 (1) | YOLO polygon; quad/OBB (9 tokens) | 0.065 | 5.52 | 2 |
| processed_bundle_v1 | 8255 | 1GB | 1024x1000 (147) | YOLO polygon; quad/OBB (9 tokens) | 0.077 | 5.42 | 4 |
| fcd_108_grouped | 270 | 12MB | 512x512 (1) | PNG masks |  | 3.89 | 1 |
| glass_merged_linear | 1276 | 51MB | 640x640 (1) | YOLO polygon; quad/OBB (9 tokens) | 0.449 | 4.29 | 3 |
| glass_merged_seg | 1276 | 51MB | 640x640 (1) | YOLO polygon; quad/OBB (9 tokens) | 0.219 | 6.45 | 2 |

## Warnings per dataset

### CarDD_release
- 4000 image stem(s) occur more than once within a split (e.g. forward/backward pairs); label matching by stem is ambiguous.
- Masks: median stroke width ~255.744 px at 640 scale: thick strokes; use for localization pre-training only (see plan Section 5).
- 12000 images but 4000 masks: counts differ by >5%.

### SSGD
- none

### agdd_aircraft_glass
- Lines with 9 tokens found: either oriented boxes (4 corner points) or 4-point polygons. Open a few files and confirm before converting.
- Boxes only: no masks. Cannot be used for mask IoU / segmentation training as-is.
- All images share one square size: likely resized/stretched (Roboflow-style). Crack geometry/aspect may be distorted; physical width is not recoverable.
- 219 image stem(s) occur more than once within a split (e.g. forward/backward pairs); label matching by stem is ambiguous.
- Large images (>2000 px wide): tile before training at 640; cracks become invisible if downscaled whole.

### crackSegUltralytics
- All images share one square size: likely resized/stretched (Roboflow-style). Crack geometry/aspect may be distorted; physical width is not recoverable.

### crackseg9k
- No image files found.
- No labels recognized (no YOLO txt, COCO json or mask PNGs).

### cubit_inseg
- Lines with 9 tokens found: either oriented boxes (4 corner points) or 4-point polygons. Open a few files and confirm before converting.
- 6 archive file(s) in the folder (possibly unextracted or leftover).
- labels dir 'test/labels': 1 degenerate polygon(s) (<3 points or zero area).
- labels dir 'train/labels': 6 degenerate polygon(s) (<3 points or zero area).
- labels dir 'val/labels': 1 degenerate polygon(s) (<3 points or zero area).
- Large images (>2000 px wide): tile before training at 640; cracks become invisible if downscaled whole.

### fcd_108_facade
- All images share one square size: likely resized/stretched (Roboflow-style). Crack geometry/aspect may be distorted; physical width is not recoverable.

### glass_defect_detection2
- Lines with 9 tokens found: either oriented boxes (4 corner points) or 4-point polygons. Open a few files and confirm before converting.
- All images share one square size: likely resized/stretched (Roboflow-style). Crack geometry/aspect may be distorted; physical width is not recoverable.

### glass_segmentation_2026
- Lines with 9 tokens found: either oriented boxes (4 corner points) or 4-point polygons. Open a few files and confirm before converting.
- All images share one square size: likely resized/stretched (Roboflow-style). Crack geometry/aspect may be distorted; physical width is not recoverable.

### processed_bundle_v1
- Lines with 9 tokens found: either oriented boxes (4 corner points) or 4-point polygons. Open a few files and confirm before converting.
- labels dir 'labels/train': 7 degenerate polygon(s) (<3 points or zero area).
- labels dir 'labels/val': 6 degenerate polygon(s) (<3 points or zero area).

### fcd_108_grouped
- All images share one square size: likely resized/stretched (Roboflow-style). Crack geometry/aspect may be distorted; physical width is not recoverable.

### glass_merged_linear
- Lines with 9 tokens found: either oriented boxes (4 corner points) or 4-point polygons. Open a few files and confirm before converting.
- All images share one square size: likely resized/stretched (Roboflow-style). Crack geometry/aspect may be distorted; physical width is not recoverable.
- labels dir 'train/labels': 53% of label files are empty.

### glass_merged_seg
- Lines with 9 tokens found: either oriented boxes (4 corner points) or 4-point polygons. Open a few files and confirm before converting.
- All images share one square size: likely resized/stretched (Roboflow-style). Crack geometry/aspect may be distorted; physical width is not recoverable.

