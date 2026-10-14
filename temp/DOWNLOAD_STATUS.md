# Download status

**As of 2026-10-12, all 6 catalogued datasets are on disk** in `glass_crack_dataset_bundle/datasets/`. Per-dataset byte audits are in `glass_crack_dataset_bundle/download_status.json` and per-file hashes in `files_sha256.csv`.

| ID | Images | How it was obtained |
|---|---:|---|
| `glass_defect_detection2` | 578 | Roboflow v2 (yolov8) |
| `glass_segmentation_2026` | 1,176 | Roboflow v10 (yolov8); v6 does not exist |
| `ncugd_glass_crack` | 26 | Roboflow v1 (yolov8) |
| `agdd_aircraft_glass` | 438 (219 pairs) | GitHub clone |
| `cubit_inseg` | 6,996 | Google Drive via gdown; zips extracted in place |
| `fcd_108_facade` | 281 | Google Drive, downloaded manually in a browser after gdown was throttled |

Not obtained: GCC-385 and the Zhou curtain-wall dataset (no public download verified).

Known data-quality issues (cross-dataset overlap, near-duplicate split leakage, synthetic images, coarse FCD masks) and the cleaned outputs that address them are described in `README.md` under "Cleaned training sets". "Files present" is not label validation.
