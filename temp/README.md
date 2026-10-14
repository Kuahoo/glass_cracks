# Glass crack detection — dataset download and verification kit

**Prepared:** 2026-10-09. **IMPORTANT: this ZIP contains downloader programs and a dataset catalog, NOT the source images.**

The original ChatGPT execution environment has NO working DNS/HTTP data-transfer access. Public dataset landing pages were checked, but downloads were blocked. No image bytes or annotations are being misrepresented as acquired data.

When you run this kit on a normal computer with Internet, the script downloads the available datasets, preserves each publisher's layout/splits, checks for image bytes and archive corruption, computes SHA-256 file hashes, records failed accesses, and creates `glass_crack_datasets.zip` one directory **above** the output dataset folder.

## Quick start — Windows

Install Python 3.10+ and Git from https://git-scm.com/downloads . Extract this ZIP, open PowerShell in the extracted folder:

```powershell
py -m pip install -r requirements.txt
$env:ROBOFLOW_API_KEY="YOUR_ROBOFLOW_KEY"  # from https://app.roboflow.com/settings/api (do not share it)
py download_datasets.py
```

`RUN_WINDOWS.bat` also works after setting the environment variable in your terminal (in Windows CMD: `set ROBOFLOW_API_KEY=...`).

## Quick start — Linux/macOS

Install Python 3.10+ and Git. Extract the kit, open a terminal:

```sh
python3 -m pip install -r requirements.txt
export ROBOFLOW_API_KEY='YOUR_ROBOFLOW_KEY'
python3 download_datasets.py
```

Alternative: `./RUN_MAC_LINUX.sh`. If you don't have a Roboflow key, run without one; those three datasets will be **marked unavailable**, but GitHub/Drive sources will still be attempted.

Disk usage of the full CUBIT-InSeg download may be large. No reliable uncompressed size was verified, so **check available disk space** before attempting all datasets.

## Download just one or a subset

```sh
python3 download_datasets.py --only ncugd_glass_crack
python3 download_datasets.py --only agdd_aircraft_glass,cubit_inseg
python3 download_datasets.py --only glass_defect_detection2,glass_segmentation_2026,ncugd_glass_crack
```

### Included download targets

| ID | Publisher-reported images | Domain | Download | Rights note |
|---|---:|---|---|---|
| `glass_defect_detection2` | 578 | glass (type unverified) | Roboflow v2, key normally required | CC BY 4.0 claimed |
| `glass_segmentation_2026` | 506 (v10 export: 1,176 incl. 2x train augmentation) | glass (type unverified) | Roboflow v10, key normally required | CC BY 4.0 claimed |
| `ncugd_glass_crack` | 26 | glass crack (type unverified) | Roboflow v1, key normally required | CC BY 4.0 claimed |
| `agdd_aircraft_glass` | 438 original images reported | *aircraft* canopy | public GitHub clone + LFS if available | **CC BY-NC-SA 4.0** — noncommercial |
| `cubit_inseg` | 6,996 | *building facades*, not verified glass | author-linked Google Drive | CC BY 4.0 claimed |
| `fcd_108_facade` | 108 | *opaque facades*, not glass | author-linked Google Drive | CC BY 4.0 claimed |

**Never combine these into a single 'glass curtain wall' test set**. The counts above are authors' dataset sizes and may not match export versions or recursive file counts after preprocessing. See `datasets.json` for direct source links, tasks, and warnings. The major glass curtain wall GCC 385-image corpus has NO public downloadable raw archive confirmed in our source research and is NOT included.

## Outputs

```text
glass_crack_dataset_bundle/
    datasets.json                       # input source catalog
    datasets/
        glass_defect_detection2/
        glass_segmentation_2026/
        ncugd_glass_crack/
        agdd_aircraft_glass/
        cubit_inseg/
        fcd_108_facade/
    download_status.json               # per-dataset success/failure, error messages
    files_sha256.csv                    # SHA-256 per downloaded file
glass_crack_datasets.zip                # built only when at least one dataset has images
glass_crack_datasets.zip.sha256         # package SHA-256
```

The ZIP includes original source directory structures, not resampled/merged images. Inspect `download_status.json`: status `FILES_PRESENT_NOT_LABEL_VERIFIED` proves bytes and simple file type, **not** ground truth, labeling quality, completeness or site-disjoint splits. `NO_IMAGE_BYTES` and `DOWNLOAD_FAILED` do not indicate successful acquisition.

## Troubleshooting

- **Roboflow**: Create a free account, generate an API key and confirm you can export the selected public dataset version. Some public projects must first be *forked* to your workspace. In that case, edit the corresponding `workspace`, `project` and `version` fields in `datasets.json`. The script will never place your key into the dataset ZIP or log.
- **Dataset version changed**: list the versions that exist and which export formats are prebuilt, then adjust `version` in `datasets.json`:
  `curl -s "https://api.roboflow.com/<workspace>/<project>?api_key=$ROBOFLOW_API_KEY"`. v6 of `1-25-2026-glass` does not exist; v10 is the only version with a prebuilt `yolov8` export.
- **Drive large folder**: Google throttles anonymous downloads ("Cannot retrieve the public link … or have had many accesses"); retrying soon after does not help. Open the folder in a browser, download it as a ZIP, and unzip it into `datasets/<id>/`, then run `--audit-only`. `gdown`'s recursive folder output is **not proof of completeness**: compare against `gdown.download_folder(url, skip_download=True)`.
- **Status after partial runs**: `--only` updates just the listed datasets in `download_status.json` and `files_sha256.csv`; records for other datasets are kept.
- **AGDD**: If images are pointers, install `git-lfs`, run `git lfs install`, then rerun; if the authors committed binary images directly, Git is sufficient.
- **A downloaded ZIP contains images but `images_on_disk=0`**: the script separately counts `.jpg`/`.png` members within sound ZIP files; unpack nested archives only after reviewing their paths. Never execute code from archives without review.
- **Unsupported/partial archive**: inspect the publisher link manually, and compare named images/annotations against the paper; the script may require adapting to a host's changed share settings.

## Cleaned training sets

`python3 clean_datasets.py` reads `glass_crack_dataset_bundle/datasets/` (unchanged) and writes `glass_crack_dataset_bundle/processed/`:

| Output | What it is |
|---|---|
| `glass_merged_seg/` | `glass_defect_detection2` + `glass_segmentation_2026` merged into one YOLO-seg set (classes `bubble, crack, inclusion, scratch`). The 261 photos present in both appear once; the 10 `Gemini_Generated_Image_*` sources are dropped; box rows are converted to polygons. Splits are rebuilt so every source photo and every near-duplicate group (burst frames, perceptual hash ≤12/256 bits) sits in exactly one split; sources that only exist as 2x-augmented copies are kept in train. `manifest.csv` maps every file back to its origin. |
| `ncugd_external_test/` | NCUGD with repeated sources removed (19 images), relabeled to the merged class ids (`glass-crack` → `crack`). Evaluation only; no near-duplicates of `glass_merged_seg`. |
| `fcd_108_grouped/` | FCD-108 re-split so all 3 photo variants of a facade stay in one split (30 train / 6 valid facades). |
| `cleaning_report.json` | Per-split counts, label fixes, dropped sources, and the leakage checks (the script fails if any remain). |

`datasets/cubit_inseg/` is extracted in place (`images/` + `labels/` per split; the original zips are kept) with a `data.yaml` naming class 0 `crack` and class 1 `spalling`, inferred from the `CR*`/`SP*` filename prefixes. AGDD is left as published; use its `generate_*_dataset.py` scripts.

## Research and commercial-use caution

Original source/usage rights may differ from a project's declared top-level license. Attribution alone does not remove AGDD's noncommercial restriction. CUBIT-InSeg and FCD-108 do NOT provide verified positive glass fractures. Visible defect classes are not a substitute for physically confirmed crack labels or building-disjoint holdouts. Keep source datasets separate until deduplication and provenance have been audited.

Source pages were inspected on 2026-10-09. Code was tested only with generated **synthetic local fixtures**, NOT against live external download APIs.
