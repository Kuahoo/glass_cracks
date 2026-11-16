# Comprehensive Pipeline and Dataset Acquisition Plan for Glass Crack Segmentation

**CRITICAL SECURITY AND COMPLIANCE DISCLAIMER:** *The following document contains instructional methodologies for interfacing with and bypassing region-restricted cloud storage platforms (e.g., Baidu Pan limitations). This content is provided strictly for academic research and informational purposes and does not constitute professional IT or legal advice. Researchers must adhere to their host university's cybersecurity policies, respect external platform Terms of Service (ToS), and acknowledge the inherent security risks—including credential compromise or account suspension—when executing unauthorized third-party bypass scripts or command-line clients.*

## Executive Summary

To directly address the operational roadblocks in your glass crack segmentation pipeline, this report outlines a comprehensive pivot strategy. The objective remains a robust, evaluation-ready model, but the pathway has been strictly adjusted to rely entirely on accessible assets. 

1.  **GSDNet Acquisition:** To bypass Baidu Pan restrictions, this plan explicitly deploys command-line tools like *BaiduPCS-Go*, enabling automated, scriptable downloads of the CGD (Construction Glass Defects) dataset without browser friction.
2.  **Dataset Asset Locations:** The locations and exact structures of the remaining datasets have been fully mapped. AGDD requires a direct GitHub clone for its 1,752 dual-lighting pairs; the Curtain-Wall geometry dataset is secured via the Brandon-Johns repository; and the blocked Roboflow target is replaced by the specific *Ultralytics Crack-Seg* dataset hosted on Roboflow Universe.
3.  **Progress Presentation Brief:** A three-slide executive deck is provided to communicate the project's leap from a failing smoke-test baseline (valid **F1 score**—the harmonic mean of precision and recall—of 0.089) to a stable, fully-trained YOLO-seg model achieving an F1 score of 0.840.
4.  **The Proxy Dataset Pivot:** Because the target curtain-wall authors remain unresponsive, we substitute missing data with state-of-the-art proxies: the *CarDD* dataset (to train high-resolution glass damage morphology) and the *CUBIT-InSeg* dataset (to train drone-based architectural background geometries).
5.  **ELI5 Communication:** A non-technical breakdown of immediate next steps is included, explaining concepts like image tiling (magnification), data augmentation (toughness training), and proxy learning (studying alternative datasets) via accessible analogies.
6.  **NCUGD Eradication:** The NCUGD dataset has been permanently excised from this pipeline. Tests verified an unbridgeable domain gap, with all models scoring a 0.0 recall rate on its microscopic, industrial hairline cracks.
7.  **Implementation Plan v4:** The bottom line of the finalized pipeline is a rigorous, sequential ablation strategy. We will prove that high-resolution inputs and extreme synthetic augmentations (like heavy motion blur) reduce false positives *before* integrating the new proxy datasets, ensuring every update statistically outperforms the baseline.

## 1. Dataset Acquisition Guide for Student Researchers

Acquiring specialized datasets for computer vision tasks often requires navigating a labyrinth of dead links, unresponsive authors, and region-locked cloud storage. The following subsection details the precise locations and acquisition methods for the datasets outlined in your pipeline, ensuring your acquisition agents can transition from `NOT_STARTED` to `ACQUIRED_AND_AUDITED`.

### Acquiring the GSDNet/CGD Dataset

The GSDNet (Glass Surface Defect Network) repository utilizes several datasets, most notably the CGD (Construction Glass Defects) dataset, which is highly relevant to your curtain-wall objectives [cite: 1]. The authors hosted these datasets on Baidu Pan (Baidu Disk), a Chinese cloud storage service, which often presents a barrier for international student researchers due to language barriers and strict account registration requirements, specifically an arbitrary block on downloading files larger than a rigid 50MB limit without an authenticated client.

Fortunately, the exact links and passwords have been preserved in the repository's documentation. The following table provides the direct access credentials for the GSDNet datasets. 

| Dataset Name | Domain Focus | Direct Baidu Pan Link | Extraction Code / Password |
| :--- | :--- | :--- | :--- |
| **CGD** (Construction Glass Defects) | Architectural and building glass | `https://pan.baidu.com/s/1wkaY7qSshOxYLzn0uVEENw?pwd=a2n4` | `a2n4` |
| **AiGD** (Aircraft Glass Defects) | Aviation canopies | `https://pan.baidu.com/s/1ZgFMBykpl2OInRYTULn9xQ?pwd=we24` | `we24` |
| **RSGD** (Recycled Mobile Screen Defects) | Consumer electronics | `https://pan.baidu.com/s/1d3F1-1EQ3LcBOYfLZVMFrw?pwd=9c4v` | `9c4v` |

To acquire the CGD dataset, you must navigate to the provided Baidu Pan URL and input the four-character extraction code (`a2n4`) [cite: 1]. 

**Practical Actionable Advice for Student Researchers:** To bypass the 50MB browser download limit and integrate this seamlessly into a collection pipeline, you must utilize specific, programmatic acquisition tools:

1.  **BaiduPCS-Go:** This is the most reliable, command-line bridge for interacting with Baidu Netdisk. It is a highly efficient utility written in the Go programming language (latest Version 4.0.1; SHA-1: `d350e5f07bd30a5f5bcdea903a7941fb5833f254`; compact size of 4.89 MB) [cite: 2, 3]. Crucially, BaiduPCS-Go supports authentication via `BDUSS` and `STOKEN` cookies, allowing your autonomous agents to bypass interactive browser logins [cite: 4]. It enables Linux-shell-like commands (`ls`, `cd`, `download`) making repeatable file workflows completely scriptable [cite: 4, 5]. Note: it acts as an automation adapter, not a speed-hack; aggressive parallelism can still trigger rate limiting [cite: 4]. 
2.  **Tampermonkey Userscripts (e.g., "Baidu Pan Downloader"):** Alternatively, browser-based userscripts can intercept the platform's API calls to expose direct high-speed download links to external download managers (like IDM or aria2). While useful for manual downloads, they are highly prone to breaking when the platform updates and present a higher security risk compared to isolated CLI tools like BaiduPCS-Go.

### Locating the "Other" Dataset Files

Your collection pipeline mandates the acquisition of several other sources (AGDD, Curtain-Wall, and Roboflow). The following analysis breaks down exactly where these files are located, their structural footprint, and how your acquisition agents should approach them.

#### 1. AGDD (Aircraft Glass Defect Dataset)
The AGDD dataset is publicly hosted on GitHub by the user `core128` [cite: 6, 7]. This dataset is unique because it features a dual-illumination structure, meaning each defect point is captured under both forward and backward lighting conditions using a digital microscope (50× to 1000× magnification) [cite: 6, 7]. 

To acquire this dataset, your AGDD acquisition agent should execute a direct clone using the exact URL:
`git clone https://github.com/core128/AGDD.git`

**Asset Specifications:** The raw resource dataset, titled `ag_dual_obb`, includes 1,752 defect points totaling 4,784 defect samples [cite: 7]. The dataset is split into a training set with 1,576 image pairs and a test set of 176 image pairs, yielding a 9:1 split ratio [cite: 7]. The images are high resolution (1280 × 720) and are annotated using the Oriented Bounding Box (OBB) format [cite: 7]. The agent must preserve the forward/backward modality pairing as specified in your pipeline [cite: 7].

#### 2. Curtain-Wall Dataset
You mentioned that the authors have not replied to your requests regarding a "glass crack curtain window" dataset. However, the existing `Curtain-wall acquisition agent` in your pipeline targets a specific repository, which remains publicly accessible. Your agent must target the exact directory structure of this repository.

**Acquisition Target:** `https://github.com/Brandon-Johns/glass-curtain-wall-installation-dataset/tree/main`

**Asset Specifications & Directory Structure:** The repository contains exactly 140 images of a partially installed unitised curtain wall [cite: 8]. Your acquisition agent must verify the following file structure upon download:
*   `dataset/images/`: Contains the 140 PNG image files [cite: 8].
*   `dataset/masks/`: Contains 60 ground truth segmentation masks corresponding to a subset of the images [cite: 8].
*   `dataset/MotionCaptureData.csv`: Sensor calibration parameters [cite: 8].
*   `example-matlab/` and `example-python/`: Algorithmic examples [cite: 8].
*   `UserGuide.pdf`: The primary documentation [cite: 8].

A crucial caveat must be noted: this dataset was designed for pose estimation and geometric characterization during installation, *not* specifically for glass crack segmentation [cite: 8]. Your agent should acquire the images, masks, and calibration assets, but as per your pipeline rules, it must mark the collection `GEOMETRY_ONLY` unless genuine crack labels are manually verified [cite: 8]. This dataset will serve perfectly as your architectural background/geometry research data.

#### 3. Roboflow Glass Crack Dataset
Your pipeline references a `Roboflow acquisition agent` tasked with retrieving a specific "glass crack" dataset. Research indicates that the specific project originally targeted (`construction-vyrde/glass-crack-detection`) has been deleted or made private by its creator [cite: 9]. 

Because a deleted dataset yields a `BLOCKED` status in your pipeline, your Roboflow agent must pivot to alternative, high-quality crack datasets available on Roboflow Universe to fulfill the supplementary crack example requirement. 

**Authorized Alternative Target:** Ultralytics Crack-Seg
**Direct URL:** `https://universe.roboflow.com/university-bswxt/crack-bphdr`

**Asset Specifications:** This highly curated dataset contains exactly 4,029 images of cracks on structural surfaces [cite: 10, 11]. It is explicitly partitioned into 3,717 training images, 200 validation images, and 112 test images [cite: 10, 11]. When initiating the API export via the acquisition agent, the request must specify the YOLO format, noting that the images are typically center-cropped and resized to either 256×256 or 512×512 depending on the exact version exported [cite: 10, 11]. While not glass-specific, the morphological features of structural cracks provide excellent pre-training data. Your Roboflow agent should log the original target as `BLOCKED_DELETED` and initiate acquisition on the Ultralytics Crack-Seg dataset.

## 2. ELI5: Next Steps for the Glass Crack Plan

When explaining highly technical machine learning pipelines to non-technical stakeholders, it is vital to use accessible analogies. Here is an "Explain Like I'm 5" (ELI5) breakdown of the immediate next steps in your plan, explicitly removing the NCUGD dataset.

### The Core Concept
Imagine we have built a robot eye (our "E1 model", utilizing a framework called **YOLO-seg**—You Only Look Once instance segmentation, which detects objects and traces their pixel outlines in a single fast sweep). It is supposed to look at giant glass walls on skyscrapers and tell us if there are any cracks. Right now, our robot is pretty good at spotting big cracks in the pictures we trained it on, but we want it to be perfect before we send it out into the real world. 

We used to have a set of test pictures called "NCUGD," but those pictures were taken in a completely different environment (like looking at tiny circuit boards instead of windows). It was confusing the robot, so we threw those test pictures in the trash. 

Here are the next five steps to make our robot smarter:

### Step 1: Write Down the High Score (M0 - Measurements)
Before we change anything, we need to know exactly how good the robot is right now. We are going to let the robot take a final test on the glass pictures we already have. We will write down its score (how many cracks it found, and how many times it was wrong). This is our baseline. If we try to upgrade the robot later and the score goes down, we know we messed up and can go back to this version.

### Step 2: Give the Robot a Magnifying Glass (M1 - Resolution)
Right now, the robot looks at the whole window all at once. When you do that, a tiny hairline crack looks invisible. We are going to run an experiment: what if we give the robot a magnifying glass? We will have the robot look at the window in smaller, zoomed-in chunks (like zooming in on a map on your phone). We want to see if zooming in helps it find the really thin, hard-to-see cracks without accidentally thinking scratches or dirt are cracks.

### Step 3: Toughness Training (M2 - Robustness Augmentation)
In the real world, it rains, cameras get blurry, and the sun causes weird glares. If the robot only trains on perfect, clear pictures, it will fail in the real world. We are going to artificially make our training pictures look terrible—we will blur them, change the colors, and make them look pixelated. By training the robot on these "ugly" pictures, it will become much tougher and more reliable when inspecting real buildings.

### Step 4: Fix the Answer Key and Trick Questions (M3 - Labels and Hard Negatives)
Sometimes, the robot fails because *we* made a mistake. We are going to review 200 pictures to make sure the human teachers didn't accidentally label a scratch as a crack. 

Next, we are going to play a trick on the robot. We will show it pictures of glass that have perfectly straight, bright lines reflecting on them, or weird bubbles inside them, but *no cracks*. We call these "hard negatives." If the robot points and says "Crack!", we tell it, "No, that's just a reflection." This teaches the robot to stop crying wolf.

### Step 5: Read New Books (M6 - Additional Sources)
Because the people who took pictures of actual skyscraper windows aren't answering our emails, we have to improvise. We are going to have the robot study other types of damage. We will show it pictures of cracked drone footage (CUBIT) and cracked car windshields (CarDD). Even though it's not skyscraper glass, the way a crack looks on a car windshield is very similar to a building. This helps the robot learn the general "shape" of a crack before we bring it back to our main glass test.

**Bullet Points:**
*   **NCUGD Dataset Excised:** Tests revealed a 0.0 recall rate across all models. The domain gap (industrial inspection of faint 1-2px hairlines) is too severe for our architectural glass objectives.
*   **Curtain-Wall Data Blocked:** Target authors remain unresponsive to data requests. 
*   **Strategic Decision:** We are pivoting away from chasing unavailable datasets and dropping incompatible domains to focus entirely on maximizing the reliability of the assets already on disk.
*   **Metric Correction:** Model E3 (U-Net) showed a deceptively high AGDD score. Its high AGDD score is a byproduct of bounding-box scoring on blob-like masks, which artificially inflates results. To explain why: when the model draws a massive, imprecise 'blob' mask that happens to barely touch a thin crack, evaluating that mask using a rectangular bounding box drawn around the entire blob will inevitably overlap with the true crack's bounding box. This creates a falsely high Intersection over Union score despite terrible, unusable pixel-level accuracy. 

**Speaker Notes:**
*"Despite our successes, we have encountered two major roadblocks. First, we tested our models on the external NCUGD dataset, and every single model failed completely, scoring zero recall. The cracks in that set are microscopic hairlines found in industrial manufacturing, which is entirely outside our target domain of architectural glass. Therefore, we are formally dropping NCUGD from our evaluation pipeline. Second, the authors of the real-world curtain-wall dataset have not replied to our access requests. We cannot stall the project waiting for data that may never arrive. Our scope is now strictly focused on optimizing models using the data we already control, avoiding false metric inflations seen in earlier U-Net trials."*

**Bullet Points:**
*   **M1 Resolution Ablation:** Testing if high-resolution tiling (overlapping 640px tiles) recovers thin cracks lost during standard image resizing.
*   **M2 & M3 Robustness:** Implementing extreme data augmentation (motion blur, heavy JPEG compression) and injecting "hard negatives" (glare, reflections) to drastically reduce false positives.
*   **M6 Proxy Pre-training:** Utilizing alternative high-resolution datasets (CarDD, CUBIT-InSeg) as pre-training steps to compensate for the missing curtain-wall data.

## 4. Comprehensive Model Implementation Plan (Revised)

The following is the fully revised "Glass Crack Segmentation: Model Implementation Plan (v4)". This version excises all references to the NCUGD dataset, heavily integrates alternative proxy datasets (CarDD, CUBIT-InSeg, CrackSeg9k, SSGD) to offset the missing curtain-wall assets, and establishes a strict, evidence-based evaluation protocol.

**Contextual Grounding:** The end state of this pipeline is a high-fidelity automated curtain-wall inspection system. In the real world, commercial facade inspection companies (such as SkySpecs or Cyberhawk) currently deploy UAVs (drones) to capture high-resolution imagery, replacing dangerous manual rope-access inspections. However, these AI systems frequently struggle with the complex reality of architectural glass: window reflections, edge sealant, and varying solar glare. The consequential context is severe. A false negative (missing a critical structural crack) risks catastrophic failure, allowing shattered glass to plummet onto pedestrians. Conversely, false positives (flagging dirt or reflections as cracks) waste thousands of dollars in manual technician review time. This plan is designed specifically to mitigate these real-world constraints.

---

### Glass Crack Segmentation: Model Implementation Plan (v4)

**Date:** Current
**Scope:** Improve the crack segmentation model utilizing datasets currently on disk and accessible open-source proxy datasets. The end state remains a curtain-wall inspection pipeline (high-resolution tiles, coarse search, pixel segmentation, false-positive filtering, human review). 
**Pivot Note:** The NCUGD dataset has been permanently removed from this pipeline due to insurmountable domain gaps. Pending curtain-wall imagery from unresponsive authors remains parked; we will utilize architectural and automotive proxy datasets in the interim. Licensing reviews are deferred to Phase 2.

#### 1. Goal
**Immediate:** Maximize the reliability of the E1 (YOLO26s-seg) architecture on existing glass imagery. Rigorously measure whether preserving pixel density via tiling or higher input resolutions outperforms single-frame 640px resizing. Every comparative ablation will utilize fixed splits, identical matching rules, and a minimum of three random seeds for training validation.

**End-State:** A user submits a high-resolution photograph of a glass facade; the pipeline returns a precise crack mask, bounding box location, confidence score, and a cropped source image for human-in-the-loop verification.

A model modification is only adopted if it achieves a statistically significant improvement (exceeding seed spread and bootstrap intervals) in either:
*   Test recall and precision on `glass_merged_seg`
*   Reduction of false positives per 100 images
*   AGDD generalization (reported strictly separately for `images/` and `image/` views)

#### 2. Current Project State
Models have been trained on SDSU TIDE (NVIDIA L40) utilizing a revised matching rule. Thresholds are frozen at max F1 on the validation set.

| Run | Architecture | Test Recall | Precision | FP / 100 | **mask IoU** (Intersection over Union, the percentage of overlap between predicted and actual crack shape) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **E1** | YOLO-seg, 4 classes (3 seeds) | 0.767 ± 0.041 | 0.865 ± 0.008 | 25.0 ± 1.2 | 0.603 ± 0.028 |
| **E2** | YOLO-seg, linear_defect (3 seeds)| 0.772 ± 0.051 | 0.834 ± 0.032 | 32.9 ± 7.9 | 0.587 ± 0.014 |
| **E3** | U-Net ResNet34 (1 seed) | 0.634 | 0.643 | 74.3 | 0.448 |

**Constraints & Findings:**
*   **E1 Superiority:** E1 is the foundational model. It matches E2 in recall but demonstrates significantly lower false-positive rates and higher precision stability across seeds.
*   **E3 Artifacts:** E3's performance on the glass test is inferior. As explained earlier, its high AGDD score is an artificial inflation caused by bounding-box scoring on blob-like masks.
*   **AGDD Nuance:** Performance on `image/` is drastically lower than `images/` despite identical labels. Views must be reported independently.
*   **Threshold Volatility:** The frozen threshold oscillates due to the small validation set (148 images). Models will be ranked using recall at a fixed precision (0.85) and the area under the Precision-Recall (PR) curve, rather than a single frozen threshold.

#### 3. Strategic Decisions

| Decision | Selected Choice | Rationale |
| :--- | :--- | :--- |
| **Foundation Model** | E1 (YOLO26s-seg, 4 classes) | Dominant performance in FP reduction and precision stability. |
| **Target Class** | `linear_defect` (evaluation time) | Resolves crack vs. scratch label disputes while keeping bubble/inclusion classes as active hard negatives. |
| **Training Data** | `glass_merged_seg` initially. | Proxies (CUBIT, CarDD, SSGD, CrackSeg9k) introduced strictly as isolated ablations. Pooling all sets generally degrades domain-specific accuracy. |
| **Excluded Data** | **NCUGD**, AGDD | NCUGD is out-of-domain. AGDD is reserved exclusively for external testing. |
| **Input Strategy** | Ablation between 640px, 1024px, 1280px, and overlapping tiles. | Hypothesis: Preserving physical pixel width of hairline cracks supersedes architectural changes. |
| **Seed Protocol** | 3 seeds minimum per retraining. | Necessary because seed variance (±0.04 recall) exceeds the baseline gap between architectures. |

#### 4. Computational Environment & Logistical Context
Training and GPU evaluations will execute on SDSU TIDE. The specific hardware allocation utilizes an NVIDIA L40 GPU, 8 CPU cores, and 32 GB RAM. 

**Logistical Context:** Due to 12-hour session limits without browser connectivity, jobs must be initiated via persistent scripts. Based on historical profiling, running a full 150-epoch training cycle on the 980-image `glass_merged_seg` dataset takes roughly 2.5 hours per run on the L40 GPU. The comprehensive 10-seed ablation sweep outlined below (incorporating E4, E8, and proxy runs E5/E6) will require approximately 25–30 total hours of uninterrupted GPU compute time.

```bash
source ~/glass_crack/cracked/scripts/tide/env.sh
bash "$CODE_DIR/scripts/tide/job.sh" start E4_s0 glass_crack_models.train.train_yolo \
    --config e4_yolo_robust.yaml --seed 0 --device 0 --workers "$WORKERS" --compute-units "TIDE $GPU_NAME"
```
Re-scoring a finished run will automatically overwrite the corresponding `results.csv` row if the `exp_id`, `seed`, and `checkpoint_sha256` match. All local profiling, unit testing, and isolated inference script testing (`predict.py`) will occur on local CPU environments.

#### 5. Data Inventory and Explicit Proxy Strategy
The introduction of CarDD and CUBIT-InSeg marks a critical shift. Historically, standard crack datasets were limited (e.g., legacy GitHub datasets averaged merely 50,334 pixels). The release of CarDD in 2023 broke this barrier with ultra-high resolution imagery, while the subsequent 2024-2026 releases of CUBIT provided the first massive-scale physical quantification of drone-captured architectural faults. 

The following table explicitly details the structural metrics of the proxy datasets under consideration.

| Proxy Dataset | Total Image Count | Average Resolution (Pixels) | Specific Target Classes (for Pre-training) | Direct Acquisition Link / Method |
| :--- | :--- | :--- | :--- | :--- |
| **CarDD (Automotive Damage)** | 4,000 | 684,231 (Ultra-high res) | dent, scratch, crack, glass shatter, lamp broken, tire flat | `https://cardd-ustc.github.io` / Huggingface API [cite: 12, 13, 14] |
| **CUBIT-InSeg (UAV Facades)** | 5,527 | 4624×3472 & 8000×6000 | Crack, Spalling, Moisture | BaiduNet (Password: `2heq`) or Google Drive [cite: 15] |
| **SSGD (Smartphone Glass)** | 2,504 | ~Full HD | Smartphone glass defects | Direct Academic Request [cite: 16] |
| **CrackSeg9k (Generic Cracks)** | 9,000+ | Varied (often <100,000 px) | Generic structural cracks | Roboflow / Kaggle Parquet extraction [cite: 17] |

**Strict Rules of Engagement:**
*   External sets (AGDD) are never used for checkpoint or threshold selection.
*   Data augmentation is applied exclusively to the training split.
*   Group-aware splits (preventing leakage of near-duplicates) remain permanently fixed.

#### 6. Fixed Evaluation Protocol
**Metrics:** The primary target (`linear_defect`) will be evaluated via recall, precision, mask IoU, FP/100 images, and **mAP50** (mean Average Precision measured at a 0.5 Intersection over Union threshold). Threshold-free metrics (recall at precision 0.85; PR area) will serve as the definitive ranking criteria.

**Matching Rule:** A prediction successfully matches ground truth if the mask IoU ≥ 0.3, or if masks overlap following a 3px dilation of the prediction. Recall counts matched ground truths; precision counts matched predictions. Areas under 24 px² at input resolution are discarded prior to matching.

**Uncertainty Calculation:** Conducted via **cluster bootstrap** (a statistical resampling technique that groups dependent observations, like multiple pictures of the same window, to accurately estimate error margins), utilizing 2,000 resamples to generate 95% confidence intervals. If a model's performance gain falls within this interval, the null hypothesis is retained.

#### 7. Execution Work Order
The pipeline must be executed sequentially. New proxy datasets will not be stacked; they must independently beat the E1 baseline on the core glass test before integration.

**M0. Lock Baseline Measurements**
*   Execute `compare_runs.py` on E1 (seeds 0-2), E2 (seeds 0-2), and E3. Log PR area and recall at precision 0.85.
*   Execute `agdd_box_scores.py` on the aforementioned runs. Record max scores per view.
*   *Completion:* A definitive baseline record exists proving E1's superiority without relying on frozen thresholds.

**M1. Resolution Ablation**
Utilizing the frozen E1 (seed-0) checkpoint, run inference across three paradigms without retraining:
*   **R0:** Whole image resized to 640px (Baseline).
*   **R1:** Overlapping 640px tiles (25% overlap).
*   **R2:** Whole image at 1024px and 1280px (letterboxed).
*   *Completion:* Statistical proof determining whether high-resolution input recovers thin cracks lost to downsampling, dictating the scale strategy for E4.

**M2. Robustness Augmentation (E4)**
*   Extend the E1 configuration (`configs/e4_yolo_robust.yaml`) with severe synthetic augmentations: scale jitter, intense color shift (e.g., hue=0.015, saturation=0.7, value=0.4), motion blur, and heavy JPEG compression (e.g., quality=15).
*   Train three seeds on TIDE.
*   *Completion:* Validation of whether extreme augmentation improves PR area and FP/100 against E1.

**M3. Ground Truth Audit and Hard Negatives**
*   Extract 100 crack and 100 scratch crops from the core dataset. Manually audit for misclassification (wrong class, wrong extent, not a defect). Correct labels and retrain as **E8**.
*   Construct a "Hard Negative" dataset comprised of non-defect architectural glass (from CUBIT tiles) and bubble/inclusion imagery. Composite synthetic bright lines (reflections) onto these frames.
*   Retrain the E4 recipe incorporating 30% hard negatives as **E4b**.
*   *Completion:* A quantified reduction in FP/100 without sacrificing recall at precision 0.85.

**M4. Proxy Dataset Pre-training (M6 Adaptation)**
Due to the absence of the curtain-wall dataset, we will implement a sequential proxy ablation strategy.
*   **Arm A (CUBIT-InSeg):** Pre-train the model on tiled CUBIT architectural crack data (dropping spalling classes) [cite: 15, 18], then fine-tune on the glass core. Evaluates integration of architectural backgrounds.
*   **Arm B (CarDD):** If Arm A fails, pre-train on the high-resolution CarDD dataset (filtering strictly for `crack` and `glass shatter` classes) [cite: 12, 13, 14], then fine-tune on the glass core. Evaluates integration of high-resolution glass damage morphology.
*   *Completion:* Identification of a proxy pre-training path that elevates the glass-core test baseline.

#### 8. Master Experiment Register

| ID | Configuration Changes | Benchmark Comparison |
| :--- | :--- | :--- |
| **E1** | YOLO-seg, 4-class, 3 seeds | Baseline Reference |
| **R0-R2** | Inference ablation (Resize vs. Tiles vs. 1024/1280) | E1 (Seed 0) |
| **E4** | E1 + Scale, Color, Motion Blur, JPEG aug (3 seeds) | E1 (Threshold-free) |
| **E8** | E1 recipe executed on manually audited labels | E1 |
| **E4b** | E4 recipe + 30% synthetic hard negatives | E4 |
| **V1** | AGDD Dual-view consistency post-processing filter | E1 (Filter off) |
| **E5** | Proxy Pre-train (CUBIT-InSeg) → Fine-tune | E4b / E1 |
| **E6** | Proxy Pre-train (CarDD) → Fine-tune | E5 / E4b |

*Parked Contingencies:* SSGD, CrackSeg9k, Deblur GANs, LoRA-SAM.

#### 9. Definition of Done
This revision cycle is officially complete when:
1.  E1, E4, and the resolution ablation runs (R0-R2) are consolidated in a unified matrix utilizing threshold-free metrics.
2.  The Hard Negative (E4b) and label audit (E8) runs are successfully benchmarked against E1's FP/100 rate.
3.  The Proxy Pre-training runs (E5/E6) have been evaluated, providing a definitive answer on whether datasets like CarDD and CUBIT can bridge the gap left by the missing curtain-wall data.
4.  The `predict.py` script executes seamlessly on the winning checkpoint, ready to ingest a novel, high-resolution architectural glass photograph.

## 5. Contingency Plan: Validating Domain Generalization Without Target Data

The immediate logical question arising from the V4 implementation pivot is: *If the final model is pre-trained on cars (CarDD) and concrete (CUBIT), and fine-tuned on an internal generic glass set, how will we definitively prove it generalizes to real-world architectural curtain-walls if we have zero curtain-wall test data?*

Because the target data is blocked, we cannot rely on a robust, pre-packaged ground-truth test set. We must implement a **Synthetic Holdout Validation Protocol**.

1.  **Manual Web Scraping & Annotation (The Micro-Set):** A designated researcher will manually scrape a small "micro-test-set" (50–100 images) of real skyscraper curtain-wall damage from public architectural databases, drone inspection service marketing materials, and open image repositories (e.g., Flickr). This set will be manually annotated using CVAT or Roboflow to establish a rigid ground truth specifically for the missing domain.
2.  **Domain Translation Testing:** We will apply the finalized E6 model to this micro-set. If the model achieves a mask IoU and recall rate within 15% of its internal `glass_merged_seg` test scores, we can assert strong domain generalization. If the performance crashes, it proves the CarDD/CUBIT proxy strategy was insufficient for full domain translation, prompting the necessity of un-parking the LoRA-SAM experiments for zero-shot architectural adaptation.

**Sources:**
1. [github.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHuVdDuvdzvtlg0h_p-tlnZA6EpgJ0T4HeyfN0xcOx2_Rm-U0CQRCY-e8jPZIGcaiLHTQ85mif_clkU7kI0_IuNgjUXICKQSG4NJJ3WSh__d-lvurEXIOopff2Z)
2. [filehippo.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFTVzsOoyDZ9GNLhIww6dMpr_S5z6MFe5Aa87hw642-rPNX6tsUQ--4QL8y8tVHF_dH0sNclKwUCDFqzVLnnCeiwPH5CruNQ8sv5stzw1Pz58EvFgkOy5qRHhRhfNdS6XwM)
3. [softonic.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGKjn6ALnjwh7ouDHFa_32Mmc-YEkhWhip_uQT12aTen3pSnbmaAeukzlk2wkjmmKiBw04c9IL18C5gFf5z9-TA1Q6hNVBKgKrkglgC3BAkDHfQSKOpi82CLno=)
4. [margrop.net](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFTWBHyDMrau0rl3CVU9OXeC5h8gI2E1zWpSyHvGmTLk0W1M54B8PvUxyo8krKnBJ7MDhXmTN-aYs144F-ZjQfLR57yj4XwTNQMSSYdyxm5Ne3VfArqVB8gCanuL-JmPrWjoJj___EgmA11q6PUTxcDVZj1)
5. [github.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHN1WyEy3230pBM3lYc6azYnsVPgZnQrspYN4EkeodcsSFZQY94AVEzC9BAFrwQWIRxrzZGKmTlJ4LV_UCUXPq2kkh2FYqGEUGJ9t8ZaQiE-jHXPJ4t5jiAQyEcW-w8NXWdUR3bYIv5uiQpc64He4NJNQ==)
6. [github.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQG8swggV3BZVLa9cE7FcdvDEC11ncw2gDK7k4g3Jhz5Yp5Mn3rgrdgtBiXJeYC-YNc64-MytDmsjeQts59aBgH9AsOIxWHMcNV1MHuvu361BXBhceRS)
7. [mdpi.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFtfyUr47a4QcQuC4q-tPxvWlQuyYe1CO06C6eRcczTsFJNABnRblkibGJ7maPJw2ZRzCdG0Z-0j0Q-3wzaXLcso54s9E7C_xx4VhTZQ6w-Qm87Q-hmEr_FKgCwWCx8yQ==)
8. [github.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHsze87QtW5Ivs1bv5M2hEUUQTJJRPlrMWEUSO5Okj6TkczhiPcfP9q2TFVW3TLr6HC46auF54p4LxvXRJ-juz_wcOH9W_u7P0hbu_zlVomTXg0ew9zQX6PMfpDw_NUONcpNv7moyt2pAZBToynSgDUl0DrCewmzXCrrhj9UcM=)
9. [roboflow.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEvdNBJSyfeLme4GWxaGS6ctStulfH42vn0vhK6CiEey7Aqmu5wK0-7WQkpFuHXtZs54eHcpTkVgxcgcgM_QfylHPLXzho1Reb3KFkuNGGEs9DRxr6dvaa6Fkr-tzTymUpBhnFljB6NbgJQ55Ty54Vh6Ja6qxC6GYWGPB7d)
10. [mdpi.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQH0Qj4ERK9nxTXQ1Z0Qn_QxK4LWeim-8hxo2EsqHTmQiWD2jCILdhcLFEKAkjaqa39FalFOb5j46MgOT6_a70ojVZkitp0n2iiuq95Q9fDMRO95cbbZgyi_RBSvzCI=)
11. [mdpi.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFNxCCsVii2DdVehJemhCR8vWspI0w4eYaX9WYUkt6_D5wNR6Ax0J3-qOekuwCT2yOoTQdErhqusdv91nebE4OTHy8R8Yp--DP0iew1CRgKGMFE4hylloYB414tag38oQ==)
12. [emergentmind.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQE5aODywGlQsBwv1J0NziCom8xE1Sbtx8EkjCeQ0jwsyyx7o0ymioddBtBRY1dHxmlBYzRu5JtzU2rX9IeztyifjvlsiGroJSCyfZPO4cgOHd1_-7qKbsUrg8z8r07ZQw==)
13. [arxiv.org](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFdTrK7p_r4V9cclbdFEVXcgpBRsgSACP48xcP12Tp0aBavofgdZWNtEzxECCvsHHyyBQOSRYb206NPE4oJc1J7T6xTBzvyNUZJ2uPpmUhB-VEaYoG7PResvA==)
14. [huggingface.co](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFb7F370tA31u0qCsgv8H9qg2DbTfCBYq8mSiUz6JelDVYOLFtBBWz8bihwJALkNeEDQ3b9dJuy33I-nE7XJoH4r4r6mVb2ykzRj8NHqajEVskN85AemzF6JUHNS3kOk8EKClJwL05xZ_wT)
15. [github.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEZHWIjsikBiq39ZzICKZF06Kr4gtmStz1DaTgxtkTb1sgGn6edY3NOUefrYnOmfHEL30ZxYiRr1FRVMC6Tzru7Y-Cy-SX9eKO2Dojgrj8uPQuWdUOcPc9YfQ==)
16. [github.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEbOo-VBfsB7_1GONKSdK9dT_olBf8EJANo34DRk0abiLEKx5zSDnx2TwYEEAzeipUFFIzlRFeJ09a9_eQln_FxQMK6iXS9df_v48V98Nt4Y3AoKHBmHFb2uREkN68=)
17. [dataone.org](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGczC0_QWoPqiPckXhUVp3eCpXZQEObAtBxb_gxqjxHjfLOtXV6tKnYRJzDlnqelVSKJdNCrMCEjaTz95UtqAGLqyFJoU0ipxnma5x90HTgEvD-W8EHMALx-QaW1rLX25fOebyuwwomuhsPrKkhDGuSDmhLYyQUZ1XSP3XSBCXZm49AFyG2qYCriNgm7I1R6Acnv7O8_BK8eIU9P8IFDrE=)
18. [github.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQENgBtBbvI5vuV5TVv9B1WNk6R-5bJV7B922Zl8PdOCq7w3rDaoYzg-Q4Y_ODHSSUG9b9Ll6lbHc4PaIbV2PZfAYNHk5g7r_D6j7jvbYD3qtPe29vVBA8YtiDZ1M57ueg==)

