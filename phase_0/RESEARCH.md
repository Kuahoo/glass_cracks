# RESEARCH.md — Glass / Glass-Curtain-Wall Crack Detection R&D

**Research snapshot:** 2026-09-16  
**Objective:** establish the technical basis for a founding-engineer program that detects and quantifies visible cracks in architectural glass and glass curtain walls using computer vision / machine learning, while correctly separating visible-defect detection from structural-safety assessment and hidden curtain-wall attachment defects.

---

## 0. Executive result

The best scalable architecture is **not** a single-frame crack segmenter. The strongest engineering direction is a **tiered, physics-guided, multi-view RGB inspection system**:

1. capture a short calibrated image burst/video of each pane from slightly different viewpoints;
2. localize the glass/pane and estimate its plane;
3. register the pane across frames with a plane homography;
4. segment crack candidates with an edge-aware, parameter-efficient model;
5. require multi-view consistency so surface-tied cracks survive while viewpoint-dependent reflections/transmissions are suppressed;
6. estimate geometry only when ground-sampling distance (GSD) supports it;
7. quantify uncertainty and automatically request/reacquire better views when evidence is weak;
8. escalate a small minority of high-risk/ambiguous cases to polarization/RGB-NIR or dedicated nondestructive testing (NDT).

The decisive moat is likely to be **the data/validation program**, not a novel backbone alone. The most directly relevant published glass-curtain crack dataset found in this review contains only **385 high-resolution images from two Shenzhen buildings**, despite strong reported in-domain segmentation numbers. That is insufficient evidence for deployment across buildings, coatings, glazing systems, climates, reflection conditions, cameras, cities, and defect morphologies.

**Final panel result after three design waves: 8.4/10 — PASS.**  
The score is intentionally below 9 because: (a) a new glass-specific, building-disjoint dataset is still required; (b) ordinary exterior RGB cannot directly measure crack depth or hidden sealant integrity; and (c) sub-millimeter crack-width metrology is resolution-limited at practical UAV stand-off distances.

---

# 1. Research protocol and independent R&D ownership

The work is organized as independent research lanes. Each lane owns only its section and score history.

| Lane | Owner role | Scope |
|---|---|---|
| A | Glass / fracture-mechanics R&D | glass strength, fracture, flaw physics, thermal/impact/NiS mechanisms |
| B | Curtain-wall systems R&D | glazing systems, frame interfaces, failure taxonomy, inspection context |
| C | Crack-computer-vision R&D | crack segmentation, foundation models, thin-structure metrics |
| D | Transparent-optics R&D | glass localization, reflection/transmission, polarization, NIR, motion cues |
| E | NDT / sensing R&D | acoustics, vibrometry, DIC, photoelasticity, active/controlled optical methods |
| F | Data / benchmarking R&D | datasets, annotation, splits, metrics, generalization design |
| G | Field acquisition / metrology R&D | UAV/camera geometry, blur, GSD, calibration, multi-view capture |
| H | Systems / deployment R&D | inference pipeline, uncertainty, escalation, economics, QA/MLOps |

### Scoring convention

Scores are engineering judgments, not experimentally proven performance. Every score is 0–10 and is deliberately conservative.

- **Cost**: higher means lower cost to deploy/operate.
- **Feasibility**: higher means fewer unresolved engineering blockers.
- **Reliability**: robustness to realistic facade conditions and failure modes.
- **Accuracy**: expected ability to produce the claimed output, not benchmark accuracy on an unrelated dataset.
- **Mean**: arithmetic mean of the four dimensions.

A technique can score below 7 and remain valuable as a specialist escalation tool. The loop requirement is applied to the **final working solution and its critics**; weak methods are not artificially rescored upward just to pass.

---

# PART I — RESEARCH FIRST

# 2. Lane A — Glass and fracture mechanics

**Owner:** Glass / fracture-mechanics R&D agent  
**Do not edit from other lanes.**

## 2.1 What “glass strength” actually means

Architectural soda-lime-silica glass is brittle and flaw-controlled. Its intrinsic atomic-scale strength is far above its practical strength; real glass fails from surface/edge flaws created during manufacturing, handling, installation, cleaning, weathering, impact, and service. A 2023 proof-of-concept NDT paper summarizes conventional float-glass fracture toughness as roughly **0.6–0.8 MPa√m** and explicitly notes that humidity/stress-corrosion can drive subcritical crack growth under sustained loading.

The core Mode-I relation is

\[
K_I = Y\,\sigma\sqrt{\pi a}
\]

where:

- \(K_I\) = stress-intensity factor,
- \(Y\) = geometry factor,
- \(\sigma\) = tensile stress normal to the crack,
- \(a\) = effective flaw/crack size.

Fast fracture is expected when

\[
K_I \ge K_{IC}.
\]

Equivalently, the critical tensile stress is

\[
\sigma_c = \frac{K_{IC}}{Y\sqrt{\pi a}}.
\]

Using \(K_{IC}=0.75\) MPa√m and \(Y=1.12\) only as an illustrative edge-crack calculation:

| effective flaw size `a` | illustrative `sigma_c` |
|---:|---:|
| 0.05 mm | 53.4 MPa |
| 0.10 mm | 37.8 MPa |
| 0.50 mm | 16.9 MPa |
| 1.00 mm | 11.9 MPa |
| 2.00 mm | 8.45 MPa |

**Engineering consequence:** image-visible surface length/width is not enough to infer residual strength. Crack **depth**, orientation, location, current tensile stress, glass heat treatment, edge condition, humidity/load duration, and geometry all matter. A CV model should therefore output **inspection evidence and triage**, not claim an autonomous structural certification from RGB alone.

### Weibull / weakest-link behavior

Glass strength is statistical because the population and severity of surface flaws are statistical. ISO 1288-1 explicitly describes bending-strength tests as a basis for statistical evaluation of glass strength. This means a product should preserve distributions/uncertainty rather than collapsing every visible defect to a deterministic “remaining strength.” Larger stressed areas also have more opportunity to contain a critical flaw.

## 2.2 Annealed, heat-strengthened, fully tempered, laminated, IGU

The product must record glass construction when known because failure morphology and consequence differ.

- **Annealed:** no intentional tempering stress field; can retain large sharp fragments and can sustain a crack without immediate whole-pane fragmentation.
- **Heat-strengthened (HS):** induced surface compression/interior tension lower than fully tempered.
- **Fully tempered (FT):** higher stored residual stress; a sufficiently deep surface crack that penetrates the compressive zone can trigger rapid fragmentation.
- **Laminated:** plies bonded to an interlayer; post-breakage retention depends on laminate design/interlayer and support.
- **Insulating glass unit (IGU):** multiple lites separated by a sealed cavity; exterior vision may not identify which lite carries a defect without parallax/focus or inside access.
- **Coated/spandrel/fritted:** coatings and ceramic frit materially change image appearance and may hide or mimic defects.

ASTM C1048-25 is the active specification for heat-strengthened and fully tempered flat glass used in general building construction. ASTM C1172-24e1 covers laminated architectural flat glass. ASTM C1279-23 covers nondestructive photoelastic measurement of edge/surface stresses in annealed, heat-strengthened, and fully tempered flat glass.

## 2.3 Crack / breakage mechanisms relevant to vision

### Edge and handling damage

Edges are high-value inspection zones. Chips, shells, scratches, frame contact, hard-particle damage, or insufficient edge clearance can provide severe flaw origins. Even when a pane is visually intact, edge damage can dominate strength.

### Thermal-stress breakage

Differential temperature creates tensile stress. Typical forensic guidance describes thermal cracks as often originating at a glass edge and initially propagating approximately perpendicular to it before later branching. This is useful as a **diagnostic clue**, not a guaranteed classifier.

### Impact / point load

Impact can create radiating and circumferential crack patterns, local cones/chips, or a direct origin. On fully tempered glass, penetration through the surface compression zone can release stored strain energy and fragment the pane.

### Nickel sulfide (NiS) and other inclusions

NiS inclusions are a recognized mechanism for delayed spontaneous breakage of thermally toughened glass. Phase transformation/volume change around an inclusion can create a critical local stress state. Critically, the frequently cited “butterfly” fracture origin is **not uniquely diagnostic**: published glass-fracture work notes that a butterfly morphology is a consequence of the tempered stress field and can occur with other origins. Root-cause claims should require origin recovery/microscopy or other forensic evidence.

### Scratches vs cracks

Scratches are not merely cosmetic. A 2026 Engineering Structures study developed a vision/non-contact framework connecting scratch depth/location to residual wind-load capacity in hidden-frame curtain-wall glass. For our data taxonomy, “scratch” must be a separate labeled class with possible transition to crack-like damage; a binary crack/not-crack dataset will confuse safety-relevant surface damage with harmless linear patterns.

## 2.4 What CV can and cannot infer from fracture physics

**CV can plausibly estimate:**

- visible centerline and connected components;
- surface-projected length;
- orientation and branching topology;
- edge origin candidate if visible;
- pane-relative position;
- width only when optical resolution/calibration supports it;
- temporal growth if repeat inspections are registered;
- pattern descriptors useful for forensic triage.

**CV alone generally cannot establish:**

- subsurface crack depth;
- residual stress field;
- whether an inclusion is NiS;
- actual local applied stress;
- hidden structural-silicone debonding;
- certified residual load capacity without validated mechanics + material inputs.

### Lane A score history

| Iteration | Cost | Feasibility | Reliability | Accuracy | Mean | Critic note |
|---|---:|---:|---:|---:|---:|---|
| A0: infer structural safety from RGB crack mask | 9.0 | 7.0 | 3.5 | 3.5 | 5.75 | **FAIL** — missing depth/stress/material state |
| A1: geometry + morphology + conservative risk features | 8.8 | 8.8 | 7.5 | 7.0 | 8.03 | PASS for triage, not certification |
| A2: triage + uncertainty + NDT escalation | 8.2 | 8.5 | 8.5 | 8.0 | 8.30 | **PASS** |

**Weakest unresolved issue:** crack depth and actual stress state remain unobservable from ordinary exterior RGB.

### Key Lane A sources

1. Karlsson et al., “Non-destructive strength testing of microindented float glass by a nonlinear acoustic method,” *Construction and Building Materials* 391 (2023), 131748. https://doi.org/10.1016/j.conbuildmat.2023.131748
2. ISO 1288-1:2016, *Glass in building — Determination of the bending strength of glass — Part 1*. https://www.iso.org/standard/62743.html
3. ASTM C1048-25, *Heat-Strengthened and Fully Tempered Flat Glass*. https://store.astm.org/c1048-25.html
4. ASTM C1279-23, *Non-Destructive Photoelastic Measurement of Edge and Surface Stresses...*. https://store.astm.org/c1279-23.html
5. ASTM architectural glass standards index. https://store.astm.org/products-services/standards-and-publications/standards/glass-standards-and-ceramic-standards.html
6. Kasper, “Spontaneous cracking of thermally toughened safety glass…,” *Glass Structures & Engineering*. https://doi.org/10.1007/s40940-018-0083-8
7. NIST, *Fractography of Ceramics and Glasses* (tempered-window fractography discussion). https://www.govinfo.gov/content/pkg/GOVPUB-C13-PURL-gpo2659/pdf/GOVPUB-C13-PURL-gpo2659.pdf
8. Pan et al., “Safety performance assessment of scratched glass panels in a curtain wall system: Development of a vision-based framework,” *Engineering Structures* 348 (2026), 121822. https://doi.org/10.1016/j.engstruct.2025.121822

---

# 3. Lane B — Curtain-wall systems and failure taxonomy

**Owner:** Curtain-wall systems R&D agent  
**Do not edit from other lanes.**

## 3.1 Curtain wall is not “just a glass window”

A curtain wall is typically a **non-load-bearing building-envelope/cladding system** connected to the primary structure. NPSA guidance separates common systems into:

- stick systems;
- unitized systems;
- semi-unitized systems;
- bolt-fixed glazing.

A stick system uses mullions/transoms and field-installed glazing; unitized systems use prefabricated story-height modules. Hidden-frame / structural-silicone systems can transfer glass loads through structural sealant to a subframe. Inspection logic therefore must know whether a visible line is a crack, a mullion/transom edge, gasket, pressure plate, toggle channel, structural silicone joint, frit transition, splice, or interior feature seen through the glass.

## 3.2 Important curtain-wall defect classes

A useful inspection ontology is broader than “crack.”

### Glass-related

- surface crack;
- edge crack/chip;
- scratch / abrasion;
- full breakage / fragmented pane;
- lamination/interlayer anomaly;
- IGU seal failure / fogging / condensation;
- coating/frit damage;
- inclusion-origin fracture evidence;
- thermal/impact morphology clues.

### Support / frame / sealant-related

- structural silicone debonding;
- weather-seal failure;
- gasket displacement;
- frame distortion / joint displacement;
- fastener/anchor problems when externally visible;
- water intrusion staining;
- panel displacement.

### Confounders that a crack detector will see

- mullion/transom edges;
- gasket lines;
- reflections of overhead wires, trees, railings, facade fins;
- interior blinds and curtains;
- interior partitions and ceiling grids;
- bird-strike residues;
- dirt/water streaks;
- cleaning scratches;
- reflections of neighboring cracked/linear structures;
- facade sealant beads.

Interior blinds/curtains are especially important because the user’s phrase “glass window curtains” can also describe visual content behind a window. For the ML system they are **hard negatives**, not target defects.

## 3.3 System-level risk

A visible pane crack and hidden structural-silicone debond are separate conditions. A perfect crack model can still miss a dangerous hidden-frame attachment defect. The deployed product should therefore expose separate modules:

- **Visible glass condition**;
- **Frame/seal/gasket condition**;
- **Hidden attachment condition (when an NDT module is used)**;
- **Inspection completeness / confidence**.

This avoids the dangerous UI state “no crack found = facade safe.”

### Lane B score history

| Iteration | Cost | Feasibility | Reliability | Accuracy | Mean | Critic note |
|---|---:|---:|---:|---:|---:|---|
| B0: one binary “glass damaged” class | 9.0 | 9.0 | 5.0 | 5.0 | 7.00 | Borderline; poor diagnosis/confounder handling |
| B1: explicit defect + confounder ontology | 8.5 | 8.5 | 8.0 | 8.0 | 8.25 | PASS |
| B2: separate visible vs hidden-condition modules | 8.0 | 8.2 | 8.8 | 8.3 | 8.33 | **PASS** |

### Key Lane B sources

1. UK NPSA, “Introduction To Glass Curtain Wall Systems.” https://www.npsa.gov.uk/building-protection/windows-glazed-facades/introduction-glass-curtain-wall-systems
2. Hong et al., “Debonding Detection in Hidden Frame Supported Glass Curtain Walls Using the Nonlinear Ultrasonic Modulation Method…,” *Sensors* 18 (2018), 2094. https://doi.org/10.3390/s18072094
3. Zhou & Tiong, “Defects inspection system for building facades using drones and deep learning method,” *Expert Systems with Applications* 298 (2026), 129715. https://doi.org/10.1016/j.eswa.2025.129715

---

# 4. Lane C — Crack computer vision and machine learning

**Owner:** Crack-CV R&D agent  
**Do not edit from other lanes.**

## 4.1 Direct glass-curtain crack literature

### GlassCurtainCrackNet — Huang & Liu

The most directly relevant model found is “An iterative up-sampling convolutional neural network for glass curtain crack detection using unmanned aerial vehicles” (*Journal of Building Engineering*, DOI 10.1016/j.jobe.2024.110814).

**Method**

- modified ResNet feature extractor;
- region proposal + RoIAlign;
- iterative up-sampling crack mask module;
- UAV imaging with a planned zigzag flight path.

**Reported results**

- accuracy: **99.89%**;
- precision: **91.1%**;
- recall: **72.1%**;
- F1: **79.7%**;
- mIoU: **78.7%**.

The authors compare against SOLO, Swin, Mask R-CNN, and ConvNeXt V2.

**Critical interpretation**

99.89% pixel accuracy is not the number to optimize for a thin-crack task because background pixels dominate. The much lower 72.1% recall is more safety-relevant: approximately 28% of positive pixels/events under that evaluation definition are missed. The dataset is also narrow (see Lane F).

### GlassCurtainCrackDeblurNet — motion blur

Huang & Liu’s 2024 Sensors paper addresses UAV motion blur with a GAN and enhanced U-shaped Transformer. Its glass-curtain deblur dataset contains **4,141 clear/blurred pairs**: 2,757 train / 1,384 test; 2,899 without cracks, 828 simple cracks, 414 complex cracks. Blur was synthesized from sequences of clear high-frame-rate frames and manually screened.

**Takeaway:** acquisition quality is a first-class model input. Deblurring can help, but a system should first minimize blur physically with shutter speed, stabilization, standoff/velocity control, and capture redundancy. Synthetic blur is useful but cannot fully represent rolling-shutter, autofocus, vibration, compression, rain, glare, and prop-wash effects.

### Adversarial defense

A 2026 follow-up proposes a GAN/attention adversarial purification network evaluated on the GCC dataset. Security robustness is useful for a production system, but natural domain shift, reflections, coatings, unseen buildings, motion blur, weather, and optical resolution are higher-priority risks than adversarial-image attacks for the initial product.

## 4.2 General crack segmentation literature that transfers

### DeepCrack

A classical hierarchical FCN/DSN crack-segmentation architecture. Its dataset has 537 manually annotated images; the paper reports mIoU 85.9 and best F-score 86.5. Valuable historical baseline, but not transparent glass.

### OmniCrack30k

CVPRW 2024 aggregated **30k samples from more than 20 datasets** across asphalt, ceramic, concrete, masonry, steel and other materials. The paper reports nnU-Net mean `clIoU_4px` of 64%, more than 10 points above other evaluated approaches. This benchmark is valuable for pretraining/generalization research but still does not substitute for glass-curtain data.

### CrackSAM (2024)

Uses parameter-efficient fine tuning (adapter/LoRA) of Segment Anything for civil-infrastructure crack segmentation and validates a lightweight knowledge-distillation route. The paper explicitly reports improved robustness/generalization relative to 12 comparison methods across four datasets.

### SECrackSeg (2025)

Combines:

- frozen SAM2 214M-parameter backbone;
- lightweight S-Adapter;
- multi-scale dilated convolution;
- hybrid upsampling;
- edge-aware attention;
- weighted BCE + weighted IoU and multi-granularity supervision.

On CFD the paper reports precision 0.895, recall 0.938, F1 0.915, mIoU 0.854. These numbers **must not be transferred to glass**; they demonstrate an architecture pattern valuable for thin structures and low-data adaptation.

### Segment Any Crack / SAC (2026)

SAC tunes only normalization parameters of SAM. On OmniCrack30k it reports F1 **61.22%**, IoU **44.13%**, while showing favorable zero-shot stability and low trainable-parameter overhead. This is an important warning: generalization benchmarks are much harder than narrow in-domain datasets.

### YOLO11 + SAM semi-supervision (2025)

A two-stage system uses YOLO11/Oriented YOLO11 for region-level labels and SAM for segmentation. It reports average Dice ~0.7 and claims region labels can reduce annotation work by up to **30×**. This is highly relevant to building a proprietary dataset: obtain coarse boxes/polygons cheaply, use model-assisted masks, then human-correct the thin centerline/boundary.

## 4.3 Model design lessons

1. **Do not start by inventing a new backbone.** Data, optics, view consistency, and evaluation are larger unknowns.
2. **Use parameter-efficient adaptation** first: SAM2 adapter, LoRA, norm tuning, or a strong CNN/Transformer hybrid.
3. **Thin-line objectives matter:** Dice/IoU alone reward thick masks and can penalize topology oddly. Add centerline/topology-aware metrics/loss.
4. **Keep a lightweight deployable student.** A large foundation model can serve as teacher/label assistant; deployment may use a distilled U-Net/SegFormer/YOLO-seg-scale model.
5. **Use image-level / pane-level detection in addition to pixels.** A safety workflow cares about “did we miss a defective pane?” more than aggregate background accuracy.

### Lane C score history

| Iteration | Cost | Feasibility | Reliability | Accuracy | Mean | Critic note |
|---|---:|---:|---:|---:|---:|---|
| C0: single-frame off-the-shelf crack model | 9.5 | 9.5 | 4.5 | 5.5 | 7.25 | Cheap but glass-domain false positives/misses |
| C1: glass fine-tune + edge-aware SAM2 adapter | 8.8 | 9.0 | 6.8 | 7.8 | 8.10 | In-domain pass; cross-building risk remains |
| C2: add multi-view consistency + uncertainty | 8.3 | 8.3 | 8.2 | 8.5 | 8.33 | **PASS** |

### Key Lane C sources

1. Huang & Liu, *Journal of Building Engineering* (2024), GlassCurtainCrackNet. https://doi.org/10.1016/j.jobe.2024.110814
2. Huang & Liu, *Sensors* 24 (2024), GlassCurtainCrackDeblurNet. https://doi.org/10.3390/s24237713
3. GlassCurtainCrackAdversarialDefense (2026). https://www.sciencedirect.com/science/article/pii/S0263224126002538
4. Liu et al., DeepCrack. https://doi.org/10.1016/j.neucom.2019.01.036
5. Benz & Rodehorst, OmniCrack30k, CVPRW 2024. https://openaccess.thecvf.com/content/CVPR2024W/VAND/html/Benz_OmniCrack30k_A_Benchmark_for_Crack_Segmentation_and_the_Reasonable_Effectiveness_CVPRW_2024_paper.html
6. CrackSAM, *Construction and Building Materials* 431 (2024), 136573. https://doi.org/10.1016/j.conbuildmat.2024.136573
7. Chen et al., SECrackSeg, *Sensors* 25 (2025), 2642. https://doi.org/10.3390/s25092642
8. Rostami et al., Segment Any Crack, *Journal of Computing in Civil Engineering* 40(3) (2026). https://doi.org/10.1061/JCCEE5.CPENG-7090
9. Kuttuva & Prawin, YOLO11-SAM structural defect segmentation. https://doi.org/10.1016/j.autcon.2025.106709

---

# 5. Lane D — Transparent optics, glass localization, reflections and multimodality

**Owner:** Transparent-optics R&D agent  
**Do not edit from other lanes.**

## 5.1 Why glass is uniquely difficult

Concrete cracks are dark geometric structures on an opaque textured substrate. Architectural glass can show a superposition of:

- transmitted scene behind the glass;
- reflected scene in front of the glass;
- actual surface defect;
- frame/interior lines;
- coating/tint/frit;
- sky/specular saturation.

The defect may have weak intrinsic contrast while a reflected tree branch or wire has much stronger contrast. This is why “train on concrete cracks, deploy on glass” is not credible.

## 5.2 Glass-surface segmentation literature

### Trans10K / TransLab

Trans10K contains **10,428 real images** of transparent objects with manual annotations. TransLab uses boundary awareness because transparent object interiors inherit surrounding appearance. It is useful for pretraining transparent-boundary understanding, although its object distribution is not specifically high-rise curtain walls.

### Video glass-surface detection

The 2024 VGSD work built **VGSD-D: 19,166 labeled frames from 297 videos**. The key observation is directly useful to our problem: reflection appearance changes with camera motion, providing temporal evidence for the presence of a glass surface.

AAAI 2026 MVGD-Net advances this with **312 glass scenarios / 19,268 frames** and explicitly uses motion inconsistency: reflected/transmitted objects behave as if farther from the pane, so their apparent motion differs from content on the local physical plane.

### MSNet (AAAI 2026)

MSNet decomposes glass imagery into transmission, reflection and surrounding semantics. Its conceptual relation is that reflection tends to resemble surrounding content while differing from transmission. This supports building a crack pipeline that first models the glass optical mixture rather than asking a crack network to learn every reflection as a negative by brute force.

## 5.3 NIR

NRGlassNet (2024) uses paired RGB and near-infrared. The authors observed that glass reflections are often suppressed in NIR relative to RGB and built **RNGD with 1,378 RGB-NIR pairs**. A dual Swin-Transformer network uses the difference in reflection intensity to detect glass surfaces.

**Field caveat:** adding NIR increases hardware, synchronization, calibration, exposure and payload complexity. It is best considered an optional ambiguity-reduction tier until field tests prove that the reflection benefit justifies operational cost.

## 5.4 Polarization

A 2024 Optics Communications paper evaluates 12 polarization parameters and an edge-enhanced, extensible segmentation network for transparent objects. Polarization can separate/specify reflected components because Fresnel reflection changes polarization state with angle.

**Field caveats:**

- solar illumination and viewing angle vary;
- coatings alter polarization;
- a rotatable polarizer or polarization camera adds hardware/workflow;
- a single CPL setting can remove useful crack contrast as well as glare;
- high-rise drone payload and control complexity matter.

Therefore polarization is high-value for controlled/close-range verification and as an optional capture mode, not mandatory baseline hardware.

## 5.5 Physics opportunity: multi-view planar consistency

A surface crack lies approximately on the glass plane. Under a pinhole-camera model, points on a plane map between two camera views by a homography:

\[
\mathbf{x}' \sim \mathbf{H}\mathbf{x}.
\]

If frames are warped to a reference pane using \(H_{t\rightarrow0}\), a real surface-tied crack should become spatially consistent (subject to blur/noise). Reflections and transmitted objects generally originate on different virtual/physical depth layers and are viewpoint dependent, so the same pane-plane homography will not make them perfectly stationary.

This is not merely an intuition: video glass-detection research explicitly exploits reflection/motion inconsistency. The new contribution for crack inspection is to use that optical fact as a **false-positive rejection mechanism for crack evidence**.

For per-frame crack probabilities \(P_t\), define registered consensus:

\[
C(p)=\frac{1}{N}\sum_{t=1}^{N} P_t\!\left(H_{t\rightarrow0}^{-1}p\right)
\]

and temporal variance:

\[
V(p)=\operatorname{Var}_t\left[P_t\!\left(H_{t\rightarrow0}^{-1}p\right)\right].
\]

A strong surface crack hypothesis should tend toward **high consensus / low variance**. This is a testable hypothesis, not yet a claimed result.

### Lane D score history

| Iteration | Cost | Feasibility | Reliability | Accuracy | Mean | Critic note |
|---|---:|---:|---:|---:|---:|---|
| D0: RGB single-frame glass localization | 9.2 | 9.2 | 6.0 | 6.5 | 7.73 | Reflection ambiguity remains |
| D1: mandatory RGB+NIR/polarization | 6.0 | 6.8 | 8.0 | 8.2 | 7.25 | Pass but payload/workflow hurts scale |
| D2: RGB multi-view baseline; NIR/polarization on demand | 8.5 | 8.3 | 8.3 | 8.3 | 8.35 | **PASS** |

### Key Lane D sources

1. Xie et al., “Segmenting Transparent Objects in the Wild” (Trans10K / TransLab). https://arxiv.org/abs/2003.13948
2. Yan et al., “NRGlassNet: Glass surface detection from visible and near-infrared image pairs,” *Knowledge-Based Systems* 294 (2024), 111722. https://doi.org/10.1016/j.knosys.2024.111722
3. “Transparent objects segmentation based on polarization imaging and deep learning,” *Optics Communications* 555 (2024), 130246. https://doi.org/10.1016/j.optcom.2023.130246
4. Lu et al., “MVGD-Net: A Novel Motion-aware Video Glass Surface Detection Method,” AAAI 2026. https://doi.org/10.1609/aaai.v40i9.37700
5. Cheng et al., “Multi-Semantic Modeling for Glass Surface Detection in the Wild,” AAAI 2026. https://doi.org/10.1609/aaai.v40i5.37322

---

# 6. Lane E — NDT, controlled optics and specialist sensing

**Owner:** NDT / sensing R&D agent  
**Do not edit from other lanes.**

The purpose of this lane is not to replace scalable RGB inspection. It identifies sensors that solve **different observability problems** and can validate or escalate the vision system.

## 6.1 Industrial dark-field / bright-field inspection

Commercial flat-glass inspection systems use controlled illumination and cameras to expose scratches, inclusions, coating defects and other imperfections. Vendor examples include Dark Field Technologies’ laser/camera systems. Industrial line systems can combine dark-field and bright-field channels because the geometry and lighting are fixed.

**Strength:** excellent defect contrast and repeatability in factory/bench conditions.  
**Weakness:** difficult to reproduce on a large sunlit exterior facade because the system cannot surround each pane with controlled illumination.

**Decision:** use controlled dark-field/bright-field rigs as a **ground-truth / laboratory characterization tool**, not the primary high-rise sensor.

## 6.2 Deflectometry

Deflectometry projects a known pattern and measures its distortion in a specular surface. MVTec documents robust detection of scratches, dents, cracks and other specular-surface defects using a DLP/projector plus industrial camera.

**Strength:** highly sensitive to slope/shape changes.  
**Weakness:** requires known projected pattern/geometry and sufficient reflection; difficult to illuminate a distant facade in daylight.

**Decision:** useful for close-range robotic/bench inspection and dataset truthing.

## 6.3 Photoelastic stress measurement

ASTM C1279-23 covers nondestructive photoelastic edge/surface stress measurement in architectural glass. Full-field digital photoelastic approaches can map residual stress in tempered glass.

**Strength:** directly related to stress state rather than only visible damage.  
**Weakness:** optical access/polarization geometry and transmitted-light requirements complicate exterior in-service use, especially for IGUs/coated glass.

**Decision:** laboratory or specialist close-range tier.

## 6.4 Nonlinear acoustic wave inspection

The 2023 float-glass NDT proof-of-concept used nonlinear acoustic response of 10×10 cm annealed float-glass specimens with Vickers-induced defects. Its normalized defect value correlated with indentation severity and showed a linear trend with failure stress, but the study is explicitly an initial proof of concept.

For hidden-frame curtain-wall structural sealant, 2018 research used piezoceramic transducers and nonlinear ultrasonic modulation on a 200×200×24 mm specimen with progressive debond lengths. The nonlinear coefficient increased with debond length and the reported MSE improved by 41% versus nonlinear harmonics under their test configuration.

**Decision:** powerful specialist NDT, but contact transducers and coupling/workflow make it unsuitable for every pane in a high-throughput exterior survey.

## 6.5 Laser Doppler vibrometry (LDV)

An August 2026 *Mechanical Systems and Signal Processing* paper demonstrates **UAV-mounted LDV** plus remote excitation and physics/data compensation for hidden-frame structural-adhesive debonding. It validates full-scale progressive debonding and defines panel-specific equivalent-degradation risk states in that experiment.

This is one of the most important adjacent technologies because it is remote and addresses a hidden defect that RGB cannot see.

**Decision:** integrate as an optional specialist service / future sensor module for hidden-frame systems, not as the MVP crack sensor.

## 6.6 3D digital image correlation / vibration

2025–2026 work uses 3D-DIC and video motion magnification to extract curtain-wall modal shapes and localize structural-adhesive damage. Full-field motion is attractive, but controlled excitation, stereo calibration, stable imaging, and processing complexity limit its baseline scalability.

### Lane E score history

| Technique / iteration | Cost | Feasibility | Reliability | Accuracy | Mean | Decision |
|---|---:|---:|---:|---:|---:|---|
| Dark/bright-field as facade-wide primary | 4.0 | 4.5 | 8.5 | 9.0 | 6.50 | **REJECT primary**; retain lab/close-range |
| Deflectometry as facade-wide primary | 3.5 | 4.0 | 8.5 | 9.0 | 6.25 | **REJECT primary** |
| Contact nonlinear ultrasound on every pane | 3.0 | 4.0 | 8.2 | 8.0 | 5.80 | **REJECT primary** |
| UAV-LDV for hidden debonding escalation | 5.5 | 7.0 | 8.0 | 8.0 | 7.13 | PASS specialist tier |
| Tiered NDT only after visual risk trigger | 7.0 | 8.0 | 8.7 | 8.5 | 8.05 | **PASS** |

### Key Lane E sources

1. Dark Field Technologies, glass/coating inspection systems. https://darkfield.com/industries/glass/
2. MVTec HALCON deflectometry. https://www.mvtec.com/knowledge-base/developers-corner/article/inspection-of-specular-surfaces-with-deflectometry-in-halcon
3. ASTM C1279-23. https://store.astm.org/c1279-23.html
4. Karlsson et al. nonlinear acoustic float-glass NDT. https://doi.org/10.1016/j.conbuildmat.2023.131748
5. Hong et al. hidden-frame nonlinear ultrasonic modulation. https://doi.org/10.3390/s18072094
6. Hong et al. nonlinear ultrasonic Lamb wave adhesion-strength detection. https://doi.org/10.1061/(ASCE)AS.1943-5525.0000870
7. Guo et al., “Remote vibration-based debonding identification in hidden-frame glass curtain walls using a UAV-mounted LDV system,” *Mechanical Systems and Signal Processing* 257 (2026), 114638. https://doi.org/10.1016/j.ymssp.2026.114638

---

# 7. Lane F — Datasets, labels and benchmarking

**Owner:** Data / benchmarking R&D agent  
**Do not edit from other lanes.**

## 7.1 Directly relevant datasets

### GCC — GlassCurtainCrack

The 2026 adversarial-defense paper describes the original GCC dataset as:

- **385 images**;
- resolution **5472×3648**;
- DJI Mavic 2 Pro;
- Osmanthus Building + CNA Kate Apartments;
- Shenzhen.

This is the most important dataset fact in the review. It means a strong published result is still based on two buildings in one city, with one principal camera family and a small number of high-resolution images.

### GlassCurtainCrackDeblur dataset

- 4,141 paired clear/blurred images;
- 2,757 train / 1,384 test;
- 2,899 no-crack, 828 simple-crack, 414 complex-crack pairs;
- same general building context as the related Shenzhen work;
- blur generated by combining video frames and manually screening.

Useful for blur restoration but not a geographically diverse crack benchmark.

## 7.2 Generic crack datasets

### OmniCrack30k

- 30k samples;
- >20 source datasets;
- ~9 billion total pixels;
- multiple materials.

Use for pretraining and measuring generic crack generalization, not glass-domain acceptance.

### DeepCrack

- 537 manually annotated RGB images;
- pixel segmentation masks.

### CrackForest / CFD

- 118 color images;
- 480×320;
- smartphone road imagery;
- includes hard negatives such as stains/markings/shadows.

### Crack500

- ~500 high-resolution road crack images, often cropped for model training.

### SDNET2018

- 230 source images split into **56,092** 256×256 patches;
- 8,484 crack / 47,608 non-crack;
- bridge decks, walls, pavements;
- reported crack widths 0.06–25 mm.

Important warning: the “56k images” are patches from only 230 source images. Dataset unit-of-independence must always be reported.

## 7.3 Transparent / glass-surface datasets

- **Trans10K:** 10,428 real transparent-object images with masks.
- **VGSD-D:** 19,166 labeled frames from 297 glass videos.
- **MVGD-Net dataset:** 19,268 frames from 312 scenarios.
- **RNGD (NRGlassNet):** 1,378 aligned RGB-NIR pairs.

These are valuable for learning “where is glass?” and reflection cues, but they do not provide architectural-glass crack ground truth.

## 7.4 Required proprietary dataset: GlassCrack-X

Working name only; do not optimize branding yet.

### Sampling unit

The fundamental split unit is **building/site**, not image. Frames from the same pane, video, flight, facade, or building must never leak between train and test.

### Minimum acquisition axes

Collect across:

- multiple cities/climates;
- low-rise and high-rise;
- stick/unitized/semi-unitized/structural-glazed systems;
- annealed/HS/FT/laminated/IGU when known;
- clear/tinted/low-E/reflective/fritted/spandrel glass;
- clean/dirty/wet/dusty panes;
- morning/noon/evening/backlight/cloudy;
- sky, vegetation, neighboring buildings and traffic reflections;
- frontal/oblique views;
- multiple distances and focal lengths;
- at least two independent camera families;
- UAV + handheld/robotic close-range where safe;
- pre- and post-cleaning on selected panes;
- intact and damaged panes.

### Required labels per pane/defect

- pane polygon / glass region;
- crack pixel mask;
- crack skeleton/centerline;
- visible endpoints and branch nodes;
- pane-relative crack coordinates;
- edge-origin flag/candidate;
- visible projected length;
- width trace only where calibrated/resolved;
- class: crack / scratch / chip / breakage / seal line / contaminant / reflection confounder / unknown;
- confidence / annotator disagreement;
- inspection distance, focal length, resolution, shutter/ISO/aperture when available;
- view angle estimate;
- blur score / exposure saturation;
- glass and system metadata when known;
- forensic cause only if independently established—not guessed from appearance.

### Hard-negative set

Intentionally oversample:

- reflected wires/tree branches;
- reflected facade fins/railings;
- blinds and curtains behind glass;
- interior partition seams;
- window-cleaning streaks;
- scratches;
- gaskets/mullions/transoms;
- silicone lines;
- bird strike residue;
- water streaks/rain droplets;
- seal failure haze;
- coating defects;
- edge frit patterns;
- camera compression/rolling-shutter artifacts.

### Ground truth hierarchy

1. **Routine label:** trained human annotation from calibrated imagery.
2. **Expert label:** facade/glass engineer review of positives and hard negatives.
3. **Metrology subset:** microscope/macro/controlled dark-field or structured optical measurement.
4. **Forensic subset:** removed panes or known-cause specimens, when available.

## 7.5 Evaluation metrics

Do not use pixel accuracy as a primary KPI.

### Pixel / shape

- precision, recall, F1;
- IoU / Dice;
- centerline IoU or `clDice`/skeleton F1;
- boundary F-score;
- Hausdorff/Chamfer distance where meaningful.

### Instance / pane

- crack-instance recall;
- defective-pane recall;
- false positives per 100 panes;
- missed-defect count by severity bucket;
- precision-recall curves under a fixed false-alarm budget.

### Metrology

- projected length error;
- width MAE by true width bin **only when ≥ the resolution threshold**;
- endpoint / branch-node localization error.

### Robustness slices

Report every major metric vs:

- building/site;
- city;
- glass type/coating;
- reflection severity;
- view angle;
- GSD/distance;
- blur;
- illumination/weather;
- defect width/contrast;
- camera.

### Calibration / uncertainty

- expected calibration error (ECE);
- Brier score or negative log likelihood;
- selective-risk curve: performance after abstaining on the most uncertain cases;
- reacquisition success rate.

## 7.6 Acceptance test design

A model cannot be promoted on a random image split. Recommended gates:

1. **Building-disjoint holdout** — never seen during training/tuning.
2. **City-disjoint holdout** — geographic/domain shift.
3. **Camera-disjoint holdout** — hardware transfer.
4. **Glass/coating stress test** — reflective/tinted/fritted subsets.
5. **Hard-negative challenge set.**
6. **Repeatability:** same panes on separate days/lighting.
7. **Field operator study:** capture failures and reacquisition rate.

### Lane F score history

| Iteration | Cost | Feasibility | Reliability | Accuracy | Mean | Critic note |
|---|---:|---:|---:|---:|---:|---|
| F0: rely on GCC + generic crack data | 9.0 | 9.0 | 4.0 | 5.5 | 6.88 | **FAIL** — insufficient domain coverage |
| F1: proprietary multi-building glass dataset | 7.0 | 8.0 | 8.0 | 8.2 | 7.80 | PASS, labeling cost high |
| F2: building-disjoint + assisted labels + metrology subset | 7.5 | 8.5 | 8.8 | 8.7 | 8.38 | **PASS** |

### Key Lane F sources

1. GCC details reported in 2026 follow-up: https://www.sciencedirect.com/science/article/pii/S0263224126002538
2. GlassCurtainCrackDeblurNet: https://doi.org/10.3390/s24237713
3. OmniCrack30k: https://openaccess.thecvf.com/content/CVPR2024W/VAND/html/Benz_OmniCrack30k_A_Benchmark_for_Crack_Segmentation_and_the_Reasonable_Effectiveness_CVPRW_2024_paper.html
4. DeepCrack: https://doi.org/10.1016/j.neucom.2019.01.036
5. SDNET2018: https://doi.org/10.1016/j.dib.2018.11.015
6. Trans10K: https://arxiv.org/abs/2003.13948
7. NRGlassNet/RNGD: https://doi.org/10.1016/j.knosys.2024.111722
8. MVGD-Net: https://doi.org/10.1609/aaai.v40i9.37700

---

# 8. Lane G — Field acquisition, optics and metrology

**Owner:** Field acquisition / metrology R&D agent  
**Do not edit from other lanes.**

## 8.1 Published camera reference and GSD reality check

The GCC work used a DJI Mavic 2 Pro. DJI’s official specifications give:

- 1-inch CMOS;
- 20 MP;
- ~77° horizontal FOV;
- 5472×3648 still images;
- minimum focusing range 1 m;
- AEB/burst capability.

Using FOV directly, an ideal fronto-parallel scene width at distance \(D\) is

\[
W(D)=2D\tan(77^\circ/2)\approx1.591D.
\]

Horizontal GSD is approximately

\[
GSD(D)=\frac{W(D)}{5472}\approx0.0002907D\;\text{m/pixel}.
\]

| distance | ideal horizontal GSD | 3-pixel physical span |
|---:|---:|---:|
| 1 m | 0.291 mm/px | 0.872 mm |
| 2 m | 0.581 mm/px | 1.744 mm |
| 3 m | 0.872 mm/px | 2.617 mm |
| 5 m | 1.454 mm/px | 4.361 mm |
| 10 m | 2.907 mm/px | 8.722 mm |

This is an approximation: obliquity, lens distortion, stabilization crop, motion blur, demosaicing, focus, compression and perspective worsen useful resolution.

A line can sometimes be **detected** even if its physical width is sub-pixel because it changes integrated pixel intensity. But reliable **width measurement** needs multiple effective pixels across the feature and a calibrated point-spread function. Do not conflate detectability with metrology.

## 8.2 Minimum acquisition contract

Every inference should know or estimate:

- camera intrinsics;
- focal length / zoom state;
- image dimensions;
- distance or scale cue;
- shutter time and exposure;
- view direction / pane normal estimate;
- blur quality;
- saturation/glare percentage;
- registration quality across views.

If these are missing, geometry outputs should degrade gracefully to “detected / location only.”

## 8.3 Multi-view burst rather than one hero image

Capture a short burst/video while translating laterally/vertically a small amount. Desired properties:

- overlap >70–80% of pane;
- enough viewpoint change for reflections to move measurably;
- not so much perspective change that registration becomes fragile;
- fast shutter;
- locked focus/exposure where possible;
- optional exposure bracket for sky/glare extremes;
- at least one near-normal view if safe and operationally feasible.

The burst is both a **signal-processing tool** and a **QA tool**: motion provides reflection cues; redundant frames provide deblur/selection; failed frames can be rejected.

## 8.4 Pane calibration / metric scale

Preferred scale sources, in order:

1. calibrated camera + range/pose + pane plane;
2. stereo or visual-inertial depth when validated;
3. known facade dimensions from drawings/BIM;
4. repeatable mullion/transom dimensions verified per system;
5. fiducial target only in close-range/manual work.

Do not infer absolute crack width from an uncalibrated screenshot.

## 8.5 Capture-quality gates

Reject/reacquire before model scoring if:

- motion blur exceeds threshold;
- autofocus missed the pane;
- crack candidate lies in clipped glare;
- glass occupies too few pixels;
- pane plane fit/registration residual is high;
- view angle is extreme;
- rain/dirt occludes the region;
- GSD cannot support requested metrology.

### Lane G score history

| Iteration | Cost | Feasibility | Reliability | Accuracy | Mean | Critic note |
|---|---:|---:|---:|---:|---:|---|
| G0: arbitrary operator photos | 9.5 | 9.5 | 4.0 | 4.5 | 6.88 | **FAIL** — uncontrolled resolution/blur/view |
| G1: calibrated flight + quality gates | 8.2 | 8.5 | 7.8 | 8.0 | 8.13 | PASS |
| G2: calibrated multi-view burst + auto reacquire | 7.8 | 8.2 | 8.8 | 8.5 | 8.33 | **PASS** |

### Key Lane G sources

1. DJI Mavic 2 official camera specifications. https://www.dji.com/mavic-2/info
2. Huang & Liu GlassCurtainCrackNet. https://doi.org/10.1016/j.jobe.2024.110814
3. Huang & Liu GlassCurtainCrackDeblurNet. https://doi.org/10.3390/s24237713
4. MVGD-Net motion cues. https://doi.org/10.1609/aaai.v40i9.37700

---

# 9. Lane H — Production system, uncertainty and deployment

**Owner:** Systems / deployment R&D agent  
**Do not edit from other lanes.**

## 9.1 Product boundary

The MVP should make one defensible claim:

> **Automated visual inspection assistance that finds, segments, localizes and tracks visible architectural-glass crack/scratch candidates, quantifies them when optical resolution permits, and routes uncertain/high-risk cases for review or specialist NDT.**

It should **not** claim autonomous structural certification.

## 9.2 Proposed modular runtime

```text
Capture
  -> Frame/pane quality gate
  -> Pane/glass segmentation + geometry
  -> Multi-view registration
  -> Candidate crack segmentation
  -> Surface-consistency fusion
  -> Crack/scratch/confounder classification
  -> Geometry + temporal comparison
  -> Uncertainty calibration
  -> Risk triage
       -> accept/report
       -> reacquire alternate view
       -> human review
       -> optional NIR/polarization
       -> optional structural NDT
  -> asset history / report
```

Every module emits its own confidence and failure code. A broken optional module must not silently corrupt another module’s output.

## 9.3 Suggested model split

### Teacher / annotation model

- SAM2-based edge-aware adapter or another high-capacity foundation segmentation model;
- trained/fine-tuned on full-resolution tiles;
- used for assisted labeling, difficult cases and offline benchmarking.

### Production student

- compact segmentation network optimized for tiled high-resolution inference;
- distillation from teacher;
- quantized where hardware supports it;
- outputs crack probability + embedding/features needed for temporal fusion.

### Glass/pane model

Separate model for pane/glass mask and facade geometry. This reduces the crack model’s search space and provides a geometric coordinate system.

## 9.4 Loss design hypothesis

A reasonable starting loss is:

\[
L = \lambda_1L_{BCE}+\lambda_2L_{Dice}+\lambda_3L_{centerline}+\lambda_4L_{boundary}+\lambda_5L_{temporal}.
\]

- `BCE`: pixel discrimination / class weighting;
- `Dice`: region overlap under heavy imbalance;
- `centerline`: topology/thin-structure preservation;
- `boundary`: crack-edge precision;
- `temporal`: registered-view consistency.

Weights must be learned/tuned on building-disjoint validation, not selected for a pretty mask on a few examples.

## 9.5 Uncertainty and abstention

For safety-relevant use, **abstention is a feature**. The model should explicitly return:

- high-confidence intact;
- high-confidence defect;
- uncertain — reacquire;
- uncertain — expert review;
- out-of-distribution / unsupported glass condition.

Confidence should be calibrated on building-disjoint data. A “selective risk” curve should show how error drops as the system abstains on uncertain cases.

## 9.6 Repeat inspection / growth monitoring

Once a pane has a stable pane-coordinate system, future inspections can register to historical observations. This enables:

- newly appeared defect detection;
- visible crack-growth estimate;
- change in branching/length;
- maintenance prioritization.

Longitudinal change may ultimately be more valuable than one-time classification because it converts perception into asset-management evidence.

## 9.7 Human factors

Reports must show:

- original image/video frame;
- pane outline;
- defect mask/centerline;
- scale/GSD and whether width is actually resolvable;
- confidence and uncertainty reason;
- prior-inspection comparison;
- suggested next inspection action;
- explicit statement when cause/depth/structural safety is not determined.

### Lane H score history

| Iteration | Cost | Feasibility | Reliability | Accuracy | Mean | Critic note |
|---|---:|---:|---:|---:|---:|---|
| H0: model -> binary alarm | 9.0 | 9.5 | 5.0 | 6.0 | 7.38 | No QA/uncertainty workflow |
| H1: modular confidence + review | 8.5 | 9.0 | 8.0 | 8.0 | 8.38 | PASS |
| H2: reacquisition + tiered escalation + asset history | 8.0 | 8.5 | 8.8 | 8.3 | 8.40 | **PASS** |

---

# PART II — WAVE 1: SYNTHESIS AND HYPOTHESES

Research above is the basis for the following ideas. These are hypotheses until experimentally validated.

# 10. Hypothesis 1 — Plane-registered multi-view crack consensus

## Idea

Use a 5–20 frame burst with small camera translation. Estimate the pane plane/homography, warp every frame to the reference pane, infer crack probability per frame, and fuse only evidence that remains stable in pane coordinates.

## Why it should work

- A physical crack is located on/very near the pane plane.
- A pane is approximately planar, so its image mapping is well represented by a homography for modest views.
- Reflections/transmitted objects are generated by different depth layers and change with viewpoint.
- Video glass-detection literature already demonstrates that motion/reflection inconsistency is useful for identifying glass.

## Falsification experiment

Dataset: minimum 50 cracked panes + 200 intact panes with strong linear reflections. Each pane receives 10–30 view sequences under at least two illumination states.

Compare:

- S0: best single frame;
- S1: averaged independent frame predictions;
- S2: homography-registered probability average;
- S3: registered average + temporal variance gate;
- S4: learned temporal fusion.

Primary endpoints:

- defective-pane recall at fixed 1 false positive / 100 panes;
- crack centerline F1;
- reflected-line false-positive rate;
- performance vs viewpoint displacement and registration residual.

**Go criterion:** at fixed defective-pane recall, ≥30% reduction in reflection-caused false positives versus best single-frame baseline on building-disjoint test.  
**Kill/modify criterion:** no consistent improvement after controlling registration quality.

**R&D score:** Cost 8.5 / Feasibility 8.5 / Reliability 8.3 / Accuracy 8.5 = **8.45 PASS**.

---

# 11. Hypothesis 2 — Active view selection beats mandatory multimodal hardware

## Idea

Instead of giving every drone a polarization/NIR rig, train a policy that recognizes ambiguous glass conditions and requests a second viewpoint/exposure. Escalate to NIR/polarization only when RGB multi-view confidence remains poor.

## Why it should work

A different viewpoint changes Fresnel reflection and scene overlap at virtually zero sensor BOM cost. Multi-view video glass work shows motion itself creates a strong glass cue. NIR/polarization is valuable but operationally expensive if forced on every pane.

## Experiment

For an ambiguity benchmark, collect RGB multi-view, RGB+polarizer angle sweep, and paired RGB-NIR on the same 100–300 panes. Measure incremental defect recall and false-positive reduction per added capture second and per added hardware dollar/payload gram.

**Go criterion:** active RGB reacquisition closes ≥70% of the performance gap between single RGB and the best multimodal configuration on reflection-heavy cases.

**R&D score:** 8.4 / 8.0 / 8.0 / 8.0 = **8.10 PASS**.

---

# 12. Hypothesis 3 — Two-stage “glass first, crack second” reduces false positives

## Idea

Infer pane/glass mask and facade geometry first. Run crack detection only inside verified glass, with explicit boundary/frame hard negatives.

## Physics / CV rationale

A crack detector otherwise spends capacity distinguishing cracks from every facade/background line. Pane masks create a semantically and geometrically constrained search region and allow pane-level metrics.

## Experiment

Compare identical crack backbone:

- full image;
- rectangular pane crop;
- exact pane mask with boundary exclusion;
- pane mask + frame/confounder auxiliary classes.

Test especially on mullions, wires, blinds, reflections and facade joints.

**R&D score:** 9.0 / 9.2 / 8.3 / 8.2 = **8.68 PASS**.

---

# 13. Hypothesis 4 — Uncertainty-triggered high-resolution tiling saves compute

## Idea

Run a low/medium-resolution pane model first; tile full resolution only where candidate evidence or uncertainty is high. Maintain overlap so thin lines at tile borders are not lost.

## Expected benefit

Most pane area is defect-free. Full-resolution segmentation everywhere is compute-inefficient. A coarse-to-fine cascade can preserve thin-crack sensitivity while reducing GPU time.

## Experiment

Compare full-resolution exhaustive tiling vs candidate tiling at equal pane-level recall. Measure GPU-seconds/pane, memory, bandwidth and missed defects by width/GSD.

**R&D score:** 9.0 / 9.0 / 8.0 / 8.2 = **8.55 PASS**.

---

# 14. Hypothesis 5 — Longitudinal change detection is a stronger product signal than one-time morphology

## Idea

Register repeat surveys to stable pane coordinates and detect new/growing defects. The model can require temporal change to promote a low-confidence ambiguous line into a maintenance alert.

## Rationale

A static reflected wire can cause a false positive today; its pane-coordinate location may change drastically with the next inspection. A physical crack remains fixed and can grow. Longitudinal persistence is therefore additional evidence.

**R&D score:** 8.5 / 8.2 / 8.8 / 8.6 = **8.53 PASS**.

---

# 15. Hypothesis 6 — Predict “resolvability” before crack width

## Idea

A dedicated metrology gate computes expected pixel width from GSD/PSF/registration confidence. If a crack is below the validated resolution regime, report detection/centerline only—no numeric width.

## Why it matters

This prevents false precision. For the published Mavic 2 Pro reference, ideal GSD already reaches ~0.58 mm/pixel at 2 m and ~1.45 mm/pixel at 5 m.

**R&D score:** 9.3 / 9.3 / 9.0 / 8.5 = **9.03 PASS**.

---

# PART III — WAVE 2: EXPERIMENTAL PROGRAM

# 16. Phase 0 — Laboratory optical truth bench

**Goal:** understand what a crack/scratch looks like under controllable variables before flying a facade.

Build a rig with replaceable panes/specimens, calibrated camera, macro reference, controlled angle, controllable background, illumination, polarizer, optional NIR, and dark-field/bright-field illumination.

Variables:

- crack/scratch width/depth class;
- annealed/tempered/laminated/IGU specimens;
- front/back surface where known;
- viewing angle;
- reflection background;
- focus/defocus;
- motion blur;
- glare/saturation;
- dirty/wet surface;
- polarizer angle;
- RGB vs NIR.

Outputs:

- optical response curves;
- minimum detectable line by GSD/SNR;
- model confusion matrix crack vs scratch vs reflection;
- calibrated resolution rule.

Pass gate: repeatable measurement protocol and no claims below validated resolvability.

---

# 17. Phase 1 — Small real-facade feasibility dataset

**Target:** roughly 50–100 buildings or as many as can be legally/safely accessed, rather than thousands of near-duplicate frames from one facade.

Prioritize diversity over raw image count.

Minimum target:

- 1,000+ panes with expert-reviewed defect status;
- 100+ true visible crack/scratch defects if obtainable;
- ≥5–10 cities/sites if program scale permits;
- repeated short bursts, not isolated images;
- high number of hard negatives.

Run experiments H1–H6 above.

**Phase 1 kill conditions:**

- surface cracks cannot be separated from reflected lines at operational distances even with multi-view capture;
- required stand-off is operationally unsafe/impractical;
- true positive prevalence is too low to collect sufficient real data and synthetic data fails to transfer;
- inspection time/pane becomes uneconomic.

---

# 18. Phase 2 — Building-disjoint benchmark

Freeze:

- label ontology;
- calibration procedure;
- primary metrics;
- training buildings;
- validation buildings;
- sealed test buildings.

No tuning on sealed test buildings.

Recommended release-quality report:

1. per-pane sensitivity/specificity;
2. false alarms / 100 panes;
3. crack-instance recall;
4. centerline F1 / clDice;
5. IoU/Dice;
6. calibration / abstention curve;
7. results by GSD, view angle, reflection, glass type, city, camera;
8. processing time and operator capture time;
9. reacquisition rate;
10. width MAE only in validated width bins.

---

# 19. Phase 3 — Safety and operational validation

Run paired inspections:

- system + facade/glass professional;
- blinded double review on positives and a random sample of negatives;
- destructive/close-range truth on available test specimens;
- optional specialist NDT on hidden-frame cases.

Measure **missed actionable defect rate**, not just pixel metrics.

---

# PART IV — CRITIC WAVES

# 20. Critic Wave 0 — single-frame RGB crack segmenter

**Candidate:** drone takes individual photos; a GlassCurtainCrackNet/SAM-style model outputs masks.

| Critic | Score | Reason |
|---|---:|---|
| CV researcher | 6.8 | feasible, but small-domain evidence and reflection ambiguity |
| Glass/fracture scientist | 5.5 | segmentation is not structural severity; depth/stress missing |
| Facade engineer | 6.0 | misses hidden sealant/frame failure; uncertain defect taxonomy |
| Field robotics engineer | 7.2 | easy capture but blur/GSD/view quality not controlled enough |
| Safety/QA reviewer | 5.8 | no calibrated abstention or reacquisition |
| CEO/scale reviewer | 8.5 | low hardware cost and scalable capture |

**Mean: 6.63 — FAIL.**  
**Weakest score: fracture/safety validity.**

### Required changes

- redefine output as visible-defect triage;
- add pane localization and quality gates;
- add uncertainty;
- explicitly split hidden attachment defects from glass cracks.

---

# 21. Critic Wave 1 — multimodal “max sensor” design

**Candidate:** RGB + polarization + NIR + crack model + glass model on every scan.

| Critic | Score | Reason |
|---|---:|---|
| CV researcher | 8.0 | richer signal |
| Glass/fracture scientist | 7.8 | still cannot directly recover depth/stress |
| Facade engineer | 8.0 | better surface evidence |
| Field robotics engineer | 6.8 | payload, synchronization, calibration, capture workflow |
| Safety/QA reviewer | 8.0 | stronger ambiguity handling |
| CEO/scale reviewer | 6.4 | BOM/maintenance/operations reduce scalability |

**Mean: 7.50 but not all critics pass — LOOP.**  
**Weakest score: cost/scalability.**

### Required changes

- move NIR/polarization to optional escalation;
- extract more information from ordinary RGB motion first;
- use active reacquisition rather than permanent sensor overhead.

---

# 22. Critic Wave 2 — tiered multi-view RGB + optional specialist escalation

**Candidate:** calibrated multi-view RGB baseline, pane plane registration, edge-aware PEFT crack model, temporal surface consistency, uncertainty/reacquisition, metrology gate, optional NIR/polarization, optional NDT.

| Critic | Score | Reason |
|---|---:|---|
| CV researcher | 8.6 | attacks domain-specific reflection failure with geometry + temporal evidence |
| Glass/fracture scientist | 8.0 | correctly limits claims and routes structural uncertainty to NDT |
| Facade engineer | 8.2 | pane/system-aware taxonomy; hidden defects separated |
| Field robotics engineer | 8.3 | RGB payload simple; burst capture operationally plausible |
| Safety/QA reviewer | 8.4 | quality gates, calibrated abstention, building-disjoint validation |
| CEO/scale reviewer | 8.8 | commodity imaging for most panes; expensive sensing only by exception |

**Mean: 8.38 — PASS. Every critic ≥7.**

Persisted weakest critic: **Glass/fracture scientist = 8.0**, due to unobservable depth/stress from RGB. That limitation is fundamental and should not be “optimized away” with a higher confidence score.

---

# PART V — FINAL WORKING SOLUTION

# 23. Reference architecture: GlassCrack Sentinel (working name)

## 23.1 Baseline hardware

- high-resolution RGB camera with known intrinsics;
- stabilized UAV/robot/handheld platform appropriate to site;
- range/pose estimate when metric output is required;
- no mandatory specialized sensor.

Optional modules:

- circular/rotatable polarizer or polarization camera;
- aligned NIR camera;
- UAV-LDV / specialist NDT for hidden attachment risk;
- controlled-light close-range rig for forensic truthing.

## 23.2 Acquisition

Per pane:

1. identify pane bounds;
2. capture 5–20 frame burst with small lateral motion;
3. preserve enough resolution to meet the requested defect size;
4. run blur/glare/focus/GSD quality checks;
5. auto-reacquire if quality fails.

## 23.3 Vision stack

### Stage A — pane / glass localization

Combine facade geometry with glass-surface segmentation. Return pane polygon and boundary confidence.

### Stage B — registration

Estimate robust planar transform for the pane. Track registration residual and reject frames that violate the plane model.

### Stage C — crack proposal / segmentation

Use a glass-fine-tuned edge-aware segmentation model. Start with a SAM2 adapter or normalization-tuned foundation-model teacher; distill later for deployment.

### Stage D — multi-view surface-consistency fusion

Warp logits/features to pane coordinates. Compute consensus, variance and temporal/geometry residual. Penalize candidates that move like reflection/transmission rather than a surface feature.

### Stage E — morphology and confounder head

Classify crack / scratch / frame-seal / reflection-like / contaminant / unknown. Extract skeleton, branches, endpoints and pane-relative location.

### Stage F — metrology gate

If calibration/GSD and model resolution support it, estimate projected length/width. Otherwise explicitly mark width as **not resolvable**.

### Stage G — risk / action engine

Use visible evidence + pane metadata + history to choose:

- no visible defect;
- monitor;
- reacquire;
- expert review;
- close-range optical verification;
- structural NDT.

It does not output “structurally safe” from RGB alone.

## 23.4 Training strategy

1. generic crack pretraining from OmniCrack30k/DeepCrack/CFD/Crack500/SDNET as appropriate;
2. transparent-surface pretraining/auxiliary learning from Trans10K/video glass datasets;
3. glass-curtain fine tune on proprietary data;
4. hard-negative mining from false positives in field videos;
5. multi-view consistency training;
6. teacher-assisted labeling with expert correction;
7. distillation to production model;
8. continual retraining only after a frozen regression benchmark passes.

## 23.5 Initial deployment target

Do **not** initially promise every glass defect at every distance. A credible v1 target is:

- visible, surface-presenting cracks/scratches on accessible architectural facade panes;
- calibrated minimum resolvable feature size by capture configuration;
- inspection triage and repeat-monitoring;
- building-disjoint validated performance.

Later modules add hidden-sealant/adhesion monitoring and stronger structural risk models.

---

# 24. Final panel scoring

The final panel scores the **Wave-2 architecture**, not an unbuilt claim of production accuracy.

| Panel role | Cost efficiency | Scalability | Technical validity | Operational reliability | Overall |
|---|---:|---:|---:|---:|---:|
| Academic / CV | 8.3 | 8.6 | 8.8 | 8.2 | 8.48 |
| Fracture / glass peer reviewer | 8.0 | 8.2 | 8.5 | 8.2 | 8.23 |
| Facade engineering reviewer | 8.2 | 8.5 | 8.3 | 8.4 | 8.35 |
| Robotics / field operations | 8.4 | 8.7 | 8.2 | 8.3 | 8.40 |
| CEO / shareholder | 8.8 | 8.9 | 8.0 | 8.1 | 8.45 |

**Panel mean: 8.38/10 — PASS.**

### Why it is not a 9 or 10

- glass-specific training/validation data are not yet broad enough in public literature;
- field performance of the proposed multi-view crack-consistency idea must still be experimentally measured;
- sub-mm crack width is not measurable at common stand-off distances with ordinary 20 MP wide-angle UAV imagery;
- RGB cannot recover subsurface crack depth or hidden sealant debonding;
- regulatory/inspection acceptance requires jurisdiction- and application-specific validation beyond ML benchmark scores.

These are real constraints, not reasons to inflate the score.

---

# 25. Top engineering priorities, in order

1. **Build the building-disjoint glass crack/scratch dataset.** This is the critical path and likely the strongest long-term moat.
2. **Implement the capture-quality/GSD gate before training a fancy model.** Bad optics cannot be repaired reliably by architecture alone.
3. **Reproduce GlassCurtainCrackNet-style in-domain baseline** on any accessible glass dataset and report recall/IoU—not just accuracy.
4. **Create a SAM2-adapter / CrackSAM-style strong teacher** and a conventional U-Net/SegFormer baseline.
5. **Implement pane detection and homography registration.**
6. **Run the multi-view reflection-rejection experiment.** This is the highest-value novel hypothesis.
7. **Build hard-negative mining around reflected wires/trees/blinds/frame lines.**
8. **Calibrate uncertainty and automatic reacquisition.**
9. **Establish lab truth with macro/dark-field/polarization and known scratches/cracks.**
10. **Only then optimize deployment/student model and edge inference.**

---

# 26. Minimal MVP acceptance criteria

Before calling the system a working crack-inspection MVP, require all of the following on a sealed, building-disjoint test set:

- pane detection succeeds on ≥99% of evaluable panes or explicitly rejects the rest;
- all capture failures are reported rather than silently scored;
- defective-pane recall target is specified **before** testing;
- false positives are reported per 100 panes;
- crack centerline/topology metric is reported;
- performance is stratified by reflection severity and GSD;
- width is reported only above a validated resolution limit;
- model provides calibrated uncertainty/abstention;
- high-confidence negatives are audited by experts;
- no claim equates “no visible crack” with “structurally safe.”

Numeric recall/false-positive release thresholds should be set after the first representative field dataset because choosing an unsupported number now would be score inflation.

---

# 27. Research gaps that remain worth attacking

1. **Public glass-crack benchmark scarcity.** The current direct-domain evidence is tiny versus opaque-surface crack benchmarks.
2. **Crack depth from remote optics.** Surface imagery alone is fundamentally limited; investigate multi-focus, photometric, interferometric or acoustic fusion only if economics justify it.
3. **Which lite of an IGU is cracked.** Multi-focus/parallax/polarization may help; needs targeted experiments.
4. **Coated/reflective architectural glass.** Low-E and reflective coatings can radically alter cues and NIR/polarization behavior.
5. **Wet/dirty glass.** Water films and streaks can create severe line-like false positives.
6. **Thermal stress + visual evidence.** Couple thermal camera/environmental data with a mechanics model if thermal-break prediction becomes a product goal.
7. **Visible morphology vs forensic cause.** Build a cause-labeled specimen dataset only with verified origins; do not infer cause from “butterfly” or crack pattern alone.
8. **Repeat-inspection registration over months/years.** Facade viewpoint and camera change demand robust pane-coordinate matching.
9. **UAV regulatory/operational constraints.** Site access, wind, obstacle distance and local aviation rules can dominate real costs.
10. **Hidden-frame adhesion.** UAV-LDV is promising enough to track as a second product line or premium inspection tier.

---

# 28. Source ledger

This ledger prioritizes peer-reviewed papers, standards bodies, official manufacturer specifications, and recognized technical organizations. Vendor pages are used only to document existing industrial inspection approaches, not as independent performance validation.

## Glass / facade domain

- Huang, J.; Liu, G. “An iterative up-sampling convolutional neural network for glass curtain crack detection using unmanned aerial vehicles.” *Journal of Building Engineering* (2024). DOI: https://doi.org/10.1016/j.jobe.2024.110814
- Huang, J.; Liu, G. “An Improved Generative Adversarial Network-Based and U-Shaped Transformer Method for Glass Curtain Crack Deblurring Using UAVs.” *Sensors* 24(23), 7713 (2024). https://doi.org/10.3390/s24237713
- “GlassCurtainCrackAdversarialDefense: A convolutional and attention-based adversarial defense network for glass curtain wall crack detection.” 2026. https://www.sciencedirect.com/science/article/pii/S0263224126002538
- Pan et al. “Safety performance assessment of scratched glass panels in a curtain wall system: Development of a vision-based framework.” *Engineering Structures* 348, 121822 (2026). https://doi.org/10.1016/j.engstruct.2025.121822
- Zhou, X.; Tiong, R.L.K. “Defects inspection system for building facades using drones and deep learning method.” *Expert Systems with Applications* 298, 129715 (2026). https://doi.org/10.1016/j.eswa.2025.129715
- NPSA. “Introduction To Glass Curtain Wall Systems.” https://www.npsa.gov.uk/building-protection/windows-glazed-facades/introduction-glass-curtain-wall-systems

## Glass mechanics / standards

- Karlsson et al. “Non-destructive strength testing of microindented float glass by a nonlinear acoustic method.” *Construction and Building Materials* 391, 131748 (2023). https://doi.org/10.1016/j.conbuildmat.2023.131748
- ISO 1288-1:2016. https://www.iso.org/standard/62743.html
- ASTM C1048-25. https://store.astm.org/c1048-25.html
- ASTM C1279-23. https://store.astm.org/c1279-23.html
- ASTM glass standards index. https://store.astm.org/products-services/standards-and-publications/standards/glass-standards-and-ceramic-standards.html
- Kasper. “Spontaneous cracking of thermally toughened safety glass. Part one: Properties of nickel sulphide inclusions.” *Glass Structures & Engineering*. https://doi.org/10.1007/s40940-018-0083-8
- Bonati et al. “A statistical model for the failure of glass plates due to nickel sulfide inclusions.” *Journal of the American Ceramic Society*. https://doi.org/10.1111/jace.16106
- Sakai. “Analysis of beta-phase transformation of nickel sulfide from the origin of spontaneously fractured tempered glass.” *Journal of the Ceramic Society of Japan* 132 (2024). https://doi.org/10.2109/jcersj2.23161
- NIST, *Fractography of Ceramics and Glasses*. https://www.govinfo.gov/content/pkg/GOVPUB-C13-PURL-gpo2659/pdf/GOVPUB-C13-PURL-gpo2659.pdf

## Crack segmentation / datasets

- Benz, C.; Rodehorst, V. “OmniCrack30k: A Benchmark for Crack Segmentation and the Reasonable Effectiveness of Transfer Learning.” CVPR Workshops 2024. https://openaccess.thecvf.com/content/CVPR2024W/VAND/html/Benz_OmniCrack30k_A_Benchmark_for_Crack_Segmentation_and_the_Reasonable_Effectiveness_CVPRW_2024_paper.html
- Liu et al. “DeepCrack: A deep hierarchical feature learning architecture for crack segmentation.” *Neurocomputing*. https://doi.org/10.1016/j.neucom.2019.01.036
- Dorafshan et al. “SDNET2018...” *Data in Brief* 21 (2018). https://doi.org/10.1016/j.dib.2018.11.015
- CrackSAM. *Construction and Building Materials* 431 (2024), 136573. https://doi.org/10.1016/j.conbuildmat.2024.136573
- Chen et al. “SECrackSeg...” *Sensors* 25(9), 2642 (2025). https://doi.org/10.3390/s25092642
- Rostami et al. “Segment Any Crack...” *Journal of Computing in Civil Engineering* 40(3) (2026). https://doi.org/10.1061/JCCEE5.CPENG-7090
- Kuttuva & Prawin. “Structural defect segmentation using a semi-supervised algorithm integrating YOLO and the segment anything model.” *Automation in Construction* (2025). https://doi.org/10.1016/j.autcon.2025.106709

## Transparent/glass perception

- Xie et al. “Segmenting Transparent Objects in the Wild” / Trans10K. https://arxiv.org/abs/2003.13948
- Yan et al. “NRGlassNet: Glass surface detection from visible and near-infrared image pairs.” *Knowledge-Based Systems* 294 (2024), 111722. https://doi.org/10.1016/j.knosys.2024.111722
- “Transparent objects segmentation based on polarization imaging and deep learning.” *Optics Communications* 555 (2024), 130246. https://doi.org/10.1016/j.optcom.2023.130246
- Lu, Huang & Yan. “MVGD-Net: A Novel Motion-aware Video Glass Surface Detection Method.” AAAI 2026. https://doi.org/10.1609/aaai.v40i9.37700
- Cheng, Guan & Lau. “Multi-Semantic Modeling for Glass Surface Detection in the Wild.” AAAI 2026. https://doi.org/10.1609/aaai.v40i5.37322

## NDT / specialist sensing

- Hong et al. “Debonding Detection in Hidden Frame Supported Glass Curtain Walls Using the Nonlinear Ultrasonic Modulation Method...” *Sensors* 18(7), 2094 (2018). https://doi.org/10.3390/s18072094
- Hong et al. “Interfacial Adhesion–Strength Detection... Based on Nonlinear Ultrasonic Lamb Wave.” *Journal of Aerospace Engineering* 31(5) (2018). https://doi.org/10.1061/(ASCE)AS.1943-5525.0000870
- Guo et al. “Remote vibration-based debonding identification in hidden-frame glass curtain walls using a UAV-mounted LDV system.” *Mechanical Systems and Signal Processing* 257 (2026), 114638. https://doi.org/10.1016/j.ymssp.2026.114638
- MVTec HALCON deflectometry. https://www.mvtec.com/knowledge-base/developers-corner/article/inspection-of-specular-surfaces-with-deflectometry-in-halcon
- Dark Field Technologies glass inspection. https://darkfield.com/industries/glass/

## Camera reference

- DJI Mavic 2 official specifications. https://www.dji.com/mavic-2/info

---

# 29. Founding-engineer conclusions

1. **This is an optics + geometry + data problem before it is a neural-network problem.**
2. **Reflections are the signature failure mode.** Use motion/viewpoint physics rather than hoping a classifier memorizes every reflected line.
3. **Dataset independence matters more than image count.** Two buildings and hundreds of images cannot establish generalization.
4. **Recall must be optimized and reported at pane/instance level.** Pixel accuracy is inadequate for thin sparse defects.
5. **Never report a width the camera cannot resolve.** Carry GSD/resolvability through the product.
6. **Visible defect ≠ residual strength.** Fracture mechanics requires depth/stress/material state that RGB often does not provide.
7. **No visible crack ≠ facade safe.** Hidden structural-silicone / attachment defects need separate sensing.
8. **Use expensive sensors selectively.** RGB multi-view is the scalable baseline; NIR, polarization, LDV, DIC, ultrasound and controlled optics are escalation/validation tools.
9. **Abstention and reacquisition are core features.** A safety-related system must know when the evidence is poor.
10. **The highest-value novel experiment is pane-plane multi-view consistency.** It is physics-motivated, cheap in hardware, and directly targets the glass-specific false-positive mechanism.


---

# 30. RGB-first product directive — cost and scalability constrained

**Decision date:** 2026-09-16  
**System-level owner:** Founding-engineering / architecture panel  
**Constraint:** Prefer a commodity RGB camera as the default field sensor. Additional modalities are allowed only as targeted escalation paths when their incremental reliability justifies added hardware, calibration, payload, operator training, and maintenance.

## 30.1 Decision

The production baseline is now explicitly **RGB-only at capture time**:

1. one calibrated RGB camera;
2. short burst/video capture while the camera translates laterally or obliquely by a small, controlled baseline;
3. pane detection / glass-region localization;
4. geometric registration to the pane plane;
5. thin-crack segmentation;
6. multi-view persistence / residual-motion filtering;
7. uncertainty estimation;
8. automatic RGB reacquisition when evidence is inadequate;
9. optional specialist escalation only after RGB fails a defined confidence or risk gate.

This is the preferred architecture because it extracts additional physical information from **camera motion rather than an additional sensor**. Video glass-detection research supports this premise: VGSD-Net (AAAI 2024) models dynamic reflection cues from moving RGB video, and MVGD-Net (AAAI 2026) exploits motion inconsistency between glass-plane content and reflected/transmitted objects. Neither paper solves glass-crack detection directly, but both independently support the core optical mechanism that a moving RGB camera contains useful information about reflective transparent surfaces.

**Sources:**
- Liu et al., “Multi-View Dynamic Reflection Prior for Video Glass Surface Detection,” AAAI 2024. https://doi.org/10.1609/aaai.v38i4.28148
- Lu, Huang & Yan, “MVGD-Net: A Novel Motion-aware Video Glass Surface Detection Method,” AAAI 2026. https://doi.org/10.1609/aaai.v40i9.37700

## 30.2 Why multi-view RGB is physically useful

Assume the glass pane is locally planar. Let the glass plane be at effective depth \(Z_g\) and let a reflected or transmitted scene feature correspond to an apparent virtual/object depth \(Z_v\). For a small camera translation with baseline \(B\) and focal length \(f\), the first-order horizontal image displacement for a point at depth \(Z\) is approximately

\[
 d(Z) \approx \frac{fB}{Z}.
\]

If frames are rectified with the homography induced by the glass plane, a true feature physically attached to the glass surface should have near-zero residual displacement apart from calibration error, rolling shutter, vibration, lens distortion, and segmentation noise:

\[
 r_{\text{surface}} \approx 0.
\]

A reflected or transmitted feature associated with a different effective depth retains residual motion approximately

\[
 r_{\text{nonplane}} \approx fB\left(\frac{1}{Z_v}-\frac{1}{Z_g}\right).
\]

Therefore, increasing the useful baseline \(B\), while keeping blur and perspective distortion manageable, increases the separability of glass-plane defects from non-plane visual structure. This is not a proof that every reflection can be rejected—specular geometry can be more complicated than a single virtual depth—but it gives a falsifiable physical reason that **RGB burst capture should outperform single-frame RGB on the dominant false-positive class**.

### Falsifiable test

For every crack candidate, compute its post-homography residual displacement / mask consistency across \(N\) frames. Compare distributions for:

- verified surface cracks;
- scratches on the glass surface;
- reflected wires / tree branches / railings;
- transmitted blinds / mullions / interior linear features;
- sealant/frame edges;
- water streaks / dirt.

The hypothesis passes only if a threshold or learned consistency model materially improves crack precision at matched recall on **building-disjoint glass-curtain-wall data**.

## 30.3 RGB-only acquisition protocol

The camera should do more work before the neural network is asked to compensate for poor evidence.

### Default burst

- Capture approximately 5–15 high-resolution frames per pane/region rather than one still image.
- Keep focus, focal length, white balance, and preferably exposure fixed during a burst so appearance changes do not masquerade as temporal cues.
- Move the camera enough to create measurable parallax/reflection change, but not so much that the pane becomes heavily foreshortened or the GSD becomes unacceptable.
- Record camera pose/IMU metadata when available, but do not make accurate external pose a requirement; image registration must still work from visual geometry.
- Reject frames with severe motion blur instead of forcing a prediction.

The published GlassCurtainCrack deblurring work created 4,141 clear/blurred RGB image pairs from high-frame-rate DJI Mavic 2 Pro videos, reinforcing that motion blur is a real acquisition failure mode worth handling explicitly rather than treating as ordinary augmentation.

**Source:** Huang & Liu, “An Improved Generative Adversarial Network-Based and U-Shaped Transformer Method for Glass Curtain Crack Deblurring Using UAVs,” *Sensors* 2024. https://doi.org/10.3390/s24237713

### Exposure strategy

For ordinary panes, use one fixed exposure per burst. For severe dynamic range, allow a second RGB burst at a different exposure only if the first burst saturates large glass regions. Exposure bracketing costs capture time but does not add a sensor modality.

### Reacquisition is cheaper than multimodal hardware

If uncertainty is high, the system should first request one of the following before escalating hardware:

1. a closer RGB pass;
2. a slightly different view angle;
3. a slower pass to reduce blur;
4. a second fixed exposure;
5. a larger lateral baseline;
6. a manual close-up RGB capture for a small high-risk region.

This preserves a homogeneous RGB fleet and pushes complexity into software and flight/capture policy.

## 30.4 Software stack optimized for RGB

### Stage A — pane / glass-region localization

Use a lightweight glass/pane localization model plus curtain-wall geometry. RGB-only glass detection is imperfect, so exploit facade regularity: mullion grids, repeated rectangles, vanishing lines, and temporal glass cues.

Video glass-detection datasets are much larger than currently available direct crack-on-glass datasets. VGSD-D contains 19,166 annotated frames from 297 RGB videos; MVGD-Net reports a separate dataset of 19,268 frames from 312 diverse glass scenarios. These should be considered for **pretraining glass-awareness**, not as crack labels.

### Stage B — plane rectification

Estimate pane corners / planar homography and map each observation to a canonical pane coordinate system. This makes temporal consistency measurable and gives a stable coordinate frame for repeat inspections.

### Stage C — crack candidate model

Prefer a thin-structure segmentation model with:

- high-resolution tile inference;
- edge / centerline-aware loss;
- multi-scale decoder;
- hard-negative training;
- parameter-efficient fine-tuning from a strong vision backbone rather than full training from scratch.

The directly relevant 2024 glass-curtain-wall work already demonstrates that RGB UAV imagery can be used for crack segmentation, but its dataset is too small to establish broad generalization. The system should treat that work as a domain anchor rather than a deployment benchmark.

**Source:** Huang & Liu, “An iterative up-sampling convolutional neural network for glass curtain crack detection using unmanned aerial vehicles,” *Journal of Building Engineering* 2024. https://doi.org/10.1016/j.jobe.2024.110814

### Stage D — multi-view consistency head

For each crack candidate, estimate:

\[
C = w_1 C_{\text{mask}} + w_2 C_{\text{centerline}} + w_3 C_{\text{appearance}} + w_4 C_{\text{motion}},
\]

where all terms are computed in pane-rectified coordinates. The exact weights are learned/calibrated on validation data rather than hand-tuned for production.

A genuine surface crack should generally show:

- stable position after pane registration;
- stable topology;
- stable local edge orientation;
- limited residual optical flow relative to the pane.

A reflected/transmitted line is expected to violate one or more of these more often.

### Stage E — uncertainty and abstention

The model must output at least three states:

- **defect evidence sufficient**;
- **no defect evidence at current resolution/conditions**;
- **insufficient visual evidence — reacquire/escalate**.

The third state is essential to keep RGB viable without pretending it is omniscient.

## 30.5 Cost / scalability comparison

The following scores are **architecture-fit scores**, not experimental accuracy claims. Higher cost score means cheaper/easier to own and operate.

| Architecture | Cost | Scalability | Reliability potential | Accuracy potential for visible cracks | Mean | Decision |
|---|---:|---:|---:|---:|---:|---|
| Single-frame RGB | 10.0 | 10.0 | 5.5 | 6.0 | 7.88 | Too reflection-sensitive for primary design |
| **Multi-view RGB, one camera** | **9.5** | **9.5** | **7.7** | **7.8** | **8.63** | **Primary baseline** |
| Multi-view RGB + passive fixed polarizer | 8.8 | 8.8 | 7.8 | 8.0 | 8.35 | Optional low-cost experiment |
| RGB + mechanically rotated polarization | 7.2 | 6.8 | 8.2 | 8.3 | 7.63 | Specialist / controlled capture |
| RGB + NIR pair | 6.2 | 6.5 | 8.2 | 8.2 | 7.28 | Escalation only |
| RGB + thermal/IR | 5.5 | 5.8 | 8.0 for thermal/subsurface targets | 6.5 for visible surface cracks | 6.45 | Different problem class |
| Laser vibrometry / ultrasonic NDT | 2.5 | 2.5 | 9.0 for supported hidden-defect tasks | N/A for ordinary visual crack screening | — | Premium forensic / hidden-defect tier |

Why multi-view RGB scores highest: it adds **capture time and software**, but no additional field sensor, spectral calibration, cross-modal registration, specialized payload, or heterogeneous maintenance inventory.

## 30.6 Passive polarizer: the only near-baseline hardware option worth early testing

A simple circular/linear polarizing filter can suppress some specular reflections from dielectric surfaces such as glass, especially near favorable incidence angles. However, suppression is strongly dependent on viewing geometry and filter orientation; one fixed orientation will not remove all facade reflections. A polarizer also reduces incoming light, which can force longer exposure / higher ISO and worsen motion blur.

Therefore:

- do **not** require a polarizer for the baseline;
- run it as a cheap A/B experiment using the same RGB camera;
- compare no-filter, fixed-polarizer, and manually/actuated-rotation captures;
- retain it only if matched-recall false positives materially improve enough to offset blur/exposure and operator complexity.

Relevant polarization/reflection literature confirms the physical basis but does not yet establish a deployable advantage for UAV glass-crack screening:
- Wang et al., “Polarized reflection removal with dual-stream attention guidance,” *Pattern Recognition* 2025. https://doi.org/10.1016/j.patcog.2024.110945
- Wolff & Boult-style multi-polarization reflection separation lineage; modern discussion and datasets continue to exploit angle-dependent polarized reflection cues.

## 30.7 What remains outside the RGB promise

RGB-first is appropriate only if the product claim stays narrow and honest.

RGB can target:

- visible surface crack presence;
- topology / approximate projected length;
- pane-relative location;
- visible scratch/chip morphology;
- change over repeat inspections;
- approximate width when GSD and calibration validate it.

RGB should **not** independently certify:

- crack depth through the pane;
- residual glass strength;
- hidden structural-silicone adhesion;
- subsurface delamination;
- which lite of a multi-lite IGU is definitively cracked in every case;
- absence of all safety-critical defects.

Keeping those claims out of the baseline is what allows the RGB architecture to remain cheap and scalable without inflating reliability claims.

## 30.8 RGB-first critic loop

### Critic A — “Why not just single-frame RGB?”

**Failure:** reflection/transmission line clutter is too similar to cracks on reflective curtain-wall glass.  
**Resolution:** retain RGB hardware, add camera-motion evidence.  
**Score:** 8.7/10 — PASS.

### Critic B — “Why not require NIR because it can suppress reflection?”

**Finding:** NIR/RGB pairing is technically promising, but it adds another sensor, cross-modal alignment/calibration, payload and fleet heterogeneity. NRGlassNet specifically reports that NIR reflections are often suppressed relative to RGB for glass-surface detection, so it remains a strong escalation candidate—not the default.

**Source:** Yan et al., “NRGlassNet: Glass surface detection from visible and near-infrared image pairs,” *Knowledge-Based Systems* 294 (2024), 111722. https://doi.org/10.1016/j.knosys.2024.111722

**Resolution:** keep NIR behind the RGB uncertainty gate.  
**Score:** 8.3/10 — PASS.

### Critic C — “Can one-camera multi-view scale operationally?”

**Finding:** it increases images and compute per pane, but image capture/storage/inference scale more favorably than adding a second calibrated sensor to every device. Existing facade research already uses high-resolution UAV RGB at large image volumes, and video glass-detection work operates directly on temporal RGB sequences.

**Resolution:** use short bursts, tile only candidate panes, perform inexpensive registration before expensive segmentation, and discard redundant frames.  
**Score:** 8.8/10 — PASS.

### Critic D — “Are we overclaiming the reflection physics?”

**Finding:** yes, if we claim every reflection will move differently. Some reflected structures can be locally stable, small-baseline motion may be below pixel resolution, and glass may be curved/non-planar.  
**Resolution:** multi-view consistency becomes one feature in a probabilistic decision system, not a hard law. Hard negatives must explicitly include stable reflections.  
**Score:** 7.6/10 — PASS.

### Critic E — “Does RGB remain viable on low-contrast or highly reflective coated glass?”

**Finding:** unresolved for broad deployment.  
**Resolution:** capture-condition detection + abstention + targeted escalation. Build the first dedicated validation strata around reflective coatings, low-E glass, tinted glass, frit, wet glass, night/dusk, backlit scenes, and deep interior clutter.  
**Score:** 7.1/10 — PASS, but this is now the weakest RGB-first risk.

## 30.9 New priority ordering

The RGB constraint changes the research order to:

1. **Build a glass-specific RGB field dataset with building-disjoint splits.**
2. **Prove multi-view crack/reflection separability experimentally.**
3. **Optimize capture baseline, stand-off distance, shutter speed, and GSD.**
4. **Build high-recall thin-crack segmentation with hard-negative mining.**
5. **Calibrate uncertainty + RGB reacquisition policy.**
6. **Test passive polarization as a low-cost optional accessory.**
7. **Only then test NIR for the failure strata where RGB remains below acceptance.**
8. Keep thermal, LDV, ultrasound and other NDT outside the baseline product unless the product scope expands to hidden/subsurface structural defects.

## 30.10 Updated founding-engineering recommendation

The preferred product is **not “RGB because it is cheapest.”** It is:

> **A motion-aware, uncertainty-aware RGB inspection system whose capture policy is designed around the physics of reflective glass.**

That framing matters. A cheap single-frame camera plus a generic crack model is unlikely to be robust enough. A single commodity RGB camera plus deliberate view motion, pane-plane geometry, temporal consistency, hard-negative training, and automatic reacquisition has a credible path to both low field cost and fleet-scale deployment.

**Current RGB-first architecture score: 8.6/10 — PASS.**  
The score is not raised above 9 until building-disjoint field tests demonstrate that the multi-view consistency mechanism materially improves matched-recall precision on coated/reflective curtain-wall glass without creating unacceptable flight/capture overhead.

# 31. Executive sensing decision — RGB must win by default

**Product directive (2026-09-16):** The field product should use commodity RGB as the default sensing modality. Every additional sensor must demonstrate a statistically meaningful improvement on the hard-case benchmark that cannot be recovered through better RGB acquisition, multi-view geometry, training data, or uncertainty-aware reacquisition.

## 31.1 Baseline to build first

The first production candidate is:

**one calibrated RGB camera -> short burst/video -> pane localization -> frame quality filtering -> facade/pane registration -> edge-aware crack segmentation -> multi-view consistency -> calibrated uncertainty -> RGB reacquisition -> human/NDT escalation when unresolved.**

This preserves the most scalable fleet architecture: one common camera payload, one calibration workflow, no cross-spectral registration, no synchronization between sensors, and no additional spectral-maintenance inventory.

The decision is consistent with published evidence but is not claimed to be proven for crack detection yet. Huang & Liu (2024) directly demonstrate UAV RGB crack segmentation on glass curtain walls, while Liu et al. (AAAI 2024) show that moving RGB video supplies dynamic reflection cues for glass-surface detection. The latter is adjacent evidence, not a crack-detection result; our core experiment must test whether those motion cues suppress crack false positives caused by reflected scene structure.

## 31.2 Modality hierarchy

| Tier | Modality | Role | Why it is / is not scalable |
|---|---|---|---|
| 0 | **Multi-view RGB** | Default inspection | Commodity hardware; motion gives additional optical evidence without another sensor |
| 1 | RGB exposure/viewpoint reacquisition | First recovery action | Near-zero BOM increase; costs only acquisition time |
| 1 experimental | Fixed passive polarizer on RGB | Cheap A/B challenger | Very low hardware cost, but angle-dependent and loses light |
| 2 | Rotating/multi-angle polarization | Reflection-heavy specialist cases | Strong physics, but actuator/capture/calibration complexity grows |
| 2 | Paired RGB-NIR | Glass-localization / ambiguity escalation | NIR reflections can be suppressed relative to RGB, but requires another imager and registration |
| 3 | UAV-LDV / ultrasound / specialist NDT | Hidden debonding or structural escalation | Solves an observability problem that RGB cannot; too specialized for every scan |

A May 2026 polarization-reflection paper uses a physics-based polarization rendering pipeline and a real polarization-camera test set; CVPR 2026 GlassPol likewise relies on polarization-state measurements. These are valuable evidence that polarization can resolve reflection phenomena, but they also reinforce that the measurement is no longer a commodity single-RGB-camera workflow. Accordingly, polarization remains a benchmark challenger rather than baseline hardware.

## 31.3 The experiment that decides whether another modality earns deployment

Build a matched-pane benchmark rather than comparing papers with incompatible datasets.

For the same panes and defect locations, collect:

1. single RGB;
2. 3-10 frame RGB burst from controlled viewpoint displacement;
3. RGB burst plus a fixed passive polarizer;
4. multi-angle polarization on a subset;
5. aligned RGB-NIR on a subset;
6. macro/controlled-light ground truth and expert annotation.

Stratify samples by reflection severity, viewing angle, sun position, tint/coating, interior clutter, crack width, crack orientation, pane type, and motion blur. Keep entire buildings out of training when evaluating generalization.

Report, at minimum:

- crack recall at fixed false positives per pane;
- precision and boundary F-score;
- failure-to-decide / reacquisition rate;
- building-disjoint performance;
- reflection-heavy subset performance;
- inference cost per pane;
- added capture seconds per pane;
- sensor BOM and payload mass;
- calibration/maintenance labor;
- incremental recall per added dollar and per added capture second.

**Promotion gate for any non-RGB modality:** it should not enter the standard payload merely because its raw accuracy is higher. It must deliver a material reduction in unresolved high-risk cases or false positives after the optimized multi-view RGB pipeline, with enough operational benefit to offset additional hardware, calibration, payload, acquisition time, and fleet maintenance.

## 31.4 Current architecture-fit scoring

These are conservative engineering-fit scores, not measured crack-detection accuracies.

| Candidate | Cost | Scalability | Reliability potential | Crack-screening potential | Mean | Decision |
|---|---:|---:|---:|---:|---:|---|
| Single-frame RGB | 10.0 | 10.0 | 5.5 | 6.0 | 7.88 | Baseline control only |
| **Multi-view RGB** | **9.5** | **9.5** | **7.7** | **7.8** | **8.63** | **Build first** |
| RGB + fixed passive polarizer | 8.8 | 8.8 | 7.8 | 8.0 | 8.35 | Cheap challenger experiment |
| RGB + rotating polarization | 7.2 | 6.8 | 8.2 | 8.3 | 7.63 | Escalation / controlled use |
| RGB + NIR | 6.2 | 6.5 | 8.2 | 8.2 | 7.28 | Escalation only |
| UAV-LDV / specialist NDT | 3.5 | 3.5 | 9.0 for its target defect class | N/A for ordinary visible-crack screening | N/A | Separate structural workflow |

The multi-view RGB score remains the strongest product fit because its incremental information comes from acquisition geometry rather than added sensor hardware. The score is intentionally below 9: glass-specific field data are still limited, crack depth/stress remain unobservable, and the reflection-rejection hypothesis must be validated directly on representative curtain walls.

## 31.5 Immediate R&D order

1. **Do not buy a complex sensor stack first.** Build the RGB acquisition and evaluation harness.
2. Reproduce the direct glass-curtain crack baseline on available GCC-style data.
3. Add pane registration and multi-view persistence tests.
4. Build the reflection-heavy hard-negative set before architecture tuning.
5. Test RGB reacquisition policies: closer view, lateral displacement, oblique view, exposure adjustment.
6. Run a low-cost fixed-polarizer A/B experiment using the same RGB camera.
7. Only then benchmark rotating polarization and paired NIR on the cases RGB still cannot resolve.
8. Keep hidden debonding / adhesion health as a separate NDT escalation problem rather than expanding the RGB claim.

**Current decision:** Proceed with **multi-view RGB as the founding architecture**. Treat a passive polarizer as the first inexpensive hardware experiment. Require experimental evidence before promoting NIR or active polarization into the production fleet.

### New supporting literature checked in this iteration

- Liu et al., “Multi-View Dynamic Reflection Prior for Video Glass Surface Detection,” AAAI 2024. RGB video dataset: 19,166 frames from 297 videos; exploits temporally changing reflections. https://ojs.aaai.org/index.php/AAAI/article/view/28148
- Huang & Liu, “An iterative up-sampling convolutional neural network for glass curtain crack detection using unmanned aerial vehicles,” Journal of Building Engineering, 2024. Direct UAV RGB glass-curtain crack work. https://doi.org/10.1016/j.jobe.2024.110814
- Yan et al., “NRGlassNet: Glass surface detection from visible and near-infrared image pairs,” Knowledge-Based Systems, 2024. Reports suppression of glass reflections in NIR and uses 1,378 RGB-NIR pairs. https://doi.org/10.1016/j.knosys.2024.111722
- Bian, Monno & Okutomi, “Reflection removal using recurrent polarization-to-polarization network,” Machine Vision and Applications, 2026. Uses physics-based polarization synthesis and a real polarization-camera test dataset. https://doi.org/10.1007/s00138-026-01837-6
- Wang, Lu & Tian, “Polarization State Tracing for Reflection Removal and Color-Consistent Reconstruction,” CVPR 2026. Introduces GlassPol and multi-angle polarization reconstruction. https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Polarization_State_Tracing_for_Reflection_Removal_and_Color-Consistent_Reconstruction_CVPR_2026_paper.html
- Guo et al., “Remote vibration-based debonding identification in hidden-frame glass curtain walls using a UAV-mounted LDV system,” Mechanical Systems and Signal Processing, 2026. Demonstrates a specialist remote method for concealed adhesive debonding—a target outside ordinary RGB observability. https://doi.org/10.1016/j.ymssp.2026.114638
