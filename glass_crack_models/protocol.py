"""Evaluation protocol constants (model_implementation_plan.md, Sections 4 and 4a).

These are fixed in advance and are not tuned per experiment.
"""

MERGED_NAMES = ["bubble", "crack", "inclusion", "scratch"]
LINEAR_NAMES = ["linear_defect"]
LINEAR_SOURCE_IDS = {1, 3}  # crack, scratch in glass_merged_seg
NEGATIVE_DEFECT_IDS = {0, 2}  # bubble, inclusion: removed under linear_defect, images kept

MIN_REGION_PX = 24  # at model-input resolution (a 640 px image or tile)
MATCH_IOU = 0.3
STRICT_IOU = 0.5
MATCH_DILATE_PX = 3

THRESHOLD_GRID = [round(0.05 * i, 2) for i in range(1, 20)]  # 0.05 .. 0.95
OPERATING_POINT_RULE = "max_f1"  # on merged valid, lenient matching; ties go to higher recall

BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_CI = 0.95

TILE = 640
TILE_OVERLAP = 0.25

AGDD_NAMES = ["contusion", "scratches", "crack", "spot"]
AGDD_TO_EVAL = {
    "linear_defect": {1, 2},  # scratches, crack
    "crack": {2},
}
AGDD_NONLINEAR_IDS = {0, 3}  # contusion, spot: not ground truth, hits counted as confusions
AGDD_BOX_IOU = 0.3
AGDD_STRICT_BOX_IOU = 0.5
AGDD_HIT_FRACTION = 0.5  # share of predicted pixels inside the GT oriented box
AGDD_VIEWS = ["images", "image"]
