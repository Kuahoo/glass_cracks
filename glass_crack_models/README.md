# glass_crack_models

Training, evaluation and inference code for `model_implementation_plan.md`. Run everything from the repo root as modules. Paths come from `DATA_ROOT` (default `temp/glass_crack_dataset_bundle`) and `OUT_ROOT` (default `outputs/`). On the laptop, use `.venv/bin/python`.

```sh
# once per machine/session: linear_defect set + source-weighting lists
python -m glass_crack_models.data.relabel_classes

# checks
python -m unittest discover -s glass_crack_models/tests -t .
python -m glass_crack_models.viz.overlay gt --dataset linear --n 20
python -m glass_crack_models.train.train_yolo   --config e2_yolo_linear.yaml --smoke
python -m glass_crack_models.train.train_semseg --config e3_unet_r34.yaml   --smoke

# training (resumes automatically; evaluates best checkpoint and appends to OUT_ROOT/results/results.csv)
python -m glass_crack_models.train.train_yolo   --config e2_yolo_linear.yaml --seed 0
python -m glass_crack_models.train.train_yolo   --config e1_yolo_4class.yaml --seed 0
python -m glass_crack_models.train.train_semseg --config e3_unet_r34.yaml   --seed 0

# re-evaluate a checkpoint (add --reuse-cache to re-score saved predictions without a GPU)
python -m glass_crack_models.eval.run_eval --checkpoint OUT_ROOT/runs/E2_s0/weights/best.pt --exp-id E2 --seed 0

# predict on photos (threshold from operating_point.json next to the checkpoint)
python -m glass_crack_models.infer.predict --checkpoint .../best.pt --input photos/ --out preds/

# CUBIT tiles for pretraining (Phase 4)
python -m glass_crack_models.data.tile_cubit --pilot 500 --out outputs/cubit_pilot
python -m glass_crack_models.data.tile_cubit --windows 1024 --out outputs/cubit_tiles_1024
```

Per evaluation, `OUT_ROOT/evals/<exp>_s<seed>/` holds `summary.json` (all metrics with 95% cluster-bootstrap CIs), `operating_point.json`, `pr_curves.png`, `galleries/valid_false_{negatives,positives}.jpg`, and `cache/` (uint8 score maps).
