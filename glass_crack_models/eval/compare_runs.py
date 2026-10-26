"""Threshold-free comparison of finished runs (Phase 1), from the sweeps already stored in summary.json.

    python -m glass_crack_models.eval.compare_runs                     # every run under OUT_ROOT/evals
    python -m glass_crack_models.eval.compare_runs --runs E1_s0 E1_s1 E1_s2 E2_s0 E2_s1 E2_s2 E3_s0

No GPU and no data needed. Per run, from the threshold sweeps (0.05 .. 0.95):
- test recall at a fixed precision (max recall over thresholds whose precision reaches the target),
  the FP/100 at that threshold, area under the precision/recall curve (AP, step-interpolated over the
  grid, so it covers recalls the grid reaches and is comparable across runs only), and the best test F1
  (an oracle number: never used to pick anything);
- AGDD per view at the frozen threshold (recall, crack recall, precision) and AP per view; the pair
  recall is left out on purpose (report the views separately);
- NCUGD recall at the frozen threshold and the best recall at any threshold.
Then mean +/- SD over seeds per experiment. Writes <out>/compare_runs.csv (per run) and
<out>/compare_runs.md (tables), default out = OUT_ROOT/results.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from glass_crack_models import paths
from glass_crack_models.protocol import AGDD_VIEWS

SPLIT_SWEEPS = {"valid": "merged valid", "test": "merged test"}


def _ok(x) -> bool:
    return isinstance(x, (int, float)) and not math.isnan(x)


def recall_at_precision(rows: list[dict], target: float, rkey: str = "recall", pkey: str = "precision") -> dict:
    """Highest recall among sweep rows with precision >= target; recall 0 if no threshold gets there."""
    ok = [r for r in rows if _ok(r.get(pkey)) and _ok(r.get(rkey)) and r[pkey] >= target]
    if not ok:
        return {"recall": 0.0, "thr": None, "fp100": float("nan")}
    best = max(ok, key=lambda r: (r[rkey], r["thr"]))
    return {"recall": best[rkey], "thr": best["thr"], "fp100": best.get("fp100", float("nan"))}


def pr_auc(rows: list[dict], rkey: str = "recall", pkey: str = "precision") -> float:
    """Step-interpolated area under the PR points (precision at r = best precision at recall >= r)."""
    pts = sorted((r[rkey], r[pkey]) for r in rows if _ok(r.get(rkey)) and _ok(r.get(pkey)))
    if not pts:
        return 0.0
    area, prev_r = 0.0, 0.0
    for i, (rec, _) in enumerate(pts):
        p_env = max(p for _, p in pts[i:])
        area += (rec - prev_r) * p_env
        prev_r = rec
    return area


def best_f1(rows: list[dict]) -> float:
    vals = [r["f1"] for r in rows if _ok(r.get("f1"))]
    return max(vals) if vals else 0.0


def max_recall(rows: list[dict], rkey: str = "recall") -> float:
    vals = [r[rkey] for r in rows if _ok(r.get(rkey))]
    return max(vals) if vals else 0.0


def run_metrics(summary: dict, precisions: list[float]) -> dict:
    meta, sweeps = summary["meta"], summary.get("sweeps", {})
    out = {"run": f"{meta['exp_id']}_s{meta['seed']}", "exp_id": meta["exp_id"], "seed": meta["seed"],
           "threshold": meta.get("threshold")}
    test = summary.get("test", {})
    for k in ("recall", "precision", "fp100", "mask_iou"):
        out[f"test_{k}@op"] = test.get(k, float("nan"))
    for split, key in SPLIT_SWEEPS.items():
        rows = sweeps.get(key)
        if not rows:
            continue
        out[f"{split}_ap"] = pr_auc(rows)
        if split == "test":
            out["test_best_f1"] = best_f1(rows)
            for p in precisions:
                r = recall_at_precision(rows, p)
                out[f"test_recall@p{p:.2f}"] = r["recall"]
                out[f"test_fp100@p{p:.2f}"] = r["fp100"]
                out[f"test_thr@p{p:.2f}"] = r["thr"] if r["thr"] is not None else float("nan")
    if "ncugd" in summary:
        out["ncugd_recall@op"] = summary["ncugd"].get("recall", float("nan"))
        out["ncugd_max_recall"] = max_recall(sweeps.get("NCUGD", []))
    agdd = summary.get("agdd", {})
    for v in AGDD_VIEWS:
        s = agdd.get(v)
        if s:
            out[f"agdd_{v}_recall@op"] = s.get("recall", float("nan"))
            out[f"agdd_{v}_crack_recall@op"] = s.get("crack_recall", float("nan"))
            out[f"agdd_{v}_precision@op"] = s.get("precision", float("nan"))
        rows = sweeps.get(f"AGDD {v}/ (box)")
        if rows:
            out[f"agdd_{v}_ap"] = pr_auc(rows, f"{v}_recall", f"{v}_precision")
    return out


def load_runs(eval_root: Path, names: list[str] | None) -> list[dict]:
    dirs = [eval_root / n for n in names] if names else sorted(p for p in eval_root.iterdir() if p.is_dir())
    out = []
    for d in dirs:
        f = d / "summary.json"
        if not f.exists():
            if names:
                print(f"skip {d.name}: no summary.json")
            continue
        s = json.loads(f.read_text())
        if s.get("meta", {}).get("limit") or d.name.endswith("_smoke"):
            continue
        out.append(s)
    return out


def mean_sd(vals: list[float]) -> tuple[float, float, int]:
    v = [x for x in vals if _ok(x)]
    if not v:
        return float("nan"), float("nan"), 0
    m = sum(v) / len(v)
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else float("nan")
    return m, sd, len(v)


def aggregate(per_run: list[dict]) -> dict[str, dict[str, tuple[float, float, int]]]:
    by_exp: dict[str, list[dict]] = {}
    for r in per_run:
        by_exp.setdefault(r["exp_id"], []).append(r)
    cols = [k for k in per_run[0] if k not in ("run", "exp_id", "seed")] if per_run else []
    return {e: {c: mean_sd([r.get(c, float("nan")) for r in rs]) for c in cols} for e, rs in sorted(by_exp.items())}


def _cell(stat: tuple[float, float, int], digits: int) -> str:
    m, sd, n = stat
    if n == 0:
        return "-"
    return f"{m:.{digits}f}" + (f" ± {sd:.{digits}f}" if n > 1 else "")


def markdown(agg: dict, precisions: list[float]) -> str:
    p0 = precisions[len(precisions) // 2] if precisions else 0.85
    test_cols = [(f"test_recall@p{p:.2f}", f"Recall @ P≥{p:.2f}", 3) for p in precisions]
    test_cols += [(f"test_fp100@p{p0:.2f}", f"FP/100 @ P≥{p0:.2f}", 1), ("test_ap", "PR area", 3),
                  ("test_best_f1", "Best F1 (oracle)", 3), ("valid_ap", "Valid PR area", 3),
                  ("test_recall@op", "Recall @ frozen thr", 3), ("threshold", "Frozen thr", 2)]
    agdd_cols = []
    for v in AGDD_VIEWS:
        agdd_cols += [(f"agdd_{v}_recall@op", f"`{v}/` recall", 3), (f"agdd_{v}_crack_recall@op", f"`{v}/` crack recall", 3),
                      (f"agdd_{v}_ap", f"`{v}/` PR area", 3)]
    agdd_cols += [("ncugd_recall@op", "NCUGD recall", 3), ("ncugd_max_recall", "NCUGD best recall, any thr", 3)]

    missing = (float("nan"), float("nan"), 0)

    def table(cols):
        head = "| Experiment | Seeds | " + " | ".join(c[1] for c in cols) + " |"
        sep = "|---|---|" + "---|" * len(cols)
        body = [f"| {e} | {max(st[2] for st in a.values())} | " + " | ".join(_cell(a.get(k, missing), d) for k, _, d in cols) + " |"
                for e, a in agg.items()]
        return "\n".join([head, sep, *body])

    return ("## Merged test, threshold-free (mean ± SD over seeds)\n\n" + table(test_cols) +
            "\n\n## External sets at the frozen threshold (AGDD per view; pair recall not reported)\n\n" + table(agdd_cols) + "\n")


def write_csv(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=header)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (round(v, 4) if isinstance(v, float) and _ok(v) else ("" if isinstance(v, float) else v))
                         for k, v in r.items()})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*", default=None, help="Run folder names under OUT_ROOT/evals (default: all, smoke runs skipped)")
    ap.add_argument("--evals", type=Path, default=None, help="Default OUT_ROOT/evals")
    ap.add_argument("--precisions", type=float, nargs="+", default=[0.80, 0.85, 0.90])
    ap.add_argument("--out", type=Path, default=None, help="Default OUT_ROOT/results")
    args = ap.parse_args(argv)

    summaries = load_runs(args.evals or paths.out_root() / "evals", args.runs)
    if not summaries:
        raise SystemExit("no finished runs found (looked for evals/<run>/summary.json)")
    per_run = [run_metrics(s, args.precisions) for s in summaries]
    agg = aggregate(per_run)
    out = args.out or paths.out_root() / "results"
    write_csv(per_run, out / "compare_runs.csv")
    md = markdown(agg, args.precisions)
    (out / "compare_runs.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"runs: {', '.join(r['run'] for r in per_run)}")
    print(f"wrote {out / 'compare_runs.csv'} and {out / 'compare_runs.md'}")


if __name__ == "__main__":
    main()
