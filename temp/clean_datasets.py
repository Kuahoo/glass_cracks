#!/usr/bin/env python3
"""Build deduplicated, leakage-aware training sets from the downloaded bundle.

Reads glass_crack_dataset_bundle/datasets/ (never modified) and writes:
  processed/glass_merged_seg/      GDD2 + 2026 glass merged, deduplicated, re-split by near-duplicate group
  processed/ncugd_external_test/   NCUGD deduplicated, remapped to the merged class ids, test-only
  processed/fcd_108_grouped/       FCD-108 re-split so every facade sits in exactly one split
  processed/cleaning_report.json   counts, label fixes, and leakage checks
"""
from __future__ import annotations
import argparse
import collections
import csv
import json
import random
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
RF_NAME = re.compile(r"^(.*?)(?:_(?:jpg|jpeg|png))+\.rf\.[0-9a-f]+$", re.I)
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

MERGED_NAMES = ["bubble", "crack", "inclusion", "scratch"]
GDD2_REMAP = {0: 0, 1: 1, 2: 2, 3: 3}
GSEG_REMAP = {0: 0, 1: 1, 2: 3}
NCUGD_REMAP = {0: 1}
SYNTHETIC_PREFIXES = ("Gemini_Generated_Image",)
HASH_THRESHOLD = 12


def source_key(stem):
    m = RF_NAME.match(stem)
    return m.group(1) if m else stem


def fix_yolo_lines(lines, remap):
    """Return cleaned YOLO-seg lines and a Counter of applied fixes.

    Box lines (4 coords) become 4-corner polygons so polygon and box rows never mix,
    coordinates are clipped to [0, 1], and degenerate or duplicate polygons are dropped.
    """
    fixes = collections.Counter()
    out, seen = [], set()
    for raw in lines:
        tok = raw.split()
        if not tok:
            continue
        cls = int(float(tok[0]))
        if cls not in remap:
            fixes["unknown_class_dropped"] += 1
            continue
        vals = [float(v) for v in tok[1:]]
        if len(vals) == 4:
            x, y, w, h = vals
            vals = [x - w / 2, y - h / 2, x + w / 2, y - h / 2, x + w / 2, y + h / 2, x - w / 2, y + h / 2]
            fixes["box_to_polygon"] += 1
        if len(vals) % 2:
            fixes["odd_coordinate_count_dropped"] += 1
            continue
        clipped = [min(1.0, max(0.0, v)) for v in vals]
        if clipped != vals:
            fixes["clipped_to_image"] += 1
        pts = np.array(clipped).reshape(-1, 2)
        if len(pts) < 3:
            fixes["too_few_points_dropped"] += 1
            continue
        x, y = pts[:, 0], pts[:, 1]
        area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        if area < 1e-7:
            fixes["zero_area_dropped"] += 1
            continue
        line = f"{remap[cls]} " + " ".join(f"{v:.6f}" for v in clipped)
        if line in seen:
            fixes["duplicate_dropped"] += 1
            continue
        seen.add(line)
        out.append(line)
    return out, fixes


def image_hashes(path):
    im = Image.open(path).convert("L")
    a = np.asarray(im.resize((16, 16), Image.BILINEAR), dtype=np.float32)
    d = np.asarray(im.resize((17, 16), Image.BILINEAR), dtype=np.float32)
    return (a > a.mean()).flatten(), (d[:, 1:] > d[:, :-1]).flatten()


def near_duplicate_pairs(paths, threshold=HASH_THRESHOLD):
    """Index pairs whose average hash differs by <= threshold bits and difference hash by <= 2x threshold."""
    hashes = [image_hashes(p) for p in paths]
    a = np.array([h[0] for h in hashes])
    d = np.array([h[1] for h in hashes])
    pairs = []
    for i in range(len(paths) - 1):
        da = (a[i + 1:] != a[i]).sum(1)
        dd = (d[i + 1:] != d[i]).sum(1)
        for j in np.where((da <= threshold) & (dd <= 2 * threshold))[0]:
            pairs.append((i, i + 1 + int(j)))
    return pairs


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        self.parent[self.find(a)] = self.find(b)


def read_lines(path):
    return path.read_text(encoding="utf-8").splitlines() if path.exists() else []


def label_path(image_path):
    return image_path.parent.parent / "labels" / (image_path.stem + ".txt")


def write_yolo_yaml(path, names, splits, comment=None):
    lines = [f"# {comment}"] if comment else []
    for key, rel in splits.items():
        lines.append(f"{key}: {rel}")
    lines.append(f"nc: {len(names)}")
    lines.append("names: [" + ", ".join(names) + "]")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def reset_dir(path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def build_glass_merged(datasets, out, seed, report):
    gdd2, gseg = datasets / "glass_defect_detection2", datasets / "glass_segmentation_2026"
    by_key = {"gdd2": collections.defaultdict(list), "gseg": collections.defaultdict(list)}
    for tag, root, splits in (("gdd2", gdd2, ["train", "valid", "test"]), ("gseg", gseg, ["train", "valid"])):
        for split in splits:
            for p in sorted((root / split / "images").iterdir()):
                if p.suffix.lower() in IMAGE_EXTS:
                    by_key[tag][source_key(p.stem)].append((split, p))

    dropped_synthetic = sorted(k for k in set(by_key["gdd2"]) | set(by_key["gseg"]) if k.startswith(SYNTHETIC_PREFIXES))
    items = []
    for key in sorted(set(by_key["gdd2"]) | set(by_key["gseg"])):
        if key in dropped_synthetic:
            continue
        g, s = by_key["gdd2"].get(key, []), by_key["gseg"].get(key, [])
        if s and s[0][0] == "valid":
            chosen = [("gseg", s[0][1], "original")]
        elif s and g:
            ga = image_hashes(g[0][1])[0]
            best = min(s, key=lambda sp: int((image_hashes(sp[1])[0] != ga).sum()))
            chosen = [("gseg", best[1], "original")]
        elif s:
            chosen = [("gseg", p, "original_or_augmented") for _, p in s]
        else:
            chosen = [("gdd2", g[0][1], "original")]
        for tag, p, kind in chosen:
            items.append({"key": key, "tag": tag, "path": p, "kind": kind,
                          "in_gdd2": bool(g), "in_gseg": bool(s),
                          "source_split": p.parent.parent.name})

    uf = UnionFind(len(items))
    first = {}
    for i, it in enumerate(items):
        if it["key"] in first:
            uf.union(i, first[it["key"]])
        else:
            first[it["key"]] = i
    pairs = near_duplicate_pairs([it["path"] for it in items])
    for i, j in pairs:
        uf.union(i, j)
    groups = collections.defaultdict(list)
    for i in range(len(items)):
        groups[uf.find(i)].append(i)

    fixes_total = collections.Counter()
    for it in items:
        remap = GSEG_REMAP if it["tag"] == "gseg" else GDD2_REMAP
        it["labels"], f = fix_yolo_lines(read_lines(label_path(it["path"])), remap)
        it["fixes"] = f
        fixes_total += f

    n_sources = len({it["key"] for it in items})
    targets = {"valid": round(0.15 * n_sources), "test": round(0.15 * n_sources)}
    forced, eligible = [], []
    for gid, members in groups.items():
        (forced if any(items[i]["kind"] == "original_or_augmented" for i in members) else eligible).append(gid)
    rng = random.Random(seed)
    split_of = {gid: "train" for gid in forced}
    has_crack = lambda gid: any(l.split()[0] == "1" for i in groups[gid] for l in items[i]["labels"])
    filled = {"valid": 0, "test": 0}
    for stratum in (sorted(g for g in eligible if has_crack(g)), sorted(g for g in eligible if not has_crack(g))):
        rng.shuffle(stratum)
        share = len(stratum) / max(1, len(eligible))
        quota = {k: round(v * share) for k, v in targets.items()}
        taken = {"valid": 0, "test": 0}
        for gid in stratum:
            n = len({items[i]["key"] for i in groups[gid]})
            dest = min(("valid", "test"), key=lambda k: taken[k] / max(1, quota[k]))
            if taken[dest] < quota[dest]:
                split_of[gid] = dest
                taken[dest] += n
                filled[dest] += n
            else:
                split_of[gid] = "train"

    reset_dir(out)
    for split in ("train", "valid", "test"):
        (out / split / "images").mkdir(parents=True)
        (out / split / "labels").mkdir(parents=True)
    rows = []
    stats = {s: {"images": 0, "sources": set(), "empty_labels": 0, "instances": collections.Counter()} for s in ("train", "valid", "test")}
    for gid, members in groups.items():
        split = split_of[gid]
        for i in members:
            it = items[i]
            name = f"{it['tag']}__{it['path'].name}"
            shutil.copy2(it["path"], out / split / "images" / name)
            (out / split / "labels" / (Path(name).stem + ".txt")).write_text(
                "\n".join(it["labels"]) + ("\n" if it["labels"] else ""), encoding="utf-8")
            st = stats[split]
            st["images"] += 1
            st["sources"].add(it["key"])
            st["empty_labels"] += not it["labels"]
            for l in it["labels"]:
                st["instances"][MERGED_NAMES[int(l.split()[0])]] += 1
            rows.append({"split": split, "image": name, "source_key": it["key"], "group": gid,
                         "from_dataset": "glass_defect_detection2" if it["tag"] == "gdd2" else "glass_segmentation_2026",
                         "source_split": it["source_split"], "source_image": it["path"].name, "copy_kind": it["kind"],
                         "also_in_gdd2": it["in_gdd2"], "also_in_gseg": it["in_gseg"], "n_labels": len(it["labels"]),
                         "label_fixes": ";".join(f"{k}={v}" for k, v in sorted(it["fixes"].items()))})
    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["split"], r["image"])))
    write_yolo_yaml(out / "data.yaml", MERGED_NAMES,
                    {"train": "train/images", "val": "valid/images", "test": "test/images"},
                    "glass_defect_detection2 + glass_segmentation_2026, deduplicated and re-split by near-duplicate group")

    split_keys = {s: stats[s]["sources"] for s in stats}
    leaks = {f"{a}&{b}": len(split_keys[a] & split_keys[b]) for a, b in (("train", "valid"), ("train", "test"), ("valid", "test"))}
    cross_pairs = sum(1 for i, j in pairs if split_of[uf.find(i)] != split_of[uf.find(j)])
    report["glass_merged_seg"] = {
        "classes": MERGED_NAMES,
        "splits": {s: {"images": v["images"], "unique_sources": len(v["sources"]), "empty_labels": v["empty_labels"],
                       "instances": dict(v["instances"])} for s, v in stats.items()},
        "sources_total": n_sources,
        "shared_sources_gdd2_gseg": sum(1 for k in first if by_key["gdd2"].get(k) and by_key["gseg"].get(k)),
        "dropped_synthetic_sources": dropped_synthetic,
        "near_duplicate_pairs": len(pairs),
        "groups": len(groups),
        "largest_group": max(len({items[i]["key"] for i in m}) for m in groups.values()),
        "groups_forced_to_train_augmented": len(forced),
        "label_fixes": dict(fixes_total),
        "check_source_overlap_between_splits": leaks,
        "check_near_duplicate_pairs_across_splits": cross_pairs,
    }
    assert not any(leaks.values()) and cross_pairs == 0, "leakage remains in glass_merged_seg"
    return [it["path"] for it in items]


def build_ncugd(datasets, out, merged_paths, report):
    root = datasets / "ncugd_glass_crack"
    by_key = collections.defaultdict(list)
    for split in ("train", "valid", "test"):
        for p in sorted((root / split / "images").iterdir()):
            by_key[source_key(p.stem)].append(p)
    reset_dir(out / "test" / "images")
    (out / "test" / "labels").mkdir(parents=True)
    fixes_total, instances, kept = collections.Counter(), 0, []
    for key in sorted(by_key, key=lambda k: (len(k), k)):
        p = by_key[key][0]
        labels, f = fix_yolo_lines(read_lines(label_path(p)), NCUGD_REMAP)
        fixes_total += f
        instances += len(labels)
        shutil.copy2(p, out / "test" / "images" / f"ncugd__{p.name}")
        (out / "test" / "labels" / f"ncugd__{p.stem}.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
        kept.append(p)
    write_yolo_yaml(out / "data.yaml", MERGED_NAMES,
                    {"train": "test/images", "val": "test/images", "test": "test/images"},
                    "Evaluation only: every key points at the single test split. Class ids match glass_merged_seg (glass-crack -> crack).")
    overlap = sum(1 for i, j in near_duplicate_pairs(kept + list(merged_paths)) if (i < len(kept)) != (j < len(kept)))
    report["ncugd_external_test"] = {
        "images": len(kept), "duplicate_copies_removed": sum(len(v) - 1 for v in by_key.values()),
        "crack_instances": instances, "label_fixes": dict(fixes_total),
        "check_near_duplicates_with_glass_merged": overlap,
    }


def corner_black_fraction(path, k=24):
    a = np.asarray(Image.open(path).convert("L"))
    return float(np.mean([(c < 10).mean() for c in (a[:k, :k], a[:k, -k:], a[-k:, :k], a[-k:, -k:])]))


def build_fcd(datasets, out, report):
    root = datasets / "fcd_108_facade" / "v1"
    facade_of = lambda key: re.sub(r"_v\d$", "", key)
    copies = {"train": collections.defaultdict(list), "valid": collections.defaultdict(list)}
    for split in copies:
        for p in sorted((root / split).glob("*.jpg")):
            copies[split][source_key(p.stem)].append(p)
    variants = collections.defaultdict(lambda: {"train": [], "valid": []})
    for split, d in copies.items():
        for key in d:
            variants[facade_of(key)][split].append(key)
    valid_facades = sorted(f for f, v in variants.items() if len(v["valid"]) >= 2)

    for split in ("train", "valid"):
        reset_dir(out / split)
        shutil.copy2(root / split / "_classes.csv", out / split / "_classes.csv")
    for name in ("README.dataset.txt", "README.roboflow.txt"):
        shutil.copy2(root / name, out / name)
    counts = collections.Counter()
    for facade, v in variants.items():
        dest = "valid" if facade in valid_facades else "train"
        for split in ("train", "valid"):
            for key in v[split]:
                ps = copies[split][key]
                if dest == "valid" and split == "train":
                    ps = [min(ps, key=corner_black_fraction)]
                for p in ps:
                    shutil.copy2(p, out / dest / p.name)
                    shutil.copy2(p.with_name(p.stem + "_mask.png"), out / dest / (p.stem + "_mask.png"))
                    counts[dest] += 1
    report["fcd_108_grouped"] = {
        "valid_facades": valid_facades,
        "images": dict(counts),
        "facades": {"train": len(variants) - len(valid_facades), "valid": len(valid_facades)},
        "note": "Valid facades keep all 3 photo variants; train-origin variants contribute only the unaugmented copy "
                "(the one without black rotation corners). Masks are coarse brush strokes.",
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", default=str(ROOT / "glass_crack_dataset_bundle"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    bundle = Path(args.bundle).resolve()
    datasets, processed = bundle / "datasets", bundle / "processed"
    processed.mkdir(exist_ok=True)
    report = {"seed": args.seed, "hash_threshold_bits": HASH_THRESHOLD}
    merged_paths = build_glass_merged(datasets, processed / "glass_merged_seg", args.seed, report)
    build_ncugd(datasets, processed / "ncugd_external_test", merged_paths, report)
    build_fcd(datasets, processed / "fcd_108_grouped", report)
    (processed / "cleaning_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
