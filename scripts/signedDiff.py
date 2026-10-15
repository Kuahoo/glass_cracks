import cv2, numpy as np, glob, os

ROOT = os.path.expanduser(os.environ.get("AGDD_ROOT", "~/glass_crack/bundle/datasets/agdd_aircraft_glass"))
D = ROOT + "/data"
CRACK = "2"
files = sorted(glob.glob(D + "/labels_rect/**/*.txt", recursive=True))
print("label files:", len(files))


def box_px(g, cx, cy, w, h, m=0):
    H, W = g.shape
    return (max(int((cx - w / 2) * W) - m, 0), max(int((cy - h / 2) * H) - m, 0),
            min(int((cx + w / 2) * W) + m, W), min(int((cy + h / 2) * H) + m, H))


def signed_contrast(g, cx, cy, w, h, pad=15):
    x0, y0, x1, y1 = box_px(g, cx, cy, w, h)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None
    H, W = g.shape
    inner = g[y0:y1, x0:x1].astype(np.float32)
    outer = g[max(y0 - pad, 0):min(y1 + pad, H), max(x0 - pad, 0):min(x1 + pad, W)].astype(np.float32)
    ring = (outer.sum() - inner.sum()) / max(outer.size - inner.size, 1)
    return (inner.mean() - ring) / (inner.std() + 2.0)


def load(view, stem):
    p = glob.glob(f"{D}/{view}/**/{stem}.*", recursive=True)
    return cv2.imread(p[0], 0) if p else None


res = {"images": [], "image": []}
crops = []
for lf in files:
    stem = os.path.splitext(os.path.basename(lf))[0]
    for line in open(lf):
        t = line.split()
        if len(t) != 5 or t[0] != CRACK:
            continue
        box = list(map(float, t[1:]))
        row = []
        for view in res:
            g = load(view, stem)
            if g is None:
                continue
            c = signed_contrast(g, *box)
            if c is not None:
                res[view].append(c)
            if len(crops) < 6:
                x0, y0, x1, y1 = box_px(g, *box, m=40)
                row.append(cv2.equalizeHist(cv2.resize(g[y0:y1, x0:x1], (200, 200))))
        if len(row) == 2 and len(crops) < 6:
            crops.append(np.hstack(row))

for v, r in res.items():
    r = np.array(r)
    if r.size:
        print(v, "boxes", len(r), "median signed contrast", np.median(r).round(3),
              "fraction darker-than-surround", (r < 0).mean().round(2))
    else:
        print(v, "no boxes scored")
if crops:
    out = os.environ.get("CROPS_OUT", "agdd_views_crops.png")
    cv2.imwrite(out, np.vstack(crops))
    print("wrote", out, "(left: images/, right: image/)")
