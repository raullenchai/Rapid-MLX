#!/usr/bin/env python3
"""Trio-Flash plays Tetris — zero-shot OOD decision demo.

Every piece = two classify calls to the local demo server:
  1. rotation  (candidates A-D)
  2. landing column (candidates A-J)
The model has never been trained on Tetris — this measures zero-shot
spatial decision making through the exact API customers get.

Usage:
  python3 play_tetris.py --blocks 40 --gif out.gif [--base http://localhost:8123]
"""
import argparse, json, time, urllib.request, random
from pathlib import Path

W, H = 10, 20
PIECES = {
    "I": [[[1,1,1,1]], [[1],[1],[1],[1]]],
    "O": [[[1,1],[1,1]]],
    "T": [[[1,1,1],[0,1,0]], [[1,0],[1,1],[1,0]], [[0,1,0],[1,1,1]], [[0,1],[1,1],[0,1]]],
    "S": [[[0,1,1],[1,1,0]], [[1,0],[1,1],[0,1]]],
    "Z": [[[1,1,0],[0,1,1]], [[0,1],[1,1],[1,0]]],
    "J": [[[1,0,0],[1,1,1]], [[1,1],[1,0],[1,0]], [[1,1,1],[0,0,1]], [[0,1],[1,1],[1,1]]],
    "L": [[[0,0,1],[1,1,1]], [[1,0],[1,0],[1,1]], [[1,1,1],[1,0,0]], [[1,1],[0,1],[0,1]]],
}
COLS = "ABCDEFGHIJ"

def classify(base, tok, prompt, cands):
    req_data = json.dumps({"prompt": prompt, "candidates": cands}).encode()
    for attempt in range(8):
        req = urllib.request.Request(base + "/v1/classify", req_data,
            {"Content-Type": "application/json", "Authorization": "Bearer " + tok})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                d = json.loads(r.read())
            break
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 7:
                retry = float(e.headers.get("Retry-After", 5))
                time.sleep(retry + 0.2)
                continue
            raise
    idx = d.get("choice_index")
    if idx is None:
        letter = (d.get("decision") or "A").strip().split(".")[0].upper()
        idx = COLS.index(letter) if letter in COLS else 0
    return idx, d["confidence"], d.get("disposition", ""), (time.time()-t0)*1000

def heights_and_holes(b):
    heights, holes = [], 0
    for x in range(W):
        col = [b[y][x] for y in range(H)]
        top = next((y for y, v in enumerate(col) if v), H)
        heights.append(H - top if top < H else 0)
        holes += sum(1 for y in range(top, H) if not col[y])
    return heights, holes

def drop(b, shape, x0):
    """Landing row for shape placed at column x0 (top rows free)."""
    sh = len(shape); sw = len(shape[0])
    for y in range(H - sh + 1):
        ok = True
        for r in range(sh):
            for c in range(sw):
                if shape[r][c] and (y + r + 1 >= H or b[y + r + 1][x0 + c]):
                    ok = False; break
            if not ok: break
        if ok:
            return y
    return None

def place(b, shape, x0, y):
    nb = [row[:] for row in b]
    for r in range(len(shape)):
        for c in range(len(shape[0])):
            if shape[r][c]:
                nb[y + r][x0 + c] = 1
    return nb

def clear_lines(b):
    kept = [row for row in b if any(v == 0 for v in row)]
    cleared = H - len(kept)
    return [[0]*W for _ in range(cleared)] + kept, cleared

def drop_features(b, shape, x0):
    """(land_height, opens_holes, clears) for hard-drop at column x0."""
    y = drop(b, shape, x0)
    if y is None:
        return None
    sh, sw = len(shape), len(shape[0])
    nb = place(b, shape, x0, y)
    nb2, cleared = clear_lines(nb)
    opens = 0
    for c in range(sw):
        for r in range(len(shape) - 1, -1, -1):
            if shape[r][c]:
                for yy in range(y + r + 1, H):
                    if not b[yy][x0 + c]:
                        opens += 1
                        break
                break
    land_h = H - y
    return land_h, opens, cleared

def render_frame(b, piece, score, lines, idx):
    """PNG frame via PIL, scaled."""
    from PIL import Image, ImageDraw
    CELL, PAD = 22, 14
    img = Image.new("RGB", (W*CELL + 2*PAD + 240, H*CELL + 2*PAD), (16, 18, 24))
    dr = ImageDraw.Draw(img)
    dr.rectangle([PAD-2, PAD-2, PAD + W*CELL + 1, PAD + H*CELL + 1], outline=(90, 95, 110), width=2)
    for y in range(H):
        for x in range(W):
            if b[y][x]:
                dr.rectangle([PAD + x*CELL + 1, PAD + y*CELL + 1, PAD + (x+1)*CELL - 2, PAD + (y+1)*CELL - 2], fill=(70, 200, 140))
    tx = PAD + W*CELL + 24
    dr.text((tx, PAD + 8), f"TRIO-FLASH  (zero-shot)", fill=(230, 230, 235))
    dr.text((tx, PAD + 30), f"piece: {piece}", fill=(160, 200, 255))
    dr.text((tx, PAD + 52), f"decision #{idx}", fill=(160, 200, 255))
    dr.text((tx, PAD + 90), f"score: {score}", fill=(250, 220, 120))
    dr.text((tx, PAD + 112), f"lines: {lines}", fill=(250, 220, 120))
    return img

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8123")
    ap.add_argument("--token", default="demotok777")
    ap.add_argument("--blocks", type=int, default=40)
    ap.add_argument("--gif", default="")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    random.seed(a.seed)
    b = [[0]*W for _ in range(H)]
    score = lines_total = 0
    frames = []; lat = []; decisions = 0
    seq = [random.choice(list(PIECES)) for _ in range(a.blocks)]
    t0 = time.time()
    for n, p in enumerate(seq):
        def feat(f):
            h, o, c = f
            return (f"lands at height {h}"
                    + (f", clears {c} line{'s' if c>1 else ''}" if c else "")
                    + (f", opens {o} hole{'s' if o>1 else ''}" if o else ", no new holes"))
        # decision 1: orientation — 2-way semantic trade-off (training-shaped)
        rots = PIECES[p]
        wide = [i for i, sh in enumerate(rots) if len(sh[0]) >= len(sh)]
        tall = [i for i, sh in enumerate(rots) if len(sh[0]) < len(sh)]
        wi = wide[0] if wide else 0; ti = tall[0] if tall else wi
        heights, holes = heights_and_holes(b)
        def span_cost(sh):
            best = None
            for x in range(W - len(sh[0]) + 1):
                f = drop_features(b, sh, x)
                if f and (best is None or (f[1], f[0]) < (best[1], best[0])):
                    best = f
            return best or (H, 0, 0)
        cw = span_cost(rots[wi]); ct = span_cost(rots[ti])
        rot_p = (f"Tetris piece {p}. FLAT orientation: {feat(cw)}. TALL orientation: {feat(ct)}. "
                 "Which orientation? Prefer clears and a low stack; avoid new holes.")
        opts = [f"A. FLAT: {feat(cw)}", f"B. TALL: {feat(ct)}"]
        ri, rc, rd, rms = classify(a.base, a.token, rot_p, opts)
        shape = rots[ti] if (ri == 1 and tall) else rots[wi]
        decisions += 1; lat.append(rms)
        time.sleep(2.2)  # stay under server rate limit (30/IP/min)
        sw = len(shape[0])
        # decision 2: which half (2-way), engine picks the exact column
        featsL = [f for f in (drop_features(b, shape, x) for x in range(0, max(1, min(5, W - sw + 1)))) if f]
        featsR = [f for f in (drop_features(b, shape, x) for x in range(5, W - sw + 1))] if W - sw + 1 > 5 else []
        bestL = min((f for f in featsL if f), key=lambda f: (f[1], f[0])) if any(featsL) else None
        bestR = min((f for f in featsR if f), key=lambda f: (f[1], f[0])) if any(featsR) else None
        half_p = (f"Tetris landing for piece {p}. LEFT half: {feat(bestL) if bestL else 'FULL'}. "
                  f"RIGHT half: {feat(bestR) if bestR else 'FULL'}. Which half? Prefer clears and low stack.")
        opts2 = ([f"A. LEFT: {feat(bestL)}"] if bestL else []) + (["B. RIGHT: " + feat(bestR)] if bestR else [])
        if not opts2:
            print(f"piece {n+1} ({p}) nowhere legal — game over"); break
        hi, hc, hd, hms = classify(a.base, a.token, half_p, opts2)
        decisions += 1; lat.append(hms)
        time.sleep(2.2)
        poolL = [f for f in featsL if f]
        poolR = [f for f in featsR if f]
        pool = poolL if (hi == 0 and poolL) else (poolR if poolR else poolL)
        xs = (list(range(0, max(1, min(5, W - sw + 1)))) if pool is poolL else list(range(5, W - sw + 1)))
        if not pool:
            print(f"piece {n+1} ({p}) board full — game over"); break
        best_i = min(range(len(pool)), key=lambda k: (pool[k][1], pool[k][0]))
        ci_abs = xs[best_i]
        y = drop(b, shape, ci_abs)
        if y is None:
            print(f"piece {n+1} ({p}) illegal at col {ci_abs} — game over")
            frames.append(render_frame(b, f"{p}→col{ci_abs} ✗", score, lines_total, n+1))
            break
        b = clear_lines(place(b, shape, ci_abs, y))[0]
        # scoring: lines + a bit for low stack
        _, cleared = clear_lines(place(b, shape, ci_abs, y))
        lines_total += cleared; score += cleared * 100
        frames.append(render_frame(b, f"{p} rot{ri} col{ci_abs} ✓", score, lines_total, n+1))
        print(f"[{n+1:3d}/{a.blocks}] {p} rot={ri} col={ci_abs:2d} lines={cleared} total={lines_total} "
              f"conf={rc:.2f}/{hc:.2f} disp={rd} ({rms:.0f}/{hms:.0f}ms)")
    dt = time.time() - t0
    hs, hl = heights_and_holes(b)
    print(f"\n=== GAME OVER === blocks={n+1} lines={lines_total} score={score} "
          f"final_height={max(hs)} holes={hl} decisions={decisions} "
          f"lat p50={sorted(lat)[len(lat)//2]:.0f}ms wall={dt:.0f}s")
    if a.gif:
        frames[0].save(a.gif, save_all=True, append_images=frames[1:], duration=450, loop=0)
        print(f"replay gif: {a.gif} ({len(frames)} frames)")

if __name__ == "__main__":
    main()
