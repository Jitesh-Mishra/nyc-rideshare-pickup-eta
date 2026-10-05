"""Hand-drawn SVG diagrams for the README, light + dark variants (GitHub <picture> switches them).

python scripts/make_diagrams.py   ->  docs/img/<name>-light.svg, docs/img/<name>-dark.svg
Numbers are read from reports/ so the figures never drift from the results.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "img"
REP = ROOT / "reports"

THEMES = {
    "light": dict(bg="#ffffff", card="#f6f8fa", border="#d0d7de", ink="#1f2328", ink2="#59636e", faint="#e6e9ed",
                  a="#2a78d6", a_soft="#e3eefb", b="#eb6834", b_soft="#fde8de", c="#1baf7a", c_soft="#dcf4ea",
                  muted="#afb8c1"),
    "dark": dict(bg="#0d1117", card="#161b22", border="#30363d", ink="#e6edf3", ink2="#9198a1", faint="#21262d",
                 a="#3987e5", a_soft="#13263d", b="#d95926", b_soft="#3a1d12", c="#199e70", c_soft="#0f2e23",
                 muted="#484f58"),
}
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class SVG:
    def __init__(self, w, h, t, label):
        self.w, self.h, self.t, self.label, self.parts = w, h, t, label, []

    def add(self, s):
        self.parts.append(s)

    def text(self, x, y, s, size=13, color=None, weight=400, anchor="start", mono=False, italic=False):
        self.add(f'<text x="{x}" y="{y}" font-family="{MONO if mono else FONT}" font-size="{size}" '
                 f'font-weight="{weight}" fill="{color or self.t["ink"]}" text-anchor="{anchor}"'
                 f'{" font-style=\"italic\"" if italic else ""}>{esc(s)}</text>')

    def rect(self, x, y, w, h, fill, stroke=None, rx=10, dash=None, sw=1.5):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
                 f'stroke="{stroke or "none"}" stroke-width="{sw}"{d}/>')

    def line(self, x1, y1, x2, y2, color=None, sw=1.5, dash=None, arrow=False, marker="ar"):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        m = f' marker-end="url(#{marker})"' if arrow else ""
        self.add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color or self.t["ink2"]}" '
                 f'stroke-width="{sw}" stroke-linecap="round"{d}{m}/>')

    def path(self, d, color=None, sw=1.5, dash=None, arrow=False, marker="ar"):
        da = f' stroke-dasharray="{dash}"' if dash else ""
        m = f' marker-end="url(#{marker})"' if arrow else ""
        self.add(f'<path d="{d}" fill="none" stroke="{color or self.t["ink2"]}" stroke-width="{sw}" '
                 f'stroke-linecap="round" stroke-linejoin="round"{da}{m}/>')

    def circle(self, x, y, r, fill, stroke=None, sw=1.5):
        self.add(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke or "none"}" stroke-width="{sw}"/>')

    def pill(self, x, y, s, fill, color, size=11.5, anchor="middle", pad=9):
        w = len(s) * size * 0.58 + 2 * pad
        x0 = x - w / 2 if anchor == "middle" else x
        self.rect(x0, y - size - 3, w, size + 10, fill, rx=(size + 10) / 2)
        self.text(x0 + w / 2, y + 1, s, size=size, color=color, weight=600, anchor="middle")

    def node(self, x, y, w, h, title, sub=None, accent=None, dash=None, mono_title=True):
        t = self.t
        fill = t[f"{accent}_soft"] if accent else t["card"]
        stroke = t[accent] if accent else t["border"]
        self.rect(x, y, w, h, fill, stroke, dash=dash)
        cy = y + h / 2 + (-4 if sub else 5)
        self.text(x + w / 2, cy, title, size=14, weight=650, anchor="middle", mono=mono_title)
        if sub:
            for i, s in enumerate(sub if isinstance(sub, list) else [sub]):
                self.text(x + w / 2, cy + 19 + i * 16, s, size=12, color=t["ink2"], anchor="middle")

    def render(self):
        t = self.t
        defs = (f'<defs>'
                f'<marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
                f'<path d="M0,0 L10,5 L0,10 z" fill="{t["ink2"]}"/></marker>'
                f'<marker id="ara" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
                f'<path d="M0,0 L10,5 L0,10 z" fill="{t["a"]}"/></marker>'
                f'<marker id="arb" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
                f'<path d="M0,0 L10,5 L0,10 z" fill="{t["b"]}"/></marker></defs>')
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" '
                f'height="{self.h}" role="img" aria-label="{esc(self.label)}">{defs}'
                f'<rect width="{self.w}" height="{self.h}" rx="14" fill="{t["bg"]}" stroke="{t["border"]}"/>'
                + "".join(self.parts) + "</svg>")


def header(s, title, sub):
    s.text(32, 42, title, size=19, weight=700)
    s.text(32, 64, sub, size=13, color=s.t["ink2"])


# ----------------------------------------------------------------------------- 1 architecture
def architecture(t):
    par = json.loads((REP / "parity.json").read_text())
    lat = json.loads((REP / "latency_lgb.json").read_text())["runs"][-1]
    s = SVG(1080, 560, t, "Architecture: one features module feeds both the offline training path and the "
                          "online serving path; parity is tested between them.")
    header(s, "One feature module, two execution paths",
           "Training and serving call the same derive(); a replay test proves they agree.")
    # lanes
    s.rect(400, 92, 660, 196, t["card"], rx=12, sw=0)
    s.rect(400, 300, 660, 238, t["card"], rx=12, sw=0)
    s.text(1044, 114, "OFFLINE  ·  train + evaluate", size=11, color=t["ink2"], weight=700, anchor="end")
    s.text(1044, 322, "ONLINE  ·  serve", size=11, color=t["ink2"], weight=700, anchor="end")
    # sources
    s.node(28, 245, 160, 86, "TLC trips", ["Lyft HV0005", "5.99M rows · Mar 2025"], mono_title=False)
    s.node(228, 150, 150, 80, "data.py", "label + audit")
    s.node(228, 380, 150, 80, "replay", ["event-time stream", "queries before events"])
    s.path("M188,288 C208,288 208,190 228,190", arrow=True)
    s.path("M188,288 C208,288 208,420 228,420", arrow=True)
    # features container
    s.rect(430, 128, 210, 368, t["a_soft"], t["a"], rx=14, sw=2)
    s.text(535, 156, "features.py", size=15, weight=700, anchor="middle", mono=True, color=t["a"])
    s.node(450, 172, 170, 86, "offline", ["numpy searchsorted", "per zone, vectorised"])
    s.node(450, 380, 170, 86, "online store", ["per-zone sorted buffers", "+ open-queue counters"])
    s.text(535, 296, "same window rules", size=12, color=t["ink2"], anchor="middle")
    s.text(535, 314, "same derive()", size=12, color=t["ink2"], anchor="middle", mono=True)
    s.line(500, 262, 500, 374, t["a"], 1.5, dash="4 4", arrow=True, marker="ara")
    s.line(570, 374, 570, 262, t["a"], 1.5, dash="4 4", arrow=True, marker="ara")
    s.pill(535, 352, f"parity {par['rows_matching']:,} / {par['n']:,}", t["a"], t["bg"], size=11)
    s.line(378, 190, 446, 205, arrow=True)
    s.text(410, 186, "trips", size=11, color=t["ink2"], anchor="middle")
    s.line(378, 420, 446, 423, arrow=True)
    s.text(412, 412, "events", size=11, color=t["ink2"], anchor="middle")
    # offline lane
    s.node(690, 150, 160, 96, "train.py", ["B0 · LightGBM", "PyTorch quantile"])
    s.node(880, 150, 160, 96, "evaluate", ["locked test week", "slices · switchback"])
    s.line(620, 215, 686, 200, arrow=True)
    s.text(653, 196, "19 features", size=11, color=t["ink2"], anchor="middle")
    s.line(850, 198, 876, 198, arrow=True)
    # online lane
    s.node(690, 360, 160, 86, "FastAPI /predict", [f"p99 {lat['client_roundtrip_ms']['p99']:.1f} ms",
                                                    f"@ {lat['rps_target']} RPS, local"], accent="a")
    s.node(880, 360, 160, 86, "P50 · P90", "minutes to arrival", mono_title=False)
    s.line(620, 423, 686, 403, arrow=True)
    s.text(653, 400, "as of t", size=11, color=t["ink2"], anchor="middle")
    s.line(850, 403, 876, 403, arrow=True)
    s.line(770, 246, 770, 356, arrow=True)
    s.text(778, 306, "models/", size=11, color=t["ink2"], mono=True)
    # fallback
    s.node(690, 470, 160, 52, "B0 fallback", None, dash="5 4")
    s.line(850, 496, 960, 496, dash="5 4")
    s.line(960, 496, 960, 450, dash="5 4", arrow=True)
    s.text(870, 488, "stale > 5 min", size=11, color=t["ink2"])
    s.text(870, 515, "or model error", size=11, color=t["ink2"])
    s.line(720, 446, 720, 466, dash="5 4", arrow=True)
    return s


# ----------------------------------------------------------------------------- 2 point in time
def point_in_time(t):
    s = SVG(1080, 420, t, "Point-in-time features: every window is [t minus w, t). Events at or after the "
                          "request time t are invisible, including the request itself.")
    header(s, "What the model is allowed to see at request time t",
           "Each feature counts events in one pickup zone over [t − w, t). Nothing at or after t leaks in.")
    x0, x1 = 240, 1040
    span = 72                      # minutes shown: t-62 .. t+10
    px = lambda m: x0 + (m + 62) / span * (x1 - x0)
    tx = px(0)
    rows = [("requests", "demand", 150), ("drop-offs", "supply proxy", 205),
            ("driver arrivals", "observed wait", 260), ("open queue", "still waiting at t", 334)]
    # windows
    for i, (w, lab) in enumerate([(60, "60 min"), (15, "15 min"), (5, "5 min")]):
        y = 92 + i * 12
        s.line(px(-w), y, tx - 2, y, t["a"], 2)
        s.line(px(-w), y - 4, px(-w), y + 4, t["a"], 2)
        s.text(px(-w) - 6, y + 4, lab, size=11, color=t["a"], anchor="end", weight=600)
    s.rect(px(-15), 122, tx - px(-15), 236, t["a_soft"], rx=4)
    s.rect(tx + 8, 122, x1 - tx - 10, 236, t["faint"], rx=4)
    s.text((tx + x1) / 2 + 4, 114, "invisible at t", size=11.5, color=t["ink2"], anchor="middle", weight=600)
    # rows
    import random
    rnd = random.Random(7)
    for name, sub, y in rows:
        s.text(32, y - 2, name, size=14, weight=650)
        s.text(32, y + 15, sub, size=12, color=t["ink2"])
        if name == "open queue":
            continue
        s.line(x0, y + 4, x1, y + 4, t["faint"], 1)
        n = {"requests": 34, "drop-offs": 26, "driver arrivals": 22}[name]
        for _ in range(n):
            m = rnd.uniform(-61, 9.5)
            if abs(m) < 0.4:
                continue
            vis = m < 0
            r = 5
            if vis:
                s.circle(px(m), y + 4, r, t["ink2"] if not -15 <= m else t["a"])
            else:
                s.circle(px(m), y + 4, r, t["bg"], t["muted"], 1.5)
    # open queue intervals: (request, arrival) in minutes
    iv = [(-30, -24), (-21, -12), (-9, -3), (-7, 2.5), (-4, 6), (-2, 1), (-14, -10), (-44, -37), (-52, -49), (3, 8)]
    for i, (a, b) in enumerate(iv):
        y = 310 + (i % 4) * 13
        open_ = a < 0 <= b
        col = t["b"] if open_ else (t["muted"] if a >= 0 else t["ink2"])
        s.line(px(a), y, px(min(b, 9.5)), y, col, 3.5 if open_ else 2.5)
        s.circle(px(a), y, 3.5, col)
    s.pill(px(-28), 294, "3 open at t · mean age 4.3 min", t["b_soft"], t["b"], size=11)
    # t line
    s.line(tx, 82, tx, 364, t["b"], 2)
    s.pill(tx, 82, "request at t", t["b"], t["bg"], size=11.5)
    # axis
    s.line(x0, 376, x1, 376, t["border"], 1)
    for m in (-60, -45, -30, -15, -5, 0, 10):
        s.line(px(m), 372, px(m), 380, t["border"], 1)
        s.text(px(m), 396, "t" if m == 0 else f"t{m:+d}m".replace("-", "−"), size=11, color=t["ink2"], anchor="middle", mono=True)
    return s


# ----------------------------------------------------------------------------- 3 headroom
def headroom(t):
    h = json.loads((REP / "headroom.json").read_text())
    b0, lgb, orc = h["B0_zone_how_median_mae_same_rows"], h["M1_lightgbm_mae_same_rows"], h["oracle_mae"]
    share = h["lgb_share_of_headroom_captured"]
    s = SVG(1080, 300, t, f"Headroom: baseline MAE {b0:.3f}, LightGBM {lgb:.3f}, future-label oracle {orc:.3f}. "
                          f"LightGBM captures {share:.0%} of the attainable gap.")
    header(s, "How much of the attainable gain the model captures",
           "Holdout MAE in minutes. The oracle knows each zone's real median wait for the same 10 minutes (future data).")
    lo, hi, x0, x1 = 1.60, 2.00, 70, 1010
    px = lambda v: x0 + (v - lo) / (hi - lo) * (x1 - x0)
    y = 170
    s.line(x0, y, x1, y, t["faint"], 10)
    s.line(px(orc), y, px(b0), y, t["c_soft"], 10)
    s.line(px(lgb), y, px(b0), y, t["c"], 10)
    for v in (1.6, 1.7, 1.8, 1.9, 2.0):
        s.line(px(v), y + 30, px(v), y + 36, t["border"], 1)
        s.text(px(v), y + 52, f"{v:.1f}", size=11, color=t["ink2"], anchor="middle", mono=True)
    for v, lab, col, up, ring in [(b0, "B0 baseline", t["ink"], True, False), (lgb, "LightGBM", t["c"], True, False),
                                  (orc, "oracle (uses future)", t["ink2"], True, True)]:
        if ring:
            s.circle(px(v), y, 9, t["bg"], col, 2.5)
        else:
            s.circle(px(v), y, 9, col)
        s.text(px(v), y - 22, f"{v:.3f}", size=14, weight=700, anchor="middle", mono=True, color=col)
        s.text(px(v), y - 40, lab, size=12, anchor="middle", color=t["ink2"])
    # brackets
    yb = y + 72
    s.path(f"M{px(orc)},{yb-6} L{px(orc)},{yb} L{px(b0)},{yb} L{px(b0)},{yb-6}", t["ink2"], 1.2)
    s.text((px(orc) + px(b0)) / 2, yb + 18, f"attainable: {(b0-orc)/b0:.0%} of baseline error", size=12,
           anchor="middle", color=t["ink2"])
    s.pill((px(lgb) + px(b0)) / 2, y + 4, f"captured {share:.0%}", t["c"], t["bg"], size=11)
    return s


# ----------------------------------------------------------------------------- 4 switchback
def switchback(t):
    sw = json.loads((REP / "switchback.json").read_text())
    aa, ab = sw["aa"], sw["ab"]
    s = SVG(1080, 380, t, f"Switchback replay: borough by 30-minute blocks randomised to arms. Cluster-robust "
                          f"SE is {aa['ratio_cluster_to_naive_se']:.1f} times the naive SE.")
    header(s, "Randomise blocks, not riders",
           "Errors move together inside a borough × 30-min block, so request-level standard errors are far too small.")
    import random
    rnd = random.Random(3)
    bors = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Is."]
    gx, gy, cw, ch = 140, 110, 26, 30
    for i, b in enumerate(bors):
        s.text(gx - 12, gy + i * ch + 20, b, size=12, anchor="end", color=t["ink2"])
        for j in range(16):
            arm = rnd.random() < 0.5
            s.rect(gx + j * cw, gy + i * ch, cw - 3, ch - 3, t["a"] if arm else t["b"], rx=4)
    for j, lab in enumerate(["16:00", "17:00", "18:00", "19:00", "20:00", "21:00", "22:00", "23:00"]):
        s.text(gx + j * 2 * cw, gy + 5 * ch + 18, lab, size=10.5, color=t["ink2"], mono=True)
    s.rect(gx, gy + 5 * ch + 32, 12, 12, t["a"], rx=3)
    s.text(gx + 18, gy + 5 * ch + 42, "arm A · B0 baseline", size=12, color=t["ink2"])
    s.rect(gx + 175, gy + 5 * ch + 32, 12, 12, t["b"], rx=3)
    s.text(gx + 193, gy + 5 * ch + 42, "arm B · LightGBM", size=12, color=t["ink2"])
    s.text(gx, gy + 5 * ch + 66, f"{sw['n_blocks']:,} blocks over the test week · {sw['n_requests']:,} requests",
           size=12, color=t["ink2"])
    # CI panel
    px0, px1, lo, hi = 640, 1040, -0.20, 0.0
    px = lambda v: px0 + (v - lo) / (hi - lo) * (px1 - px0)
    s.text(px0, 112, "A/B effect on MAE (min), 95% CI", size=13, weight=650)
    for k, (lab, ci, col) in enumerate([("naive (iid requests)", ab["ci95_naive"], t["ink2"]),
                                        ("cluster-robust", ab["ci95_cluster"], t["b"])]):
        y = 160 + k * 56
        s.text(px0, y - 14, lab, size=12, color=t["ink2"])
        s.line(px(ci[0]), y, px(ci[1]), y, col, 4)
        s.line(px(ci[0]), y - 7, px(ci[0]), y + 7, col, 2)
        s.line(px(ci[1]), y - 7, px(ci[1]), y + 7, col, 2)
        s.circle(px(ab["effect_mae_lgb_minus_b0_min"]), y, 5.5, col)
    s.line(px(0), 132, px(0), 236, t["border"], 1, dash="3 3")
    for v in (-0.2, -0.15, -0.1, -0.05, 0.0):
        s.text(px(v), 252, f"{v:.2f}".replace("-", "−"), size=10.5, color=t["ink2"], anchor="middle", mono=True)
    s.pill(px0 + 96, 197, f"SE ×{aa['ratio_cluster_to_naive_se']:.1f} vs naive", t["b"], t["bg"], size=11, anchor="start")
    s.text(px0, 300, f"A/A replays (no true effect): naive test 'significant' in "
                     f"{aa['false_positive_rate_naive_95']:.0%},", size=12, color=t["ink2"])
    s.text(px0, 318, f"cluster-robust in {aa['false_positive_rate_cluster_95']:.1%} (nominal 5%).", size=12,
           color=t["ink2"])
    return s


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in [("architecture", architecture), ("point-in-time", point_in_time),
                     ("headroom", headroom), ("switchback", switchback)]:
        for theme, tok in THEMES.items():
            (OUT / f"{name}-{theme}.svg").write_text(fn(tok).render())
    print("wrote", sorted(p.name for p in OUT.glob("*.svg")))
