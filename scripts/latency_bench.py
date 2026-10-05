"""Local load test: one uvicorn worker, store warmed to a replay instant, open-loop at fixed RPS.

python scripts/latency_bench.py --rps 100 --n 6000
Reports client-side round-trip and server-side compute latency (p50/p95/p99) to reports/latency.json.
"""
import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eta.data import load_zones  # noqa: E402


def pct(a):
    a = np.asarray(a)
    return {"p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)), "max": float(a.max())}


async def run(url, zones, rps, n, ts):
    lat, srv, fallbacks, errors = [], [], 0, 0
    rng = np.random.default_rng(0)
    async with httpx.AsyncClient(timeout=5) as c:
        for _ in range(50):  # warm-up, not measured
            await c.post(f"{url}/predict", json={"zone": int(rng.choice(zones)), "request_ts": ts})

        async def one(z):
            nonlocal fallbacks, errors
            t0 = time.perf_counter()
            try:
                r = await c.post(f"{url}/predict", json={"zone": int(z), "request_ts": ts})
                lat.append((time.perf_counter() - t0) * 1000)
                j = r.json(); srv.append(j["server_ms"]); fallbacks += j["fallback"]
            except Exception:
                errors += 1

        tasks, start = [], time.perf_counter()
        for i, z in enumerate(rng.choice(zones, n)):
            delay = start + i / rps - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            tasks.append(asyncio.create_task(one(z)))
        await asyncio.gather(*tasks)
        wall = time.perf_counter() - start
    return {"rps_target": rps, "rps_achieved": n / wall, "n": n, "errors": errors, "fallbacks": fallbacks,
            "client_roundtrip_ms": pct(lat), "server_compute_ms": pct(srv)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rps", type=int, nargs="+", default=[50, 100, 200])
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--model", default="lgb")
    ap.add_argument("--at", default="2025-03-27T18:00:00")
    a = ap.parse_args()
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "ETA_MODEL": a.model, "ETA_REPLAY_UNTIL": a.at}
    srv = subprocess.Popen([sys.executable, "-m", "uvicorn", "eta.serve:app", "--port", "8765",
                            "--log-level", "warning"], env=env, cwd=ROOT)
    url = "http://127.0.0.1:8765"
    try:
        for _ in range(300):
            try:
                if httpx.get(f"{url}/health", timeout=1).status_code == 200:
                    break
            except Exception:
                time.sleep(1)
        zones = load_zones()["zone"].to_numpy()
        zones = zones[zones <= 263]
        out = {"setup": f"1 uvicorn worker, 2-core cloud VM, model={a.model}, store warmed by replay to {a.at}; "
                        "local load test, client and server on the same host",
               "runs": [asyncio.run(run(url, zones, r, a.n, a.at)) for r in a.rps]}
        (ROOT / "reports" / f"latency_{a.model}.json").write_text(json.dumps(out, indent=2))
        print(json.dumps(out, indent=2))
    finally:
        srv.terminate()


if __name__ == "__main__":
    main()
