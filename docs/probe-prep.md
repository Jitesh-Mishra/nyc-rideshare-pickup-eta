# Probe prep (answers come only from what this run did)

Read this, then run `python -m eta.train` and `pytest -q` yourself before any interview.

## How the code works, in 5 lines
1. `data.py` loads Lyft (HV0005) TLC trips and builds the label: driver arrival = on_scene - request, from Mar 6. It drops negative waits and clips at 30.
2. `features.py` turns every trip into time-stamped events per zone (request, drop-off, driver arrival, open request). Each feature counts or averages events in `[t - w, t)`. Offline this uses numpy searchsorted per zone; online, `OnlineFeatureStore` keeps sorted buffers. Both call the same `derive()`.
3. `train.py` fits B0 (zone x hour-of-week quantiles), LightGBM quantile models (alpha 0.5/0.9, early stopping on val), and a PyTorch MLP with embeddings and pinball loss. It scores validation, then runs the test week once.
4. `serve.py` is the FastAPI app. `/events` feeds the store and `/predict` reads features, runs the model, and returns P50/P90. If the store is more than 5 min behind, or anything throws, it returns the B0 baseline with `fallback: true`.
5. `parity.py` replays the test week through the online store in event-time order and checks the offline features match. `switchback.py` randomises borough x 30-min blocks. `monitor.py` computes daily PSI and P90 coverage.

## Probes

**Q: Your model beats the baseline by only 5.6%. Why should I care?**
- 15s: The ceiling is low. Even an oracle that knows each zone's real median wait in the same 10 minutes only beats the baseline by about 15%. I capture 37% of that, and the bigger win is calibration: P90 coverage went from 86% to 89%.
- 60s: Pickup wait is dominated by how far the matched driver is. Public TLC data has no driver locations. I quantified the ceiling with a future-label oracle (`reports/headroom.json`): 1.648 MAE vs B0 1.945. The quantile models fix the baseline's P90 under-coverage, and on the Mar 30 shock day the gain jumps to 10.2% with P90 coverage at 84.7% vs 74.1% for the baseline. That is the case real-time features exist for.
- Evidence: `reports/results.csv`, `reports/headroom.json`, `reports/daily_errors.csv`, docs/decisions.md D7.

**Q: How do you know there's no leakage?**
- 15s: Every feature uses events strictly before the request. There's a unit test for the boundary and a test that scrambles everything after a cutoff and checks the earlier features don't move.
- 60s: Events are stamped with when they become known: arrivals at on_scene time, drop-offs at drop-off time. `trip_miles` and fares are excluded. The split is chronological, tuning used validation only, and the test week ran once (`test_runs.json`).
- Evidence: `tests/test_features.py`, `reports/test_runs.json`.

**Q: How do you guarantee training/serving parity?**
- 15s: One features module, two backends, and a test. Replaying 10,000 sampled test-week requests through the online store gives identical values on all 19 features.
- 60s: Both paths compute raw window aggregates and pass them through the same `derive()`. The replay streamer orders events by time and answers queries before applying same-timestamp events, which matches the offline `[t-w, t)` rule. The open-queue state is seeded with the requests still waiting at warm-up start.
- Evidence: `reports/parity.json`, `src/eta/parity.py`.

**Q: Is the open-queue feature leaking the future?**
- At time t you know which requests in the zone are still waiting; that's dispatch state. The caveat is that the data is completed trips only, so the queue misses requests that later cancel. In production the queue would include them, so it would be a bigger and probably more useful signal.

**Q: What does the latency number mean?**
- Local load test: one uvicorn worker on a 2-core cloud VM, open-loop at 50/100/200 RPS, 6,000 requests each, with the client on the same host. At 200 RPS LightGBM had round-trip p99 7.2 ms and server compute p99 2.3 ms, with 0 errors and 0 fallbacks. It is not a production number: no network hop and a single process.
- Evidence: `reports/latency_lgb.json`, `scripts/latency_bench.py`.

**Q: Why switchbacks? What did you actually show?**
- 15s: In a two-sided market, rider-level randomisation leaks through shared supply, so you randomise time-space blocks. I showed errors correlate within borough x 30-min blocks, so the naive SE is 7.4x too small. Naive tests flagged 81% of A/A replays as significant; cluster-robust tests flagged 4.6%.
- 60s: I can't show interference here, because predictions don't move drivers in a replay; that needs a dispatch simulator. The A/B replay (B0 vs LightGBM) gives -0.130 min with a cluster CI of [-0.183, -0.078]. That is still significant, but the CI is 8x wider than the naive one.
- Evidence: `reports/switchback.json`, docs/decisions.md D8.

**Q: Why does PyTorch not beat LightGBM?**
- Tabular data, 19 features, and most signal in zone x time, which trees handle natively. The embedding MLP ties on P50 (1.835) and is slightly better on P90 pinball (0.531 vs 0.533). I'd expect deep models to win with richer inputs like routes or driver positions, which this data lacks.

**Q: What breaks it?**
- Shock days: Mar 30 waits ran about 40% high all day, and P90 coverage fell to 84.7%. The drift monitor flagged it (PSI 1.54 on recent wait). Airports are hardest (MAE 3.40 vs 1.77 elsewhere). Evenings and Staten Island have the lowest coverage. The fix I'd add next is conformal recalibration per borough.
- Evidence: `reports/sliced_errors.csv`, `reports/monitoring.json`.

**Q: Why did validation improve but test MAE is higher overall?**
- The test week is harder (mean wait rises from 3.85 to 5.72 min across the week; the weekend shock). B0 also rose from 1.842 to 1.943. The relative gain held: -4.1% on val vs -5.6% on test.
