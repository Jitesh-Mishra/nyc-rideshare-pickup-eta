# Decisions

**D1 Label: on_scene - request, from Mar 6 (2026-10-05)**
Context: for HV0005, `on_scene_datetime` is 99.7% null before 2025-03-06 and 0% null after.
Options: pickup - request on all days (includes boarding, median 0.75 min) vs on_scene - request from Mar 6.
Decision: driver arrival (on_scene - request). That is the ETA a dispatcher quotes, and it gives 16 clean training days.
Tradeoff: we lose Mar 1 to 5 as labels. They still feed the feature windows.

**D2 Drop negative waits, clip at 30 min**
1.7% of rows have on_scene before request, with a weekday commute pattern (likely prearranged trips). They are dropped from labels and from the open-queue/arrival streams, but kept as demand and drop-off events. Waits above 30 min (0.04%) are clipped.

**D3 Split: train Mar 6 to 21, val Mar 22 to 24, test Mar 25 to 31**
The split is chronological. Validation is used for LightGBM early stopping, torch epoch selection and the feature iteration in D5. The test week ran once (`reports/test_runs.json`).

**D4 Rolling mean, not median, of recent waits**
The brief said a rolling 30-min median. A mean over clipped waits is exactly computable both offline (cumsum + searchsorted) and online (buffer sum), so parity is exact. A median would need an order-statistic structure online. Clipping at 30 limits outlier pull.

**D5 Added the open-request queue and a 10-min wait window after the first validation pass**
The first validation pass showed LightGBM only 3.7% better than B0. The open queue (requests still waiting at t, with their mean age) is something a dispatcher always knows at request time. Validation MAE moved from 1.774 to 1.767, so the gain was small. An oracle check then showed the ceiling is low (see D7), so I stopped feature work there rather than overfit validation.

**D6 Serve LightGBM, keep PyTorch as a benchmark**
They tie on P50 MAE (1.835), and PyTorch is 0.4% better on P90 pinball. LightGBM's p99 latency is 7.2 ms vs 12.9 ms at 200 RPS.

**D7 Report headroom with a future-label oracle**
A zone x 10-min block median of the realised waits is not a model, because it uses future labels. It bounds what any zone-time feature could achieve: 1.648 MAE vs B0 1.945 on the same rows. That makes a 5.6% gain interpretable (37% of the attainable gap) instead of just looking small.

**D8 Switchback: A/A replays + one A/B, no interference claim**
Without a dispatch simulator, predictions don't change outcomes, so spillover can't be measured. What can be measured honestly is within-block error correlation, and therefore how wrong request-level SEs are.

**D9 LightGBM trains on a 2M-row sample of 2.94M**
The sample keeps each fit to about 1 min per quantile on a 2-core VM. Torch uses the full training set.
