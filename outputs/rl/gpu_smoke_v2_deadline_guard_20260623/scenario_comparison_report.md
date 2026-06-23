# Scenario-Aware Semantic Control Benchmark

- scenarios: `edge_overload`
- smoke: `False`
- seeds: `0`
- episodes per scenario/policy: `1`
- train episodes per PPO variant: `2`
- tasks per episode: `4`

| scenario | policy | semantic success | accuracy LCB | accuracy mean | uncertainty | epsilon | failed eps mean | quality gap | Q_quality | Q_deadline | Q_energy | Q_utm | delay | energy | flight energy | arrival delay | coverage | payload KB | deadline vio | UTM conflict | cache | token | image |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| edge_overload | always_cache | 0.000 | 0.100 | 0.208 | 0.699 | 0.640 | 0.640 | 0.540 | 3.610 | 0.000 | 0.000 | 0.000 | 0.372 | 85.850 | 78.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| edge_overload | always_image | 0.000 | 0.510 | 1.000 | 0.795 | 0.640 | 0.640 | 0.130 | 0.844 | 1.540 | 1516.830 | 0.000 | 5.210 | 458.853 | 249.799 | 1.239 | 0.055 | 161.894 | 0.167 | 0.000 | 0.000 | 0.000 | 1.000 |
| edge_overload | always_semantic_token | 0.750 | 0.690 | 0.913 | 0.391 | 0.625 | 0.640 | 0.032 | 0.130 | 0.000 | 1227.034 | 0.000 | 6.080 | 985.080 | 918.360 | 4.555 | 0.330 | 0.873 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| edge_overload | lyapunov_greedy | 0.000 | 0.100 | 0.208 | 0.699 | 0.640 | 0.640 | 0.540 | 3.610 | 0.000 | 0.000 | 0.000 | 0.372 | 85.850 | 78.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| edge_overload | proposed_v2_deadline_guard | 0.750 | 0.690 | 0.913 | 0.391 | 0.625 | 0.640 | 0.032 | 0.130 | 0.000 | 0.000 | 0.000 | 1.525 | 66.721 | 0.000 | 0.000 | 0.000 | 0.873 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| edge_overload | semantic_greedy | 0.000 | 0.510 | 1.000 | 0.795 | 0.640 | 0.640 | 0.130 | 0.844 | 1.540 | 1516.830 | 0.000 | 5.210 | 458.853 | 249.799 | 1.239 | 0.055 | 161.894 | 0.167 | 0.000 | 0.000 | 0.000 | 1.000 |

## Notes

- `semantic_quality_gap = max(0, epsilon_k - semantic_accuracy_lcb)`.
- Proposed PPO rows use Semantic-LCB Lyapunov reward/projection, oracle warm-start, service-level curriculum, and cache shortfall penalties.
- Scenario subdirectories contain per-scenario summaries and traces; large rollout/model artifacts remain ignored.
