# Semantic Path Cache/Defer Fix2 Comparison

## Setup

- branch: `codex/semantic-path-cache-defer`
- environment baseline commit present: `6c5a064 fix(env): calibrate semantic path feasibility scenarios`
- fix2 output: `outputs/rl/semantic_path_cache_defer_fix2_20260624/`
- scenarios: `normal_patrol`, `disaster_hotspot`, `low_snr_soft`, `low_snr_blockage`, `edge_overload`, `utm_conflict`
- seeds: `0,1,2`; train episodes: `300`; eval episodes: `50`; tasks per episode: `12`; device: `cuda:0`

## Proposed PPO Metrics

| scenario | run | semantic success | task success | deadline vio | UTM vio | payload KB | cache/token/image/defer/update | joint feasible sel | deadline infeasible sel | UTM infeasible sel |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| normal_patrol | B_state_v2_fixed | 0.156 | 0.156 | 0.000 | 0.000 | 30.742 | 0.229/0.000/0.093/0.547/0.359 | n/a | n/a | n/a |
| normal_patrol | short | 0.378 | 0.230 | 0.251 | 0.000 | 0.751 | 0.257/0.688/0.000/0.004/0.050 | n/a | n/a | n/a |
| normal_patrol | fix1 | 0.275 | 0.275 | 0.000 | 0.000 | 0.548 | 0.434/0.566/0.000/0.000/0.000 | n/a | n/a | n/a |
| normal_patrol | fix2 | 0.251 | 0.098 | 0.212 | 0.000 | 0.376 | 0.184/0.356/0.000/0.454/0.005 | 0.028 | 0.800 | 0.000 |
| disaster_hotspot | B_state_v2_fixed | 0.280 | 0.064 | 0.733 | 0.000 | 0.400 | 0.600/0.400/0.000/0.000/0.000 | n/a | n/a | n/a |
| disaster_hotspot | short | 0.316 | 0.278 | 0.144 | 0.000 | 0.641 | 0.451/0.504/0.000/0.000/0.044 | n/a | n/a | n/a |
| disaster_hotspot | fix1 | 0.280 | 0.196 | 0.336 | 0.000 | 29.512 | 0.418/0.431/0.151/0.000/0.000 | n/a | n/a | n/a |
| disaster_hotspot | fix2 | 0.278 | 0.100 | 0.709 | 0.000 | 0.465 | 0.602/0.398/0.000/0.000/0.000 | 0.060 | 0.398 | 0.000 |
| low_snr_soft | B_state_v2_fixed | n/a | n/a | n/a | n/a | n/a | n/a/n/a/n/a/n/a/n/a | n/a | n/a | n/a |
| low_snr_soft | short | 0.931 | 0.271 | 0.699 | 0.000 | 0.672 | 0.305/0.450/0.000/0.010/0.235 | n/a | n/a | n/a |
| low_snr_soft | fix1 | 0.822 | 0.276 | 0.608 | 0.000 | 0.564 | 0.427/0.573/0.000/0.000/0.000 | n/a | n/a | n/a |
| low_snr_soft | fix2 | 0.287 | 0.217 | 0.092 | 0.000 | 0.255 | 0.175/0.290/0.000/0.535/0.000 | 0.276 | 0.724 | 0.000 |
| low_snr_blockage | B_state_v2_fixed | 0.948 | 0.128 | 0.854 | 0.000 | 0.918 | 0.300/0.600/0.100/0.000/0.000 | n/a | n/a | n/a |
| low_snr_blockage | short | 0.931 | 0.271 | 0.699 | 0.000 | 0.672 | 0.305/0.450/0.000/0.010/0.235 | n/a | n/a | n/a |
| low_snr_blockage | fix1 | 0.822 | 0.276 | 0.608 | 0.000 | 0.564 | 0.427/0.573/0.000/0.000/0.000 | n/a | n/a | n/a |
| low_snr_blockage | fix2 | 0.825 | 0.276 | 0.609 | 0.000 | 0.563 | 0.429/0.571/0.000/0.000/0.000 | 0.047 | 0.589 | 0.000 |
| edge_overload | B_state_v2_fixed | 0.697 | 0.649 | 0.046 | 0.000 | 0.823 | 0.611/0.278/0.000/0.000/0.111 | n/a | n/a | n/a |
| edge_overload | short | 0.609 | 0.231 | 0.569 | 0.000 | 0.864 | 0.034/0.585/0.000/0.000/0.381 | n/a | n/a | n/a |
| edge_overload | fix1 | 0.384 | 0.104 | 0.557 | 0.000 | 0.697 | 0.265/0.712/0.000/0.001/0.022 | n/a | n/a | n/a |
| edge_overload | fix2 | 0.499 | 0.081 | 0.657 | 0.000 | 0.715 | 0.208/0.773/0.000/0.000/0.019 | 0.113 | 0.711 | 0.000 |
| utm_conflict | B_state_v2_fixed | 0.000 | 0.000 | 0.000 | 0.000 | 0.997 | 1.000/0.000/0.000/0.000/0.000 | n/a | n/a | n/a |
| utm_conflict | short | 0.000 | 0.000 | 0.541 | 0.091 | 0.592 | 0.501/0.499/0.000/0.000/0.000 | n/a | n/a | n/a |
| utm_conflict | fix1 | 0.000 | 0.000 | 0.679 | 0.179 | 60.770 | 0.373/0.325/0.301/0.000/0.000 | n/a | n/a | n/a |
| utm_conflict | fix2 | 0.000 | 0.000 | 0.312 | 0.000 | 0.426 | 0.004/0.359/0.000/0.637/0.000 | 0.004 | 0.996 | 0.000 |

## Fix2 Checks

- FAIL: normal_patrol deadline near zero (deadline=0.212).
- PASS: low_snr_soft differs from low_snr_blockage (soft sem/deadline=0.287/0.092, blockage=0.825/0.609).
- PASS: edge_overload cache_update restrained (cache_update=0.019).
- FAIL: edge_overload improves over fix1 deadline (fix2=0.657, fix1=0.557).
- PASS: utm_conflict UTM violation reduced (fix2=0.000, fix1=0.179).
- PASS: no always-cache/token collapse in fix2 proposed (checked cache/token ratios).

## Interpretation

- Fix2 successfully consumes the new `low_snr_soft` preset: soft and blockage are no longer identical.
- The stricter cache-update gate works: edge_overload cache_update is near zero, and image remains disabled for proposed PPO.
- UTM control is materially safer: proposed PPO selects no UTM-infeasible paths in fix2 aggregate and UTM conflict violation is 0.000, but semantic/task success remains 0 because the feasible safe actions are mostly defer/token with large semantic gap.
- Edge-overload is still not paper-ready: deadline violation remains high and task success is far below the old B_state_v2_fixed baseline, even though cache_update overuse is gone. This points to resource/projection feasibility and calibrated scenario difficulty rather than cache_update overuse alone.
- Next step should be a targeted edge/UTM policy: use candidate `joint_feasible` as a supervised target, add explicit feasible-token preference when token is deadline-feasible, and separate defer from failure in reward so UTM-safe deferral does not erase learning signal.
