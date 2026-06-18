# InterUSS / UTM Realistic Flight Mapping

This note maps InterUSS and ASTM-style UTM concepts into the UAV-VQA semantic communication network simulator. The simulator does not depend on InterUSS code; it borrows the operational-intent and strategic-conflict abstractions needed for realistic UAV service planning.

## InterUSS Concepts

- DSS: Discovery and Synchronization Service. InterUSS DSS is an ASTM-aligned mechanism for USS-to-USS discovery/synchronization so USSs can coordinate operations while the DSS avoids storing full operational details.
- USS qualifier: InterUSS monitoring provides automated test scenarios and suites for USS interoperability/conformance checks.
- ASTM F3548 UTM: focuses on strategic coordination, strategic conflict detection, operational intent conformance, constraint awareness, and situational awareness for nonconforming or contingent operations in connected UTM environments.
- Scenario families: `nominal_planning`, `off_nominal_planning`, `flight_intent_validation`, and `subscription_notifications` represent realistic planning, intent validation, and notification-delay behavior.

Useful references:

- InterUSS monitoring: https://github.com/interuss/monitoring
- InterUSS DSS: https://github.com/interuss/dss
- InterUSS USS qualifier: https://github.com/interuss/monitoring/tree/main/monitoring/uss_qualifier
- InterUSS ASTM UTM scenarios: https://github.com/interuss/monitoring/tree/main/monitoring/uss_qualifier/scenarios/astm/utm
- ASTM F3548-21 overview: https://store.astm.org/f3548-21.html

## Mapping Into `multi_uav_env.py`

| InterUSS / UTM concept | Simulator field or behavior |
|---|---|
| Operational intent reference | `EnvTask.operational_intent_id` |
| Operational intent state | `EnvTask.operational_intent_state`, `info["operational_intent_state"]` |
| Accepted operation | UTM-enabled task before UAV action or cache-only service |
| Activated operation | UAV observe/revisit/waypoint action with available DSS and no strategic conflict |
| Nonconforming operation | off-nominal planning stress or buffered strategic conflict |
| Contingent operation | DSS outage or unavailable coordination service |
| 4D operational volume | `Area4D(center, radius, altitude_min/max, start_step, end_step)` |
| Operational priority | `EnvTask.operational_priority`, derived from task risk/priority |
| Strategic conflict detection | `_strategic_conflict_task_ids()` and `_area4d_overlaps_with_buffer()` |
| Temporal/spatial buffer | `multi_uav_env.utm.spatial_buffer_m`, `altitude_buffer_m`, `temporal_buffer_steps` |
| DSS availability | `multi_uav_env.utm.dss_available`, `info["dss_available"]` |
| DSS communication delay | `info["dss_delay_s"]`, `info["utm_dss_delay_s"]`, included in total delay |
| Subscription notification delay | `info["subscription_notification_delay_s"]`, `info["utm_notification_delay_s"]`, included when conflict updates are delayed |
| Constraint/intent violation | `info["utm_constraint_violation"]`, separate from quality/deadline violation |

Cache-only service (`service_level=0`, `sensing_decision=reuse_cache`) does not create a UAV operational intent and therefore cannot create an airspace conflict. UAV observe/revisit/waypoint actions with `service_level > 0` do create operational intent and are checked against concurrent intents.

## Realistic Scenario Presets

- `test_utm_nominal_planning`: available DSS, accepted-to-activated operational intent flow, low DSS delay.
- `test_utm_off_nominal_planning`: mobility/battery stress with critical tasks marked nonconforming.
- `test_utm_intent_conflict`: overlapping 4D volumes with spatial, altitude, and temporal buffers.
- `test_utm_dss_outage`: unavailable DSS, contingent operational intents, additional coordination delay.
- `test_utm_notification_delay`: delayed subscription/notification updates for conflict-heavy operations.

These scenarios are accepted by `env.reset(options={"formal_scenario": ...})`. They are intentionally not added to the default training scenario list returned by `available_formal_scenarios()` so existing benchmark defaults remain stable; use `available_utm_realistic_scenarios()` or pass scenario names explicitly for UTM smoke tests.

## Why This Matters For UAV-VQA Semantic Communication

The resource allocation problem is not only choosing semantic evidence and bandwidth. In realistic UAV service, the controller must also decide whether a UAV operation can be strategically coordinated, whether delayed DSS/notification updates add service delay, and whether an emergency VQA task becomes nonconforming or contingent. The new fields let the benchmark report:

- task semantic quality: `answer_accuracy_est`, `quality_violation`;
- network resources: `rate_mbps`, `bandwidth_hz`, `power_w`, CPU/GPU shares;
- UAV service cost: `fly_delay_s`, `fly_energy_j`, battery remaining;
- UTM coordination cost: `operational_intent_state`, `strategic_conflict_count`, `dss_delay_s`, `subscription_notification_delay_s`, `utm_constraint_violation`.
