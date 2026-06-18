# InterUSS / UTM Mapping Summary

The UAV semantic communication network simulator now includes a lightweight UTM-style coordination layer inspired by InterUSS monitoring, DSS, USS qualifier scenario structure, and ASTM F3548 strategic coordination concepts.

Implemented mapping:

| Simulator artifact | UTM meaning |
|---|---|
| `operational_intent_id` | stable per-task operational intent reference |
| `operational_intent_state` | accepted / activated / nonconforming / contingent |
| `Area4D` | 4D operational volume: area, altitude interval, time window |
| `strategic_conflict_count` | buffered strategic conflict count against concurrent intents |
| `dss_available` | abstraction of USS/DSS availability |
| `dss_delay_s` | coordination delay added to total task delay |
| `subscription_notification_delay_s` | delayed conflict-update notification delay |
| `utm_constraint_violation` | UTM coordination violation separated from quality/deadline violations |

Realistic scenario smoke coverage:

- `test_utm_nominal_planning`
- `test_utm_off_nominal_planning`
- `test_utm_intent_conflict`
- `test_utm_dss_outage`
- `test_utm_notification_delay`

All outputs remain environment-owned artifacts under `outputs/env/`. The simulator still does not import or depend on InterUSS; the concepts are used as explicit modeling fields for benchmark realism.
