from __future__ import annotations

from datetime import datetime
from pathlib import Path


def update_last_updated(text: str, stamp: str) -> str:
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if line.startswith("Last updated:"):
            lines[idx] = f"Last updated: {stamp}"
            break
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S Asia/Macau")
    status = Path("docs/current_status.md")
    text = update_last_updated(status.read_text(encoding="utf-8"), stamp)
    text += f"""

## Environment Thread Update - {stamp}

Completed multi-UAV VQA environment adapter without modifying V1.9 LUT/report or algorithm outputs.

New/updated environment-owned files:

```text
src/vqa_semcom/sim/multi_uav_env.py
scripts/run_multi_uav_env_smoke.py
tests/test_multi_uav_env.py
```

Implemented API and model coverage:

- Stable dependency-free Gym-like API: obs = env.reset(seed=None, options=None) and obs, reward, done, info = env.step(action).
- Multi-UAV state with position, altitude, battery, speed, camera state, current task, and cumulative flight distance.
- Dynamic task queue generated from VQA tasks with task type, risk level, view quality, freshness/cache age, deadline, quality threshold, priority, generation time, and synthetic disaster-area/grid coordinates.
- V1.9 SNR-LUT lookup uses LUT-available SNR bins first, then legacy channel fallback; sensed SNR is continuous and snr_bin is nearest measured LUT bin.
- Delay decomposition: fly, sensing, transmission, queue, inference, and model-load delay.
- Energy decomposition: flight, hover/sensing/inference, transmission, and compute energy.
- Observation contract includes task_type, risk_level, view_quality_bin, freshness_bin, sensed_snr_db, snr_bin, uav_state, edge_load, and cache_state.
- Step info contract includes answer_accuracy_est, delay_s, energy_j, payload_kb, quality_violation, deadline_violation, snr_bin, and service_level.

Verification completed:

```bash
cd /home/qiankun/phd_research/vqa_semcom
python3 -m py_compile src/vqa_semcom/sim/multi_uav_env.py scripts/run_multi_uav_env_smoke.py
python3 -m unittest tests/test_multi_uav_env.py
python3 scripts/run_multi_uav_env_smoke.py --config configs/v1_9_snr_lut.yaml --steps 6 --seed 17
```

Latest environment-only smoke output:

```text
outputs/env/env_smoke_20260618_170147/trace.csv
outputs/env/env_smoke_20260618_170147/summary.md
```

Active processes observed and left untouched:

```text
PID 1208659 V1.9 SNR detector/VLM eval
PID 1222872 legacy TCH-PPO seed0
PID 1222873 legacy TCH-PPO seed1
PID 1222874 legacy TCH-PPO seed2
```
"""
    status.write_text(text, encoding="utf-8")

    todo = Path("docs/experiment_todo.md")
    text = update_last_updated(todo.read_text(encoding="utf-8"), stamp)
    text += f"""

## Environment Thread Completed - {stamp}

Completed:

- Added src/vqa_semcom/sim/multi_uav_env.py for multi-UAV dynamic task-queue simulation.
- Added scripts/run_multi_uav_env_smoke.py; writes only under outputs/env/env_smoke_*.
- Added tests/test_multi_uav_env.py for reset observation contract, step info contract, UAV movement, and energy spending.

Recommended next environment checks:

```bash
cd /home/qiankun/phd_research/vqa_semcom
python3 -m unittest tests/test_multi_uav_env.py
python3 scripts/run_multi_uav_env_smoke.py --config configs/v1_9_snr_lut.yaml --steps 12 --seed 21
```

Expected artifacts:

```text
outputs/env/env_smoke_*/trace.csv
outputs/env/env_smoke_*/summary.md
```

Algorithm integration note:

- Import with: from vqa_semcom.sim.multi_uav_env import load_multi_uav_env.
- Use env.reset(seed=...) and env.step(action).
- Minimal action fields remain compatible with docs/interfaces.md: service_level, bandwidth, power, cpu_share, gpu_share, optional uav_assignment, optional waypoint.
- Environment info already exposes the reward components required for algorithm tables.

Open follow-ups:

1. Add a vectorized observation/action adapter if the algorithm thread requires fixed-shape tensors.
2. Calibrate mobility/energy constants against a chosen UAV platform before paper-grade experiments.
3. Once the running V1.9 SNR experiment finishes, rerun env smoke so nearest-bin lookup can use the refreshed full-bin LUT.
"""
    todo.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
