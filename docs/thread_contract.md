# Thread Contract

Last updated: 2026-06-18 Asia/Macau
Remote host: qiankun@172.27.57.160
Primary project root: /home/qiankun/phd_research/vqa_semcom
Legacy / algorithm reference root: /home/qiankun/HPPO-VQA/vqa_semcom_v0

## Collaboration Rule

All Codex threads coordinate through these files only:

- docs/thread_contract.md
- docs/interfaces.md
- docs/current_status.md
- docs/experiment_todo.md

Each thread must read docs/current_status.md and docs/interfaces.md before making changes. Each thread must update docs/current_status.md and docs/experiment_todo.md before ending work.

## Thread Ownership

### VQA / SNR-LUT / Results Controller Thread

Owns:

- configs/v1_9_snr_lut.yaml
- scripts/run_v1_detector_eval.py
- scripts/report_v1_9_snr.py
- scripts/run_v1_resource_sim.py when used only for LUT-policy reporting
- src/vqa_semcom/snr.py
- outputs/vlm/v1_9_snr_predictions.csv
- outputs/lut/v1_9_snr_semantic_quality_lut.csv
- outputs/reports/v1_9_snr_*
- outputs/sim/v1_9_snr_resource_*
- logs/v1_9_snr_main_500.log

Must not overwrite:

- algorithm thread experiment outputs under outputs/rl, outputs/hppo, runs, or algorithm-specific directories unless explicitly requested.
- environment thread simulator implementation files unless interface changes are agreed in docs/interfaces.md.

### Algorithm Thread

Owns:

- HPPO / PPO / resource allocation policy code.
- Training and evaluation scripts for resource allocation.
- Algorithm outputs under outputs/rl, outputs/hppo, outputs/resource_alloc, runs, or a clearly named new algorithm output directory.
- Optional integration adapters that consume the fixed environment and LUT interfaces.

Reference starting point:

- /home/qiankun/HPPO-VQA/vqa_semcom_v0/scripts/run_tch_ppo.py
- /home/qiankun/HPPO-VQA/vqa_semcom_v0/scripts/run_resource_alloc.py
- /home/qiankun/HPPO-VQA/vqa_semcom_v0/docs/resource_allocation_baselines.md

Must not overwrite:

- outputs/vlm/v1_9_snr_predictions.csv
- outputs/lut/v1_9_snr_semantic_quality_lut.csv
- outputs/reports/v1_9_snr_*
- running logs owned by VQA thread.

### Environment Thread

Owns:

- Multi-UAV environment dynamics.
- Task queue, mobility, delay model, energy model, edge load, cache state, and Gym-style reset/step API.
- Environment tests and environment-only smoke scripts.
- Output paths under outputs/env, outputs/sim/env_*, or another clearly named environment directory.

Must not overwrite:

- Algorithm policy checkpoints/results unless requested.
- VQA predictions, LUT, and reports.

## File Update Protocol

Before work:

1. Read docs/current_status.md.
2. Read docs/interfaces.md.
3. Check for active PIDs and avoid killing or overwriting active outputs.

During work:

1. Write new outputs into thread-owned paths.
2. Keep interface changes backward compatible unless docs/interfaces.md is updated first.

Before ending:

1. Update docs/current_status.md with completed work, active processes, blockers, and output paths.
2. Update docs/experiment_todo.md with next commands and expected artifacts.
3. If an interface changed, update docs/interfaces.md in the same turn.
