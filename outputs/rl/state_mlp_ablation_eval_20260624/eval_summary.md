# State Vector / MLP Evaluation Summary

Evaluation only: no PPO retraining was run in this stage. Each row loads a checkpoint from `outputs/rl/state_mlp_ablation_trainonly_20260624/` with `--load-ppo-model` and runs 50 rollout episodes with 12 tasks per episode on `cuda:0`.

## Checkpoint Mapping

The train-only ablation produced checkpoints for `disaster_hotspot`, `low_snr_blockage`, and `edge_overload`. Evaluation on those scenarios uses the matching scenario checkpoint. `nominal_patrol` and `utm_conflict` use the same candidate/seed `edge_overload` checkpoint as an unseen-scenario transfer test. Eval seeds 3 and 4 reuse checkpoint seeds 0 and 1 respectively because only train seeds 0,1,2 exist.

## Aggregate Results

| scenario | candidate | sem_success | task_success | acc | acc_lcb | gap | delay | deadline_vio | energy | payload_kb | cache | token | image | utm_conflict | reward |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nominal_patrol | A_state_v1_128x128 | 0.194 | 0.041 | 0.581 | 0.801 | 0.219 | 12.543 | 0.701 | 1820.374 | 7.437 | 0.108 | 0.850 | 0.042 | 0.000 | -4.110 |
| nominal_patrol | B_state_v2_128x128 | 0.151 | 0.028 | 0.583 | 0.808 | 0.220 | 15.072 | 0.832 | 2210.337 | 1.817 | 0.108 | 0.887 | 0.005 | 0.000 | -5.020 |
| disaster_hotspot | A_state_v1_128x128 | 0.192 | 0.093 | 0.539 | 0.791 | 0.308 | 1.538 | 0.292 | 156.812 | 1.025 | 0.126 | 0.874 | 0.000 | 0.000 | -1.102 |
| disaster_hotspot | B_state_v2_128x128 | 0.192 | 0.093 | 0.539 | 0.791 | 0.308 | 1.538 | 0.292 | 156.773 | 1.025 | 0.126 | 0.874 | 0.000 | 0.000 | -1.102 |
| low_snr_blockage | A_state_v1_128x128 | 0.861 | 0.096 | 0.831 | 0.878 | 0.022 | 36.929 | 0.876 | 154.698 | 1.536 | 0.076 | 0.903 | 0.021 | 0.000 | -8.112 |
| low_snr_blockage | B_state_v2_128x128 | 0.861 | 0.096 | 0.831 | 0.878 | 0.022 | 36.929 | 0.876 | 154.698 | 1.536 | 0.076 | 0.903 | 0.021 | 0.000 | -8.112 |
| edge_overload | A_state_v1_128x128 | 0.669 | 0.543 | 0.666 | 0.859 | 0.053 | 3.022 | 0.207 | 476.355 | 0.806 | 0.055 | 0.945 | 0.000 | 0.000 | 0.641 |
| edge_overload | B_state_v2_128x128 | 0.721 | 0.721 | 0.682 | 0.857 | 0.043 | 1.270 | 0.000 | 143.625 | 0.778 | 0.067 | 0.933 | 0.000 | 0.000 | 1.675 |
| utm_conflict | A_state_v1_128x128 | 0.001 | 0.000 | 0.503 | 0.724 | 0.317 | 11.475 | 0.597 | 1658.529 | 1.079 | 0.193 | 0.806 | 0.001 | 0.075 | -4.082 |
| utm_conflict | B_state_v2_128x128 | 0.000 | 0.000 | 0.511 | 0.732 | 0.309 | 12.909 | 0.701 | 1875.358 | 0.955 | 0.194 | 0.806 | 0.000 | 0.078 | -4.575 |
| OVERALL | A_state_v1_128x128 | 0.383 | 0.155 | 0.624 | 0.811 | 0.184 | 13.101 | 0.535 | 853.354 | 2.377 | 0.112 | 0.876 | 0.013 | 0.015 | -3.353 |
| OVERALL | B_state_v2_128x128 | 0.385 | 0.188 | 0.629 | 0.813 | 0.180 | 13.543 | 0.540 | 908.158 | 1.222 | 0.114 | 0.881 | 0.005 | 0.016 | -3.427 |

## A vs B Delta

Positive delta means B (`state_v2 + 128,128`) is larger than A (`state_v1 + 128,128`). Lower is better for semantic gap, delay, deadline violation, energy, payload, UTM conflict; higher is better for success, accuracy, and reward.

| scenario | delta_sem_success | delta_task_success | delta_gap | delta_delay | delta_deadline_vio | delta_payload | delta_reward |
|---|---:|---:|---:|---:|---:|---:|---:|
| nominal_patrol | -0.043 | -0.013 | 0.001 | 2.529 | 0.131 | -5.619 | -0.910 |
| disaster_hotspot | 0.000 | 0.000 | 0.000 | -0.000 | 0.000 | 0.000 | 0.000 |
| low_snr_blockage | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| edge_overload | 0.053 | 0.179 | -0.010 | -1.752 | -0.207 | -0.028 | 1.034 |
| utm_conflict | -0.001 | 0.000 | -0.009 | 1.434 | 0.104 | -0.124 | -0.493 |
| OVERALL | 0.002 | 0.033 | -0.003 | 0.442 | 0.006 | -1.154 | -0.074 |

## Why V2 Reduced Semantic Gap But Did Not Reliably Improve Success

Overall, V2 lowered the semantic quality gap from 0.184 to 0.180, but semantic success moved from 0.383 to 0.385 and task success from 0.155 to 0.188. The gap signal therefore helped conservatism/feasibility awareness more than end-to-end task completion.

Main causes observed from the rollout metrics:

- **Feature normalization mismatch:** V2 appends heterogeneous features: LCB/gap in [0,1], slack ratios that can be negative, feasibility masks, and mobility delay/energy estimates. The MLP receives them without learned feature normalization, so the extra features can change logits without consistently improving routing.
- **Scenario-dependent vector shape:** V2 mask/mobility metadata can differ by scenario. Inference now pads/truncates to the checkpoint dimension, but this confirms the V2 schema still needs a fixed canonical feature layout before it should be used as the default paper model.
- **Feasibility/slack is advisory, not an action mask:** V2 encodes per-service feasibility and slack, but the safety layer and projection still make the final service/resource correction. The policy can learn lower-gap preferences while projection dominates the actual action outcome.
- **Reward/projection can override policy learning:** deadline-aware evidence guard and payload-delay projection suppress risky image choices after the actor emits logits. This protects task feasibility, but it also means better semantic-gap prediction does not necessarily translate to higher semantic success.
- **Success is multi-constraint:** semantic success needs LCB >= epsilon, while task success also needs deadline, mobility arrival, UTM, and resource feasibility. Lower semantic gap alone does not fix high delay or deadline violations.

## Artifacts

- Per-seed results: `outputs/rl/state_mlp_ablation_eval_20260624/eval_all_seed_results.csv`
- Scenario summary: `outputs/rl/state_mlp_ablation_eval_20260624/eval_summary_by_scenario.csv`
- Per-run command and stdout/stderr are under each candidate/scenario/seed directory.
- Missing/failed run count: 0
