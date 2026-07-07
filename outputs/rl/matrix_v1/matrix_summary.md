# matrix_v1 experiment summary

> generated 2026-07-06T09:21:24 by scripts/report_matrix_v1.py
> spec: docs_spec/RL_Experiment_Standards_Survey.md (section 3 final matrix)

## Artifact index

| artifact | content |
|---|---|
| figures/fig1_convergence | reward / constraint cost + 0.08 budget / lambda_conflict, 5 learning arms, 3 seeds ±std |
| figures/fig2_multiscale | proposed convergence at 2/4/6 UAVs |
| figures/fig3_uav, fig4_load, fig5_snr | zero-shot sweeps, all 9 methods |
| figures/fig6_violation_pareto | dual-condition violation bars + utility-violation Pareto |
| figures/fig7_zeroshot | zero-shot vs retrained reference at 3 unseen points |
| tables/table1_main | dual-condition main comparison, mean±std |
| tables/table2_ablation | v3 structural ablation with ±std |
| tables/table3_sample_efficiency | ep-to-95%-plateau + wall clock |

## Headline numbers (peak condition)

- Proposed (constrained two-timescale): acc 0.574±0.000, conflict 0.296±0.000, deadline-vio 0.748±0.000, reward -5.863±0.000
- Semantic greedy: acc 0.386±0.000, conflict 0.654±0.000, deadline-vio 1.000±0.000, reward -6.382±0.000
- Always cache: acc 0.313±0.000, conflict 0.000±0.000, deadline-vio 0.000±0.000, reward -0.943±0.000
- Oracle (best feasible): acc 0.684±0.000, conflict 0.402±0.000, deadline-vio 0.642±0.000, reward -3.345±0.000
- Random: acc 0.458±0.009, conflict 0.428±0.026, deadline-vio 0.657±0.021, reward -4.317±0.144

## Convergence readings

- Proposed (constrained two-timescale): ep95 ≈ 97
- multiscale 4 UAVs (default): ep95 ≈ 97

## Criteria check

- [PASS] proposed plateaus before 1000 ep (ep95 < 900) (observed ep95≈97)
- [PASS] oracle accuracy >= proposed accuracy (upper bound sanity) (observed delta +0.1100)
- [PASS] proposed accuracy > random accuracy (observed delta +0.1154)
- [PASS] nominal proposed semantic success >= 0.92 (v3 criterion 6) (observed 0.9430)
- [FAIL] nominal proposed task success >= 0.30 (v3 criterion 6) (observed 0.2790)
