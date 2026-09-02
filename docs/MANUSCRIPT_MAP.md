# Manuscript-to-code map

| Manuscript item | Workflow |
|---|---|
| Table 1: main city-size comparison | `scripts/run_main_evaluation.sh`, then `scripts/run_tables.sh table1` |
| Table 2: leave-one-region-out comparison | `scripts/run_leave_one_region_out.sh`, then `scripts/run_tables.sh table2` |
| Table 3: module- and component-level ablation | `scripts/run_module_ablations.sh`, `scripts/run_component_ablations.sh`, then `scripts/run_tables.sh table3` |
| Explanation distillation | `scripts/run_explanation_distillation.sh` |
| Case-study predictions and LIME | `scripts/run_case_study_outputs.sh` |
| Bootstrap metric uncertainty | `scripts/run_city_bootstrap.sh` |
| Independent prediction intervals | `scripts/run_independent_calibration.sh` |
| PPO reward sensitivity | `scripts/run_ppo_penalty_sensitivity.sh` |
| LoRA similarity threshold | `scripts/run_lora_similarity_threshold.sh` |

Fig. 3(c--e), Fig. 5(b,d), and Fig. 6(b--d) were manually composed using
HTML. Fig. 2(d--f) and Fig. 4(a) were manually prepared in QGIS.

The literature-frequency values in Fig. 2(h) were collected manually through
factor-specific Web of Science searches.
