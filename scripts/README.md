# Danh m?c script

Ch?y c?c l?nh t? th? m?c g?c repo. Xem tham s? b?ng `python -m scripts.<nh?m>.<t?n> --help` ho?c `python scripts/<nh?m>/<t?n>.py --help`.

C?c t?n file ???c gi? nguy?n khi chuy?n t? `scripts/<t?n>.py` v?o nh?m b?n d??i. `--help` ch? d?nh cho script c? giao di?n argparse; c?c script v?/t?ng h?p c? ??nh c? th? ch?y ngay khi ???c g?i, n?n ??c file tr??c khi th?c thi.

## `training/` ? Hu?n luy?n t?ng m? h?nh ho?c giai ?o?n

- [`run_final_movement_only_qmix.py`](training/run_final_movement_only_qmix.py)
- [`run_relay_stress_r4_a5_qmix.py`](training/run_relay_stress_r4_a5_qmix.py)
- [`run_relay_stress_r4_qmix.py`](training/run_relay_stress_r4_qmix.py)
- [`run_thesis_curriculum_ablation.py`](training/run_thesis_curriculum_ablation.py)
- [`run_thesis_dense_learning_curve.py`](training/run_thesis_dense_learning_curve.py)
- [`run_thesis_no_curriculum_fulltask.py`](training/run_thesis_no_curriculum_fulltask.py)
- [`run_thesis_rl_curriculum.py`](training/run_thesis_rl_curriculum.py)
- [`run_v4_a0_kaggle_long.py`](training/run_v4_a0_kaggle_long.py)
- [`run_v4_a1_local_100k.py`](training/run_v4_a1_local_100k.py)
- [`run_v4_a1_r4_100k.py`](training/run_v4_a1_r4_100k.py)
- [`run_v4_a1_r4_idrand_scratch_100k.py`](training/run_v4_a1_r4_idrand_scratch_100k.py)
- [`run_v4_a1_relay_dropout_100k.py`](training/run_v4_a1_relay_dropout_100k.py)
- [`run_v4_a1_toggle6.py`](training/run_v4_a1_toggle6.py)
- [`run_v5_a1_relay_request.py`](training/run_v5_a1_relay_request.py)
- [`run_v5_curriculum.py`](training/run_v5_curriculum.py)
- [`run_v5_dynamic_tradeoff.py`](training/run_v5_dynamic_tradeoff.py)
- [`run_v5_r6_relay_request_50k.py`](training/run_v5_r6_relay_request_50k.py)
- [`run_v5_r6_relay_request_pilot.py`](training/run_v5_r6_relay_request_pilot.py)
- [`run_v6_r6_macro_relay_pilot.py`](training/run_v6_r6_macro_relay_pilot.py)
- [`run_v7_r6_macro_relay_stability_pilot.py`](training/run_v7_r6_macro_relay_stability_pilot.py)
- [`run_v8_r6_macro_relay_battery_curriculum.py`](training/run_v8_r6_macro_relay_battery_curriculum.py)
- [`run_v9_r6_battery_transition_curriculum.py`](training/run_v9_r6_battery_transition_curriculum.py)

## `evaluation/` ? ??nh gi? checkpoint, rollout v? baseline

- [`eval_relay_stress_r4_a5_lam03_qmix.py`](evaluation/eval_relay_stress_r4_a5_lam03_qmix.py)
- [`eval_relay_stress_r4_a5_lam1_qmix.py`](evaluation/eval_relay_stress_r4_a5_lam1_qmix.py)
- [`eval_relay_stress_r4_a5_qmix.py`](evaluation/eval_relay_stress_r4_a5_qmix.py)
- [`eval_relay_stress_r4_qmix.py`](evaluation/eval_relay_stress_r4_qmix.py)
- [`eval_reward_ablation_relay_kpi.py`](evaluation/eval_reward_ablation_relay_kpi.py)
- [`run_final_baseline_eval.py`](evaluation/run_final_baseline_eval.py)
- [`run_role_recovery_diagnostic.py`](evaluation/run_role_recovery_diagnostic.py)
- [`run_thesis_exhaustive_dynamic_baseline.py`](evaluation/run_thesis_exhaustive_dynamic_baseline.py)
- [`run_uav_continuous_eval.py`](evaluation/run_uav_continuous_eval.py)
- [`run_uav_operational_eval.py`](evaluation/run_uav_operational_eval.py)
- [`run_uav_role_mix_eval.py`](evaluation/run_uav_role_mix_eval.py)
- [`run_v4_final_evidence.py`](evaluation/run_v4_final_evidence.py)

## `launchers/` ? ?i?u ph?i nhi?u seed, chu?i th? nghi?m v? pilot launcher

- [`eval_v4_a0_heldout_multiseed.py`](launchers/eval_v4_a0_heldout_multiseed.py)
- [`run_final_movement_only_multiseed.py`](launchers/run_final_movement_only_multiseed.py)
- [`run_relay_stress_r4_a5_lam03_pilot.py`](launchers/run_relay_stress_r4_a5_lam03_pilot.py)
- [`run_relay_stress_r4_a5_lam1_pilot.py`](launchers/run_relay_stress_r4_a5_lam1_pilot.py)
- [`run_relay_stress_r4_a5_multiseed.py`](launchers/run_relay_stress_r4_a5_multiseed.py)
- [`run_relay_stress_r4_multiseed.py`](launchers/run_relay_stress_r4_multiseed.py)
- [`run_reward_ablation_multiseed.py`](launchers/run_reward_ablation_multiseed.py)
- [`run_role_recovery_5k_and_curriculum.py`](launchers/run_role_recovery_5k_and_curriculum.py)
- [`run_thesis_curriculum_ablation_multiseed.py`](launchers/run_thesis_curriculum_ablation_multiseed.py)
- [`run_thesis_no_curriculum_multiseed.py`](launchers/run_thesis_no_curriculum_multiseed.py)
- [`run_thesis_rl_curriculum_multiseed.py`](launchers/run_thesis_rl_curriculum_multiseed.py)
- [`run_v4_a0_final_multiseed.py`](launchers/run_v4_a0_final_multiseed.py)
- [`run_v4_a0_final_remaining.py`](launchers/run_v4_a0_final_remaining.py)
- [`run_v6_r6_macro_relay_multiseed.py`](launchers/run_v6_r6_macro_relay_multiseed.py)
- [`run_v9_multiseed_chain.py`](launchers/run_v9_multiseed_chain.py)

## `analysis/` ? Audit, ch?n ?o?n v? t?ng h?p

- [`aggregate_v9_multiseed.py`](analysis/aggregate_v9_multiseed.py)
- [`audit_r4_pareto_roles.py`](analysis/audit_r4_pareto_roles.py)
- [`audit_r4_relay_bottlenecks.py`](analysis/audit_r4_relay_bottlenecks.py)
- [`audit_relay_stress_feasibility.py`](analysis/audit_relay_stress_feasibility.py)
- [`audit_reward_ablation_baseline.py`](analysis/audit_reward_ablation_baseline.py)
- [`audit_v4_a1_r4_identity_randomization.py`](analysis/audit_v4_a1_r4_identity_randomization.py)
- [`audit_v5_dynamic_tradeoff_map.py`](analysis/audit_v5_dynamic_tradeoff_map.py)
- [`diagnose_v4_role_navigation_reward.py`](analysis/diagnose_v4_role_navigation_reward.py)
- [`summarize_reward_ablation.py`](analysis/summarize_reward_ablation.py)
- [`summarize_v4_final_stages.py`](analysis/summarize_v4_final_stages.py)

## `plotting/` ? V? bi?u ?? v? b?n ??

- [`plot_dense_learning_curve.py`](plotting/plot_dense_learning_curve.py)
- [`plot_role_recovery_5k_and_curriculum.py`](plotting/plot_role_recovery_5k_and_curriculum.py)
- [`plot_role_search_audit.py`](plotting/plot_role_search_audit.py)
- [`plot_six_method_eval.py`](plotting/plot_six_method_eval.py)
- [`plot_thesis_curriculum_ablation.py`](plotting/plot_thesis_curriculum_ablation.py)
- [`plot_thesis_curriculum_learning_curve.py`](plotting/plot_thesis_curriculum_learning_curve.py)
- [`plot_thesis_dense_learning_curve.py`](plotting/plot_thesis_dense_learning_curve.py)
- [`plot_uav_continuous_eval.py`](plotting/plot_uav_continuous_eval.py)
- [`plot_uav_operational_eval.py`](plotting/plot_uav_operational_eval.py)
- [`plot_uav_role_mix_eval.py`](plotting/plot_uav_role_mix_eval.py)
- [`render_uav_relay_map.py`](plotting/render_uav_relay_map.py)

## `validation/` ? Ki?m tra thi?t l?p

- [`verify_v4_a0_reference.py`](validation/verify_v4_a0_reference.py)
- [`verify_v4_a1_r4_idrand_setup.py`](validation/verify_v4_a1_r4_idrand_setup.py)
- [`verify_v4_a1_r4_setup.py`](validation/verify_v4_a1_r4_setup.py)
- [`verify_v4_a1_relay_dropout_setup.py`](validation/verify_v4_a1_relay_dropout_setup.py)
- [`verify_v4_a1_setup.py`](validation/verify_v4_a1_setup.py)

## BS5 H1200 và Return

| Nhóm | File mới |
|---|---|
| Huấn luyện | [`run_bs5_cr_h1200.py`](training/run_bs5_cr_h1200.py), [`collision`](training/run_bs5_cr_h1200_collision.py), [`C-R Return-v3`](training/run_bs5_cr_h1200_return.py), [`C-N Return-v3`](training/run_bs5_cn_h1200_return.py) |
| Audit và tổng hợp | `analysis/analyze_bs5_cr_h1200*.py`, `analysis/analyze_bs5_cn_h1200_return.py`, `analysis/analyze_bs5_crcn_return_v3.py`, `analysis/bs5_return*_audit.py`, `analysis/bs5_return_v4_results.py` |
| Chạy paired theo giao thức đã khóa | [`experiments/`](experiments/README.md): C-R/C-N Return-v3 và đánh giá Return-v4 |
| Kiểm tra trước khi chạy | [`verification/`](verification/README.md): Return-v3, C-R/C-N và Return-v4 |
| Notebook và bộ dựng notebook | `notebooks/bs5_cr_h1200*.ipynb`, `notebooks/bs5_crcn_return_v3_kaggle.ipynb`, `notebooks/bs5_return_v4_pair_kaggle.ipynb` và các file `build_bs5_*notebook.py` tương ứng |

Giữ nguyên vị trí các entry point và bộ dựng notebook từ gói nguồn để các lệnh trong notebook tiếp tục hoạt động.
Các checkpoint và UE tape của hai bộ chạy paired là tài sản Dataset riêng, không nằm trong ZIP mã nguồn.
