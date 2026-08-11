# Manuscript Support Analyses

## Top Demographic/Protocol Linear Attribution Drivers

| audit_model | split | feature | feature_group | coefficient | mean_shap | mean_abs_shap | max_abs_shap | positive_label_share | direction_by_mean_shap | rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| demographic_protocol_only | test | recording_year | recording_protocol | 2.64788 | -0.0246989 | 2.28178 | 5.34166 | 0.399371 | negative_label | 1 |
| demographic_protocol_only | test | recording_month | recording_protocol | 1.39966 | 0.110992 | 1.10445 | 3.91527 | 0.374214 | positive_label | 2 |
| demographic_protocol_only | test | country_India | demographic | 0.634702 | 0.0166997 | 0.348218 | 2.02736 | 0.918239 | positive_label | 3 |
| demographic_protocol_only | test | age | demographic | 0.247724 | 0.00376736 | 0.198142 | 1.14668 | 0.377358 | positive_label | 4 |
| demographic_protocol_only | test | duration_sec | recording_protocol | -0.253804 | 0.00487971 | 0.179797 | 0.763697 | 0.581412 | positive_label | 5 |

## Worst Residual IPW Balance Rows

| feature | before_smd | after_smd | before_abs_smd | after_abs_smd | weight_config | control_method | weight_cap | clip_quantile | smd_reduction | balance_severity | rank_after_weighting |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| recording_year | 1.38384 | 0.723964 | 1.38384 | 0.723964 | ipw_cap_2_q_0.95 | ipw_label_propensity | 2 | 0.95 | 0.659872 | severe_residual_imbalance | 1 |
| country_India | 0.524531 | 0.438078 | 0.524531 | 0.438078 | ipw_cap_2_q_0.95 | ipw_label_propensity | 2 | 0.95 | 0.0864529 | moderate_residual_imbalance | 2 |
| country_United States | -0.295599 | -0.250388 | 0.295599 | 0.250388 | ipw_cap_2_q_0.95 | ipw_label_propensity | 2 | 0.95 | 0.0452108 | moderate_residual_imbalance | 3 |
| gender_female | 0.420921 | 0.236072 | 0.420921 | 0.236072 | ipw_cap_2_q_0.95 | ipw_label_propensity | 2 | 0.95 | 0.184849 | minor_residual_imbalance | 4 |
| recording_month | 0.244037 | 0.235485 | 0.244037 | 0.235485 | ipw_cap_2_q_0.95 | ipw_label_propensity | 2 | 0.95 | 0.00855184 | minor_residual_imbalance | 5 |

## External AUPRC Lift Over Prevalence

| representation | model_name | feature_strategy | selection_metric | auroc | auprc | target_prevalence | absolute_auprc_lift | relative_auprc_lift | pr_lift_interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mfcc | logistic_regression | top_stable_50 | auroc | 0.534794 | 0.042176 | 0.0342096 | 0.007966 | 1.23287 | limited_lift |
| beats | logistic_regression | drop_high_shift | auroc | 0.552993 | 0.0393622 | 0.0342096 | 0.005153 | 1.15062 | limited_lift |
| opensmile_egemaps | logistic_regression | drop_high_shift | auroc | 0.551946 | 0.039114 | 0.0342096 | 0.004904 | 1.14336 | limited_lift |
| panns | logistic_regression | drop_high_shift | auroc | 0.502439 | 0.0348934 | 0.0342096 | 0.000684 | 1.01999 | near_prevalence |

## Unknown Label Summary

| label_availability | n_rows | n_participants | age_mean | age_median | duration_sec_mean | recording_year_min | recording_year_max | top_country | top_country_share | top_quality_flag | top_quality_flag_share | top_gender | top_gender_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| known | 19024 | 2114 | 35.242 | 31 | 10.0594 | 2020 | 2022 | India | 0.914371 | ok | 0.951745 | male | 0.700536 |
| unknown | 5688 | 632 | 34.8671 | 30 | 9.16544 | 2020 | 2022 | India | 0.920886 | ok | 0.955169 | male | 0.662975 |
