After first run completes, do not start COUGHVID/CNN immediately. First verify the clean Coswara baseline.

  Run these from covid_audio_btp/:

  python scripts/12_validate_artifacts.py --strict
  python -m pytest

  Then check the key new artifacts:

  Test-Path data\outputs\metrics\calibrated_branch_predictions_validation.csv
  Test-Path data\outputs\metrics\fusion_thresholds.csv
  Test-Path data\outputs\metrics\fusion_metrics.csv
  Test-Path data\outputs\metrics\subgroup_metrics.csv

  Print the main metrics:

  python -c "import pandas as pd; print(pd.read_csv('data/outputs/metrics/fusion_metrics.csv').to_string(index=False)); print(pd.read_csv('data/outputs/metrics/
  fusion_thresholds.csv').to_string(index=False))"

  If calibrated_branch_predictions_validation.csv or fusion_thresholds.csv is missing, rerun only these stages:

  python scripts/08_calibrate_branches.py --validation-predictions data/outputs/metrics/ml_predictions_validation.csv --test-predictions data/outputs/metrics/ml_predictions_test.csv --method platt
  python scripts/09_run_fusion.py --predictions data/outputs/metrics/calibrated_branch_predictions.csv --validation-metrics data/outputs/metrics/ml_baseline_metrics.csv
  python scripts/10_shift_and_confounding_checks.py --predictions data/outputs/metrics/fusion_predictions.csv --metadata data/processed/metadata_clean.csv
  python scripts/11_make_report_assets.py --metadata data/processed/metadata_clean.csv --predictions data/outputs/metrics/fusion_predictions.csv
  python scripts/12_validate_artifacts.py --strict

  Then create a result folder for review:

  cd ..
  New-Item -ItemType Directory -Force First_Clean_Run_Results
  Copy-Item covid_audio_btp\data\outputs\metrics\*.csv First_Clean_Run_Results\
  Copy-Item covid_audio_btp\reports\tables\*.csv First_Clean_Run_Results\ -ErrorAction SilentlyContinue
  Copy-Item covid_audio_btp\reports\report_outline.md First_Clean_Run_Results\ -ErrorAction SilentlyContinue
  Copy-Item covid_audio_btp\reports\slides_outline.md First_Clean_Run_Results\ -ErrorAction SilentlyContinue
  Compress-Archive -Force First_Clean_Run_Results First_Clean_Run_Results.zip

  Send me that zip or the metric outputs.

  If validation passes and metrics look sane, the next run is publication extras without COUGHVID: metadata baseline, quality-weighted fusion, abstention, bootstrap CI, paired comparison, confounding
  matching, paper tables, manifest. COUGHVID comes only after that is clean.