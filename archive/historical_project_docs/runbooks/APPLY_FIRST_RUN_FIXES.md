# First Run Fixes - 2026-06-11

This bundle contains the verified patch for the first-run issues:

- Ignore macOS AppleDouble sidecar audio files (`._*`) and `__MACOSX` archive folders.
- Fix participant-level fusion subgroup/confounding merge so fused rows are not duplicated.
- Save calibrated validation predictions and use validation-tuned fusion thresholds.

## Apply from your repo root

Run these from the root folder that contains `covid_audio_btp/`:

```bash
git status --short
git apply --check first_run_fixes_2026-06-11.patch
git apply first_run_fixes_2026-06-11.patch
```

Then verify locally:

```bash
cd covid_audio_btp
python -m pytest
```

If `git apply --check` reports conflicts, stop and show Codex the exact output. Do not manually rewrite the patch by guessing.

## After applying

For a clean rerun, delete generated artifacts, not datasets:

```bash
cd covid_audio_btp
rm -rf data/interim data/processed data/outputs reports
```

Then run your notebook again from the start. Your manually deleted `data/raw/coswara/._*` files were not real audio and do not need to be restored.
