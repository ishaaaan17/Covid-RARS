# Repository Restructure Notes

This note records the non-destructive layout cleanup.

## Principle

The canonical active package is `src/covid_rars/`. Historical evidence is preserved in frozen or archive locations, while executable code uses only the canonical package name.

## Main Relocations

| Old root location | New location |
|---|---|
| `Final_BTP_Publication_Results_2026-06-12/` | `results/frozen/Final_BTP_Publication_Results_2026-06-12/` |
| `Publication_ExternalValidation_Artifacts_2026-06-12/` | `results/frozen/Publication_ExternalValidation_Artifacts_2026-06-12/` |
| `Corrected_Coswara_NoLeakage_Results/` | `results/frozen/Corrected_Coswara_NoLeakage_Results/` |
| `Corrected_Coswara_NoLeakage_Windows_2026-06-12/` | `results/frozen/Corrected_Coswara_NoLeakage_Windows_2026-06-12/` |
| `Phase3_Coswara_Results/` | `results/frozen/Phase3_Coswara_Results/` |
| `CNN_Cough_Results/` | `results/frozen/CNN_Cough_Results/` |
| `Representation_Results_OpenSMILE_BEATs_2026-06-12/` | `results/representations/Representation_Results_OpenSMILE_BEATs_2026-06-12/` |
| `Representation_Results_OpenSMILE_BEATs_PANNs_2026-06-12/` | `results/representations/Representation_Results_OpenSMILE_BEATs_PANNs_2026-06-12/` |
| Root `*.zip` and `*.tar.gz` evidence bundles | `artifacts/bundles/` |
| Root `*.patch` files | `archive/patches/` |
| `11-June-Update/` | `archive/updates/11-June-Update/` |
| Root Gemini review exports | `archive/review_materials/` |
| Dated root status and decision notes | `archive/historical_status_snapshots/` |
| Superseded root runbooks and handoff notes | `archive/historical_project_docs/` |
| `covid_audio_btp/manuscripts/` | `manuscripts/` |

## What Was Not Changed

- No research evidence was deleted.
- Active implementation code was moved to the root-level `src/covid_rars/` package.
- No experiment was rerun.
- No result metric was edited.
- No compressed evidence bundle was regenerated.
- Historical duplicate review exports were retained rather than deduplicated.

## Active Entry Points After Cleanup

- Root README: `README.md`
- Active package: `src/covid_rars/`
- Repository map: `docs/repository/REPOSITORY_MAP.md`
- Artifact review guide: `ARTIFACT.md`
- Frozen results: `results/`
- Manuscripts: `manuscripts/`

## Verification

- Every one of the 495 tracked paths removed from the legacy layout has a
  non-empty replacement in the canonical layout or an explicitly documented
  archive location.
- The active package, script, test, and notebook inventories remain complete:
  97 package modules, 83 Python scripts, 98 Python test files, and 12 notebooks.
- Package compilation, direct imports, and command-line help checks pass from
  the repository root.
- The full test suite passes after the move (935 passed, 1 skipped).
- A detached checkout of the staged repository was installed in a fresh Python
  3.12 environment from `requirements-dev.txt`. Package installation,
  `pip check`, and the complete portable test suite passed without access to
  local data, caches, Torch, or HST checkpoints. Tests requiring those optional
  HST prerequisites skip explicitly and are covered by the separate Ubuntu HST
  profile.

Legacy HST writer identifiers remain in the trusted-writer allowlist so that
checkpoints created before the package rename can still be verified. They are
compatibility data, not active imports.
