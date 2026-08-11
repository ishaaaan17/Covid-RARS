# Temporal Robustness Causal Chain

This derived summary is for results communication and manuscript planning.

## Causal Chain

1. Participant split appears strong: AUROC 0.873.
2. Calendar-balanced split is lower: AUROC 0.787.
3. Strict early-to-late temporal holdout collapses: AUROC 0.566.
4. External transfer is similarly weak: AUROC 0.553.
5. Temporal-minus-participant AUROC difference is -0.308 with two-sided bootstrap p=0.0000.
6. Year/month attribution and ablation tables identify temporal/protocol variables as structural label predictors.

## Key Temporal Feature Rows

- demographic_protocol_only / recording_year: existing importance 2.648, temporal importance nan
- demographic_protocol_only / recording_month: existing importance 1.400, temporal importance 2.378
- full_safe_metadata / recording_year: existing importance 2.001, temporal importance nan
- full_safe_metadata / recording_month: existing importance 1.082, temporal importance 2.420
