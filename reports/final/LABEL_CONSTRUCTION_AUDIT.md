# Label Construction Audit

This note records the label source used by the project artifacts and separates label-definition mismatch from acoustic/domain shift.

- Coswara: analytic labels are taken from the processed project metadata `label_binary` column after the repository's positive/negative mapping.
- COUGHVID: analytic labels are taken from the processed external metadata `label_binary` column used for transfer evaluation.
- These labels are not assumed to be clinically identical across datasets; cross-dataset transfer is therefore interpreted as an external stress test affected by label-construction and collection-protocol differences.

Manuscript implication: external-transfer failure should be described as a real-world dataset-transfer failure, not as proof that the same clinical label has identical annotation semantics across corpora.
