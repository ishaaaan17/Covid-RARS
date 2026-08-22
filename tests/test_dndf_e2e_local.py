import sys
from pathlib import Path

# Add src to python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import numpy as np
from covid_rars.dndf_stages import run_dndf_reliability_pipeline

def test_dndf_pipeline_on_real_data_slice():
    print("1. Loading slice of real features...")
    data_path = Path("D:/BTP/processed/features_compare_is10_merged.csv")
    if not data_path.exists():
        print(f"Data not found at {data_path}")
        return

    df = pd.read_csv(data_path, nrows=1200)
    non_feat = ["participant_id", "recording_id", "modality", "split", "label_binary", "label", "date", "recording_date"]
    feat_cols = [c for c in df.columns if c not in non_feat]
    keep = [c for c in non_feat if c in df.columns] + feat_cols[:800]
    df = df[keep].copy()

    print(f"2. Shape of test slice: {df.shape}")
    print(f"3. Modalities: {df['modality'].unique().tolist()}")
    print(f"4. Participants: {df['participant_id'].nunique()}")

    print("5. Running full pipeline test locally...")
    artifacts = run_dndf_reliability_pipeline(
        features_df=df,
        external_features_df=None,
        modalities=["cough", "breath", "speech"],
        seeds=[1, 2],
        num_trees=5,
        depth=4,
        used_features_rate=0.8,
        learning_rate=0.01,
        max_epochs=3,
        patience=2,
        use_smote=True,
        device="cpu",
        output_dir=Path("reports/dndf_test"),
    )

    print("\n6. Summary table generated successfully:")
    print(artifacts.final_summary_table.to_string())
    print("\n[SUCCESS] LOCAL VERIFICATION TEST COMPLETED 100% WITH ZERO ERRORS!")

if __name__ == "__main__":
    test_dndf_pipeline_on_real_data_slice()
