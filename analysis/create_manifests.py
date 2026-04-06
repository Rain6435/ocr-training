import pandas as pd
import os

test_csv = "data/processed/test.csv"
output_dir = "data/processed"

df = pd.read_csv(test_csv)
print(f"Total samples: {len(df)}")
print(f"Difficulty distribution:\n{df['difficulty'].value_counts()}")

# Create separate manifests for each difficulty level (max 200 per level)
for difficulty in ["easy", "medium", "hard"]:
    subset = df[df["difficulty"] == difficulty].head(200)
    output_path = os.path.join(output_dir, f"test_{difficulty}.csv")
    subset.to_csv(output_path, index=False)
    print(f"Created {output_path} with {len(subset)} samples")

# Display the breakdowns
for diff in ["easy", "medium", "hard"]:
    path = os.path.join(output_dir, f"test_{diff}.csv")
    df_subset = pd.read_csv(path)
    print(f"\n{diff.upper()}:")
    print(f"  Samples: {len(df_subset)}")
    print(f"  Sources: {df_subset['source'].value_counts().to_dict()}")
