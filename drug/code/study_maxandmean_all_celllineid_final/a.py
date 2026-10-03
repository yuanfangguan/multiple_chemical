import pandas as pd

for f in [
    "../../data/features/features_maccs.csv",
    "../../data/features/features_morgan.csv",
    "../../data/features/features_rdkitfp.csv",
    "../../data/features/features_descriptors.csv",
    "../../data/features/features_mordred.csv",
]:
    df = pd.read_csv(f)
    print(f"\n{f}")
    print(df["molecule"].head(20))

