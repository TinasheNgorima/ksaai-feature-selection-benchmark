"""
make_timing_summary.py
Builds results/efficiency/timing_summary.csv from the two source files:
  results/efficiency/timing_dc_mi_xi.csv   (efficiency_dc_mi_xi.py)
  results/efficiency/timing_mic.csv        (efficiency_mic.py, genuine MINE)
timing_all.csv is no longer read (its columns were misaligned).
Run from the repository root.
"""
from pathlib import Path

import pandas as pd

EFF = Path("results/efficiency")
ORDER = ["XI", "DC", "MI", "MIC"]

base = pd.read_csv(EFF / "timing_dc_mi_xi.csv")
mic  = pd.read_csv(EFF / "timing_mic.csv")
assert "minepy" in str(mic.get("implementation", pd.Series([""])).iloc[0]), \
    "timing_mic.csv is not from the minepy timer. Rerun efficiency_mic.py."

base = base.rename(columns={"mean_s_per_feat": "mean_s_per_feature"})
df = pd.concat([base, mic], ignore_index=True)
df["method"] = df["method"].str.upper()

out = pd.DataFrame({
    "method":              df["method"],
    "n_repetitions":       df["n_repetitions"],
    "mean_time_s":         df["mean_total_s"],
    "std_time_s":          df["std_total_s"],
    "time_per_feature_ms": df["mean_s_per_feature"] * 1000,
})
dc_time = out.loc[out["method"] == "DC", "mean_time_s"].iloc[0]
xi_time = out.loc[out["method"] == "XI", "mean_time_s"].iloc[0]
out["speedup_vs_dc"] = dc_time / out["mean_time_s"]
out["xi_speedup_vs_method"] = out["mean_time_s"] / xi_time
out = out.set_index("method").loc[ORDER].reset_index()

out.to_csv(EFF / "timing_summary.csv", index=False)
print(out.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
print(f"\nSaved {EFF / 'timing_summary.csv'}")
