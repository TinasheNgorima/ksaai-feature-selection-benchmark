"""
Generates results/efficiency/timing_summary.csv from timing_all.csv.
Run from repo root in ngorima_mic environment.
"""
import pandas as pd
from pathlib import Path

df = pd.read_csv('results/efficiency/timing_all.csv')

# Normalise method labels to match METHOD_LABEL keys
label_map = {'xi': 'XI', 'mi': 'MI', 'dc': 'DC', 'MIC': 'MIC'}
df['method'] = df['method'].map(label_map).fillna(df['method'])

# Build summary columns
out = pd.DataFrame()
out['method']              = df['method']
out['mean_time_s']         = df['mean_total_s']
out['std_time_s']          = df['std_total_s']

# time_per_feature_ms: prefer mean_s_per_feature if available, else mean_s_per_feat
def get_tpf(row):
    if pd.notna(row.get('mean_s_per_feature')):
        return row['mean_s_per_feature'] * 1000
    return row['mean_s_per_feat'] * 1000

out['time_per_feature_ms'] = df.apply(get_tpf, axis=1)

# speedup relative to DC
dc_time = out.loc[out['method'] == 'DC', 'mean_time_s'].values[0]
out['speedup_vs_dc'] = dc_time / out['mean_time_s']

Path('results/efficiency').mkdir(parents=True, exist_ok=True)
out.to_csv('results/efficiency/timing_summary.csv', index=False)
print(out.to_string(index=False))
print("\ntiming_summary.csv written.")
