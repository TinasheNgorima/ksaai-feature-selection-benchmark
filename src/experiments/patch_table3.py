path = 'src/experiments/figures_tables.py'
txt = open(path, encoding='utf-8').read()

fixes = [
    ('("LightGBM", "DC",  20)', '("lightgbm", "DC",  20)'),
    ('("LightGBM", "DC",  10)', '("lightgbm", "DC",  10)'),
    ('("LightGBM", "MI",  20)', '("lightgbm", "MI",  20)'),
    ('("LightGBM", "MIC", 20)', '("lightgbm", "MIC", 20)'),
    ('("LightGBM", "XI",  20)', '("lightgbm", "XI",  20)'),
    ('("RF",       "DC",  15)', '("rf",       "DC",  15)'),
    ('("RF",       "MI",  20)', '("rf",       "MI",  20)'),
    ('("RF",       "MIC", 20)', '("rf",       "MIC", 20)'),
    ('("RF",       "XI",  20)', '("rf",       "XI",  20)'),
    ('("Lasso",    "DC",  20)', '("lasso",    "DC",  20)'),
    ('("Lasso",    "XI",  20)', '("lasso",    "XI",  20)'),
    ('("ElasticNet","DC", 20)', '("elastic_net","DC", 20)'),
    ('"LightGBM": "LightGBM", "RF": "Random Forest",', '"lightgbm": "LightGBM", "rf": "Random Forest",'),
    ('"Lasso": "Linear models"}.get(model, model)', '"lasso": "Linear models"}.get(model, model)'),
    ('if model == "ElasticNet":', 'if model == "elastic_net":'),
    ('("LightGBM","DC",10),', '("lightgbm","DC",10),'),
    ('("RF","XI",20)]:', '("rf","XI",20)]:'),
]

for old, new in fixes:
    if old in txt:
        txt = txt.replace(old, new)
        print(f"  OK: {old[:50]}")
    else:
        print(f"  MISS: {old[:50]}")

open(path, 'w', encoding='utf-8').write(txt)
print("Done.")
