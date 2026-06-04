import numpy as np, sys
sys.path.insert(0, 'src/experiments')
from experiment1_48configs import score_mic
rng = np.random.default_rng(42)
X = rng.random((100, 3)).astype(np.float32)
y = rng.random(100).astype(np.float32)
scores = score_mic(X, y)
print('MIC scores:', scores)
print('smoke test PASSED')
