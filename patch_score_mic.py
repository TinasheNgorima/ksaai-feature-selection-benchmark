"""
patch_score_mic.py
Run from repo root. Replaces broken score_mic with minepy subprocess version.
"""
from pathlib import Path

MIC_PYTHON = r"C:\Users\tngorima\envs\ngorima_mic\python.exe"

NEW_SCORE_MIC = (
    'MIC_PYTHON = r"C:\\Users\\tngorima\\envs\\ngorima_mic\\python.exe"\n\n'
    'def score_mic(X: np.ndarray, y: np.ndarray) -> np.ndarray:\n'
    '    """Compute MIC scores via minepy==1.2.6 in an isolated Python 3.10 subprocess."""\n'
    '    import subprocess, tempfile, os\n'
    '    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:\n'
    '        tmp_path = tmp.name\n'
    '    np.savez(tmp_path, X=X, y=y)\n'
    '    script = (\n'
    '        "import numpy as np, sys\\n"\n'
    '        "from minepy import MINE\\n"\n'
    '        "data = np.load(sys.argv[1])\\n"\n'
    '        "X, y = data[\'X\'], data[\'y\']\\n"\n'
    '        "scores = np.zeros(X.shape[1])\\n"\n'
    '        "mine = MINE(alpha=0.6, c=15)\\n"\n'
    '        "for j in range(X.shape[1]):\\n"\n'
    '        "    try:\\n"\n'
    '        "        mine.compute_score(X[:, j], y)\\n"\n'
    '        "        scores[j] = mine.mic()\\n"\n'
    '        "    except Exception:\\n"\n'
    '        "        scores[j] = 0.0\\n"\n'
    '        "np.save(sys.argv[2], scores)\\n"\n'
    '    )\n'
    '    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as scr:\n'
    '        scr_path = scr.name\n'
    '        scr.write(script)\n'
    '    out_path = tmp_path + "_out.npy"\n'
    '    try:\n'
    '        result = subprocess.run(\n'
    '            [MIC_PYTHON, scr_path, tmp_path, out_path],\n'
    '            capture_output=True, text=True, timeout=3600\n'
    '        )\n'
    '        if result.returncode != 0:\n'
    '            raise RuntimeError(f"MIC subprocess failed:\\n{result.stderr}")\n'
    '        scores = np.load(out_path)\n'
    '    finally:\n'
    '        for p in [tmp_path, scr_path, out_path]:\n'
    '            try:\n'
    '                os.unlink(p)\n'
    '            except FileNotFoundError:\n'
    '                pass\n'
    '    return scores\n'
)

OLD_START = 'def score_mic(X: np.ndarray, y: np.ndarray) -> np.ndarray:'
OLD_END   = 'SCORERS = {'

targets = [
    Path("src/experiments/experiment1_48configs.py"),
    Path("src/experiments/stability_30reps.py"),
]

for target in targets:
    text = target.read_text(encoding="utf-8")
    if "MIC_PYTHON" in text:
        print(f"SKIP {target.name} -- already patched")
        continue
    if OLD_START not in text:
        print(f"SKIP {target.name} -- no score_mic found")
        continue
    # Split on the old function start, then cut everything up to SCORERS =
    before, rest = text.split(OLD_START, 1)
    _, after = rest.split(OLD_END, 1)
    new_text = before + NEW_SCORE_MIC + "\n" + OLD_END + after
    target.write_text(new_text, encoding="utf-8")
    print(f"PATCHED {target.name}")

print("Done.")
