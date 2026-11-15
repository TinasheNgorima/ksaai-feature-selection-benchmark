"""Locate the Python 3.10 environment that provides minepy==1.2.6.

MIC scoring runs in a separate interpreter because minepy does not build on
Python 3.12 or later. Resolution order:
  1. the MIC_PYTHON environment variable, if set (path to a Python 3.10
     interpreter with minepy installed);
  2. otherwise the conda environment named MIC_ENV_NAME, via `conda run`.
"""
import os
import shutil

MIC_ENV_NAME = "ngorima_mic"


def mic_python_cmd():
    """Return the command prefix that runs Python in the minepy environment."""
    exe = os.environ.get("MIC_PYTHON")
    if exe:
        return [exe]
    conda = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if conda:
        return [conda, "run", "-n", MIC_ENV_NAME, "python"]
    raise EnvironmentError(
        f"MIC scoring needs the '{MIC_ENV_NAME}' conda environment "
        "(see README > Installation). Without conda, set MIC_PYTHON to a "
        "Python 3.10 interpreter with minepy==1.2.6 installed."
    )
