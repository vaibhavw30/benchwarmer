"""Regression guard: no package under backend_ml/ may shadow a stdlib module.

The harness package was renamed from `signal` to `signal_research` precisely
because `backend_ml/signal/` shadowed the stdlib `signal` module for any bare
`import signal` (joblib imports it at import time). The model scripts
(train_model.py, predict.py, backtest.py) are run from inside backend_ml/, so
Python auto-inserts backend_ml/ at sys.path[0] — a stdlib-colliding package
name there breaks `import joblib` for the whole process.

These run in subprocesses so the parent test process's import state is never
mutated. Each simulates "script launched from inside backend_ml/" by putting
backend_ml/ at sys.path[0], then confirms the stdlib still wins.
"""
import subprocess
import sys
from pathlib import Path

import pytest

BACKEND_ML = str(Path(__file__).resolve().parents[2])   # .../backend_ml


def _run(snippet):
    return subprocess.run([sys.executable, "-c", snippet],
                          capture_output=True, text=True)


def test_backend_ml_on_syspath_does_not_shadow_stdlib_signal():
    # backend_ml/ at sys.path[0] mirrors running a script from inside it.
    snippet = (
        "import sys; sys.path.insert(0, %r)\n"
        "import signal\n"
        "f = getattr(signal, '__file__', '') or ''\n"
        "print('OK' if hasattr(signal, 'SIGINT') and 'backend_ml' not in f else 'SHADOWED:' + f)"
        % BACKEND_ML
    )
    r = _run(snippet)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "OK", r.stdout + r.stderr


def test_joblib_imports_with_backend_ml_on_syspath():
    # joblib imports the stdlib `signal` module at import time — the exact
    # crash the rename fixes. Skip if joblib isn't installed for the runner.
    pytest.importorskip("joblib")
    snippet = (
        "import sys; sys.path.insert(0, %r)\n"
        "import joblib\n"
        "print('OK', joblib.__version__)"
        % BACKEND_ML
    )
    r = _run(snippet)
    assert r.returncode == 0, r.stderr
    assert r.stdout.startswith("OK"), r.stdout + r.stderr
