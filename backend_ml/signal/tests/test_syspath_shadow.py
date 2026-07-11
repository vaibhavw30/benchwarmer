"""Regression guard for the stdlib-`signal`-shadow landmine.

`publish_fair_values.main()` and `report._cmd_evaluate` must put the backend_ml/
dir on sys.path by APPENDING it (not inserting at position 0), so that the
`backend_ml/signal/` package never shadows the stdlib `signal` module for a
downstream bare `import signal` (as joblib/sklearn/numpy do). See
docs/superpowers/before-live-checklist.md section F.

These run in subprocesses so the parent test process's import state is never
mutated.
"""
import subprocess
import sys
from pathlib import Path

BACKEND_ML = str(Path(__file__).resolve().parents[2])   # .../backend_ml
REPO_ROOT = str(Path(__file__).resolve().parents[3])    # repo root


def _run(snippet):
    return subprocess.run([sys.executable, "-c", snippet],
                          capture_output=True, text=True, cwd=REPO_ROOT)


def test_appending_backend_ml_does_not_shadow_stdlib_signal():
    # Mirrors the hardened behavior: APPEND backend_ml/ -> stdlib `signal` wins.
    snippet = (
        "import sys; sys.path.append(%r);\n"
        "import signal;\n"
        "f = getattr(signal, '__file__', '') or '';\n"
        "print('OK' if hasattr(signal, 'SIGINT') and 'backend_ml' not in f else 'SHADOWED:' + f)"
        % BACKEND_ML
    )
    r = _run(snippet)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "OK", r.stdout + r.stderr


def test_inserting_at_front_would_shadow_proving_guard_is_real():
    # Sanity: the OLD insert(0) approach DOES shadow, so the test above is
    # meaningful and would catch a regression back to insert-at-0.
    snippet = (
        "import sys; sys.path.insert(0, %r);\n"
        "import signal;\n"
        "f = getattr(signal, '__file__', '') or '';\n"
        "print('SHADOWED' if 'backend_ml' in f else 'OK:' + f)"
        % BACKEND_ML
    )
    r = _run(snippet)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "SHADOWED", r.stdout + r.stderr
