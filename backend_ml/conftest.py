import os
import sys

# backend_ml has no __init__.py (namespace package). With
# `consider_namespace_packages = true` set at repo root (needed so
# backend_ml/signal doesn't get collected as a top-level module named
# "signal", shadowing the stdlib module of the same name), pytest's
# import-mode rootpath computation for sibling-style test modules in
# this directory (e.g. test_player_impact.py, which does
# `from player_impact_engine import ...`) walks up to the repo root
# instead of stopping at backend_ml/. That means backend_ml/ itself is
# no longer guaranteed to be on sys.path.
#
# Explicitly put this directory on sys.path so those sibling imports
# keep resolving, matching pre-existing behavior.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
