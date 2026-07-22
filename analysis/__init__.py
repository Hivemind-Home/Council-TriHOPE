"""Analysis package: turns runs/*/metrics.jsonl + events.jsonl into the
paper's tables and figures.

Requires the ``analysis`` extra: ``pip install -e ".[analysis]"``.

    python -m analysis.run_report runs/baselines_small_v1
"""

from .loaders import RunData, load_experiment, load_run  # noqa: F401
