"""Graph growth, lane T: synthesised edges between released holds
(``expert_revist/graph_growth_2026_10_03/PLAN.MD`` cards T0-T5). CPU only.

Run modules as ``PYTHONPATH=.:data/scripts python -m edge_synthesis.<module>``.
"""


def provenance() -> dict:
    """The sha256 of every module of this package (what generated a record), and the git revision."""
    from pathlib import Path

    from reference_curation import ids

    here = Path(__file__).resolve().parent
    return {"git_rev": ids.git_rev(),
            "sources": {f.name: ids.sha256_file(f) for f in sorted(here.glob("*.py"))}}
