# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write the routing table that labels every motion of a release with one expert (card S1 of
``expert_revist/graph_growth_2026_10_03/PLAN.MD``).

``MultiExpertSupervisedAgent`` routes each env to the expert owning its clip, through a table of motion names in
library order (``package_student_corpus.py`` writes it for multi-expert corpora). A release-v3 student distils
one teacher, so the table is all zeros. The motion order is the release graph's, which
``ContactGraph.validate_against_motion_lib`` ties to the packaged library at every construction, and the agent
checks the names again against the library it loads.

    PYTHONPATH=. python data/scripts/make_single_expert_routing.py \\
        --graph data/smpl/reference_curation/<release>/contact_graph.pt \\
        --expert results/<expert run>/epoch_3420.ckpt --out data/smpl/student_release_v3/<name>.experts.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--graph", required=True, help="the release's contact_graph.pt (motion order)")
    parser.add_argument("--expert", required=True, help="the one expert checkpoint (recorded, not loaded)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from protomotions.agents.supervised.expert_port import single_expert_routing
    from protomotions.components.contact_graph import ContactGraph
    from protomotions.utils.release_identity import file_sha256

    graph = ContactGraph.from_file(args.graph)
    table = single_expert_routing(list(graph.motion_names))
    table.update(
        experts=[args.expert],
        graph=args.graph,
        graph_sha256=file_sha256(args.graph),
        generator="data/scripts/make_single_expert_routing.py (S1)",
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table, indent=1))
    print(f"wrote {out}: {len(table['motion_expert'])} motions -> expert 0 ({args.expert})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
