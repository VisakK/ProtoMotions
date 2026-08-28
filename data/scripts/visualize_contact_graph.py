# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interactive HTML view of a contact graph: drag, zoom, filter.

Renders ``contact_graph.json`` as a self-contained vis-network page via PyVis:

* **nodes** are contact configurations — sized by total trusted dwell, colored
  by orientation bin, labeled with the compact zone form; the tooltip carries
  the full configuration string, dwell, segment/clip counts, and whether a
  frozen HOLD clip exists for the node;
* **edges** are observed transitions — width by occurrence count, directed,
  with the occurrences (clip + hold-to-hold times) in the tooltip;
* the built-in **filter menu** (top of the page) filters nodes/edges by any
  attribute this script attaches: ``orientation``, ``dwell_s``, ``num_clips``,
  ``has_hold``, edge ``count``. Drag and zoom are native; selecting a node
  highlights its neighborhood.

The page is fully self-contained (``cdn_resources='in_line'``): it opens from
disk with no network access.

Usage::

    PYTHONPATH=. python data/scripts/visualize_contact_graph.py \
      --graph-dir data/smpl/yoga_contact_graph_student44h \
      --out output/contact_graph_student44h.html
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

# Fixed-order categorical assignment over the graph's own orientation order
# (validated with the dataviz palette checker: CVD dE 9.1, normal dE 19.6 on
# the light surface; the contrast WARN is relieved by the always-on labels).
ORIENTATION_COLORS = {
    "upright": "#2a78d6",
    "inverted": "#eb6834",
    "prone": "#1baf7a",
    "supine": "#eda100",
    "side_l": "#e87ba4",
    "side_r": "#008300",
}
EDGE_COLOR = "#b8b7ae"
SURFACE = "#fcfcfb"
INK = "#1a1a19"

# Same compact zone labels as the training-time viz panel
# (protomotions/agents/evaluators/sequence_viz.py); duplicated so this script
# stays importable without torch.
ZONE_SHORT = {
    "L_FOOT": "LF", "R_FOOT": "RF", "L_SHANK": "LS", "R_SHANK": "RS",
    "L_THIGH": "LT", "R_THIGH": "RT", "PELVIS": "PV", "TRUNK": "TK",
    "HEAD": "HD", "L_UPPER_ARM": "LU", "R_UPPER_ARM": "RU",
    "L_FOREARM": "LA", "R_FOREARM": "RA", "L_HAND": "LH", "R_HAND": "RH",
}


def short_label(key: str) -> str:
    if "@" not in key:
        return key
    pairs_part, orient = key.rsplit("@", 1)
    out = []
    for pair in pairs_part.split("|"):
        if pair.endswith(":G"):
            out.append(ZONE_SHORT.get(pair[:-2], pair[:-2]))
        elif "+" in pair:
            a, b = pair.split("+", 1)
            out.append(f"{ZONE_SHORT.get(a, a)}·{ZONE_SHORT.get(b, b)}")
        else:
            out.append(pair)
    return " ".join(out) + f"\n@{orient}"


def build_network(graph: dict, args):
    from pyvis.network import Network

    net = Network(
        height="920px",
        width="100%",
        directed=True,
        bgcolor=SURFACE,
        font_color=INK,
        select_menu=True,
        filter_menu=True,
        cdn_resources="in_line",
    )

    kept_nodes = set()
    for node_id, row in enumerate(graph["nodes"]):
        dwell = float(row["total_dwell_s"])
        if dwell < args.min_dwell_s:
            continue
        orientation = row["orientation_bin"]
        motions = row.get("motions", [])
        has_hold = any(m.startswith("hold_") for m in motions)
        clip_list = "\n  ".join(motions[:8]) + ("\n  ..." if len(motions) > 8 else "")
        net.add_node(
            node_id,
            label=short_label(row["key"]),
            title=(
                f"node {node_id}: {row['key']}\n"
                f"dwell {dwell:.1f}s over {row['num_segments']} segments, "
                f"{len(motions)} clips{'  [HOLD clip]' if has_hold else ''}\n"
                f"clips:\n  {clip_list}"
            ),
            color=ORIENTATION_COLORS.get(orientation, "#4a3aa7"),
            size=10 + 3.5 * math.sqrt(dwell),
            borderWidth=3 if has_hold else 1,
            orientation=orientation,
            dwell_s=round(dwell, 1),
            num_clips=len(motions),
            has_hold=has_hold,
            font={"color": INK, "size": 13, "multi": False},
        )
        kept_nodes.add(node_id)

    kept_edges = 0
    for edge in graph["edges"]:
        if edge["count"] < args.min_edge_count:
            continue
        if edge["src"] not in kept_nodes or edge["dst"] not in kept_nodes:
            continue
        occurrences = edge["occurrences"]
        occ_lines = "\n  ".join(
            f"{o['motion'][:52]} ({o['t_hold_src']:.1f}s -> {o['t_hold_dst']:.1f}s)"
            for o in occurrences[:6]
        ) + ("\n  ..." if len(occurrences) > 6 else "")
        net.add_edge(
            edge["src"],
            edge["dst"],
            title=f"x{edge['count']}  {len(set(o['motion'] for o in occurrences))} clips\n  {occ_lines}",
            width=1.0 + 0.35 * min(edge["count"], 20),
            color={"color": EDGE_COLOR, "opacity": 0.75},
            count=edge["count"],
            arrows="to",
        )
        kept_edges += 1

    net.set_options(json.dumps({
        "physics": {
            "barnesHut": {
                "gravitationalConstant": -6000,
                "springLength": 160,
                "springConstant": 0.02,
                "damping": 0.25,
            },
            "minVelocity": 0.5,
            "stabilization": {"iterations": 300},
        },
        "interaction": {
            "hover": True,
            "tooltipDelay": 120,
            "navigationButtons": True,
            "keyboard": True,
        },
        "edges": {"smooth": {"type": "continuous"}},
    }))
    return net, len(kept_nodes), kept_edges


def inline_bootstrap(html: str) -> str:
    """Replace the CDN bootstrap tags with the vendored copies.

    PyVis embeds vis-network when ``cdn_resources='in_line'`` but its
    select/filter menu template still points at jsdelivr; inlining the two
    files (data/scripts/assets/) makes the page genuinely offline-capable.
    """
    assets = Path(__file__).resolve().parent / "assets"
    css, js = assets / "bootstrap.min.css", assets / "bootstrap.bundle.min.js"
    if not (css.is_file() and js.is_file()):
        print("WARNING: vendored bootstrap missing; the filter menu will need "
              "network access (see data/scripts/assets/)")
        return html
    import re

    css_tag = "<style>" + css.read_text() + "</style>"
    js_tag = "<script>" + js.read_text() + "</script>"
    # Function replacements: the payloads contain backslash sequences that
    # re.sub would otherwise parse as escapes.
    html = re.sub(
        r'<link[^>]*bootstrap[^>]*\.css[^>]*/?>', lambda _m: css_tag, html, count=1
    )
    html = re.sub(
        r'<script[^>]*bootstrap[^>]*\.js[^>]*>\s*</script>',
        lambda _m: js_tag,
        html,
        count=1,
    )
    return html


def inject_legend(html: str, graph_name: str, num_nodes: int, num_edges: int) -> str:
    swatches = "".join(
        f'<span style="display:inline-flex;align-items:center;margin-right:12px">'
        f'<span style="width:12px;height:12px;border-radius:50%;background:{color};'
        f'display:inline-block;margin-right:5px"></span>{name}</span>'
        for name, color in ORIENTATION_COLORS.items()
    )
    legend = (
        f'<div style="font-family:sans-serif;color:{INK};background:{SURFACE};'
        f'padding:10px 14px;border-bottom:1px solid #e3e2da">'
        f'<b>{graph_name}</b> — {num_nodes} nodes, {num_edges} edges. '
        f'Node size = trusted dwell; thick border = a frozen HOLD clip exists; '
        f'edge width = transition count. Drag nodes, scroll to zoom, use the '
        f'filter bar above the canvas.<br><span style="font-size:13px">{swatches}'
        f'</span></div>'
    )
    return html.replace("<body>", "<body>" + legend, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dir", default="data/smpl/yoga_contact_graph_student44h")
    parser.add_argument("--out", default=None,
                        help="output HTML (default output/contact_graph_<dirname>.html)")
    parser.add_argument("--min-edge-count", type=int, default=1,
                        help="hide edges observed fewer times than this")
    parser.add_argument("--min-dwell-s", type=float, default=0.0,
                        help="hide nodes with less total trusted dwell than this")
    args = parser.parse_args()

    graph_dir = Path(args.graph_dir)
    graph = json.loads((graph_dir / "contact_graph.json").read_text())
    out = Path(args.out) if args.out else Path(
        f"output/contact_graph_{graph_dir.name.replace('yoga_contact_graph_', '')}.html"
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    net, num_nodes, num_edges = build_network(graph, args)
    html = net.generate_html(str(out))
    html = inline_bootstrap(html)
    html = inject_legend(html, graph_dir.name, num_nodes, num_edges)
    out.write_text(html)
    print(f"{num_nodes} nodes, {num_edges} edges -> {out}")
    print("open it directly in a browser; the page is self-contained.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
