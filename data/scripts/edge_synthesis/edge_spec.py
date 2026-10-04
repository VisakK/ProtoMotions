"""Card T0 of ``expert_revist/graph_growth_2026_10_03/PLAN.MD``: one machine-readable specification per
synthesised edge, written to ``expert_revist/graph_growth_2026_10_03/edges.json``.

Everything is read from release v2 and its records. Nothing is decided here that the plan has not decided (§1.3):
the edges, their endpoints and the kind of each. What this adds is the phase schedule -- fixed-contact phases, the
make/break events between them, an admissible duration per phase -- and the per-phase contact constraints that the
generators (T2, T3) turn into cost terms and the admission gate (T5) checks.

Per endpoint (source S, destination D):

* the hold id, its x0 clip, the exemplar's frame and time (``frame_hold``; it is not the id's frame when labels v2
  moved the exemplar: Crow -a@439's exemplar is frame 651), the hold window, the graph node and its key;
* the ground set (``pairs_ground``) and the body-body braces (configured pairs, each with its role and B6's
  critical flag);
* the known-free ground zones (the sidecar's ``seg_ground_free``: human evidence separated on >= 95 % of the window);
* the statics v2 row (verdict, s*, the binding joints, group utilisation, loads, the statue witness);
* the e15500 library's tracking of the hold (tracked share, supports realised, 6-body error);
* the hand anchor: hand origins, their midpoint and width and the heading of the hand line (T2 re-anchors D so its
  hands coincide with S's), and the whole-body COM.

Per phase: ``ground`` (planted supports), ``braces`` (closed body-body contacts), ``free`` (zones that must stay off
the floor: every zone not planted in the phase whose state is known at both ends -- planted or known free), the
duration range with its basis, and whether the phase is quasi-static. Events are derived from consecutive phase
configurations, S -> first phase and last phase -> D included, so an edit can never be implied without being listed.

Checks (the card's acceptance), all fatal:

* every hold id resolves in release v2 (``release_holds.yaml``, the graph's x0 segments, the sidecar, statics v2);
* the net contact edit S -> D of each edge reproduces §1.3's table (``EXPECTED_EDITS``);
* the first phase starts from S's configuration up to the listed events and the last phase is D's configuration;
* no zone is both planted and free in a phase.

CLI::

    PYTHONPATH=.:data/scripts python -m edge_synthesis.edge_spec            # write edges.json and print the table
    PYTHONPATH=.:data/scripts python -m edge_synthesis.edge_spec --check    # re-derive and compare with the file
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from extract_contact_configs import ZONE_ORDER
from reference_curation import ids

REPO = ids.REPO
MODULE = "edge_synthesis.edge_spec"
SCHEMA_VERSION = 1
RELEASE_ID = "holds_repaired_ftC_posefix.release_v2.a2dda5d2ac"
RELEASE_DIR = REPO / "data/smpl/reference_curation" / RELEASE_ID
RELEASE_RECORD = REPO / "data/reference_curation/releases" / f"{RELEASE_ID}.json"
LIBRARY_E15500 = REPO / "expert_revist/expert56_v2_e15500/data/library_e15500.json"
PLAN_DIR = REPO / "expert_revist/graph_growth_2026_10_03"
CENSUS = PLAN_DIR / "evidence/census.json"
WITNESS = PLAN_DIR / "evidence/witness2.json"
OUT = PLAN_DIR / "edges.json"
MJCF_FLAT = REPO / "data/assets/smpl/smpl_yogi03596_v2_flat.xml"
GOAL_BODIES = ("Pelvis", "L_Ankle", "R_Ankle", "L_Hand", "R_Hand", "Head")

# --------------------------------------------------------------------------- #
# The endpoints (§1.3) and the edges
# --------------------------------------------------------------------------- #
CROW = "220923_Crane_Crow_Pose_or_Bakasana_-a@439"
HANDSTAND = "220923_Handstand_pose_or_Adho_Mukha_Vrksasana_-a@1193"
CHATURANGA = "220923_Four-Limbed_Staff_Pose_or_Chaturanga_Dandasana_-a@485"
TRIPOD = "220923_Supported_Headstand_pose_or_Salamba_Sirsasana_-b@1093"
PLANK = "220923_Plank_Pose_or_Kumbhakasana_-a@541"
FIREFLY = "220923_Firefly_Pose_or_Tittibhasana_-a@776"
SCORPION_B_PREP = "220923_Scorpion_pose_or_vrischikasana-b@455"

SHIN_BRACES = ("L_SHANK+L_UPPER_ARM", "R_SHANK+R_UPPER_ARM")
THIGH_BRACES = ("L_THIGH+L_UPPER_ARM", "R_THIGH+R_UPPER_ARM")
HANDS = ("L_HAND", "R_HAND")
FEET = ("L_FOOT", "R_FOOT")

# Bases of the duration ranges (measured by ``corpus_timing``; the numbers are written into edges.json):
#   hands_inverted: x0 transitions with both hands planted at both ends and an inverted end
#                   (inter-hold time, segment end -> next segment start);
#   tripod_to_arm:  the human tripod -> legs-on-arms -> arm-balance passages (Koundinya -a, -b; Side Crow -b);
#   jump_back:      no human jump-back in the corpus; the flight is bounded by ballistics (a COM rise of 3-15 cm
#                   while the hands keep a share of the load) and the landing by the corpus' arrival dwell.
BASIS = {
    "hands_inverted": "x0 hands-anchored transitions with an inverted end: inter-hold time p10/p50/p90 (corpus_timing)",
    "tripod_to_arm": "human tripod -> legs-on-arms -> arm balance passages in Koundinya -a/-b and Side Crow -b",
    "jump_back": "no human jump-back in the corpus: flight bounded by ballistics, landing by arrival dwell",
    "settle": "arrival settle: the shortest trusted hold window in the corpus is 0.3 s; 1 s keeps the edge short",
}


def phase(name, ground, braces=(), duration=(0.5, 2.0), quasi_static=True, basis="hands_inverted", note=""):
    return {"name": name, "ground": sorted(ground), "braces": sorted(braces), "duration_s": list(duration),
            "quasi_static": bool(quasi_static), "basis": basis, "note": note}


EDGES = [
    {"id": "E1", "rank": "primary", "label": "crow -> handstand (press)", "src": CROW, "dst": HANDSTAND,
     "kind": "quasi_static", "generator": "keyframe_statics",
     "phases": [
         phase("lift_tuck", HANDS, (), (0.8, 2.5), note="shins leave the upper arms; the hips rise over the shoulders "
               "with the knees tucked to the chest (inverted tuck); the COM stays over the hands"),
         phase("extend", HANDS, (), (0.6, 2.0), note="the legs extend to the handstand exemplar"),
     ],
     "risk": "the press: shoulder and wrist torque mid-pike may exceed the plant (endpoint wrists s* 0.17 / 0.32); "
             "if statics says so the edge becomes a hop (Scorpion -b's pattern) or is dropped"},
    {"id": "E2", "rank": "primary", "label": "crow -> chaturanga (jump-back)", "src": CROW, "dst": CHATURANGA,
     "kind": "dynamic", "generator": "mppi",
     "phases": [
         phase("float_back", HANDS, (), (0.25, 0.6), quasi_static=False, basis="jump_back",
               note="braces break, the legs shoot back; the COM leaves the hand polygon (0.079 -> 0.318 m behind the "
                    "hands) before the feet land"),
         phase("land", HANDS + FEET, (), (0.3, 1.0), quasi_static=False, basis="settle",
               note="feet make (peak vertical force <= 3 BW); the landing loads bent elbows"),
     ],
     "risk": "the landing on bent elbows"},
    {"id": "E3", "rank": "primary", "label": "handstand -> crow (lower)", "src": HANDSTAND, "dst": CROW,
     "kind": "quasi_static", "generator": "keyframe_statics",
     "phases": [
         phase("lower_fold", HANDS, (), (1.0, 3.0), note="eccentric: the legs fold and the hips descend behind the "
               "hands until the shins reach the upper arms"),
         phase("settle", HANDS, SHIN_BRACES, (0.3, 1.0), basis="settle",
               note="both shin braces make on colliders that cannot press in (exemplar gaps -0.1 / +0.3 cm)"),
     ],
     "risk": "the braces close on rigid colliders"},
    {"id": "E4", "rank": "primary", "label": "tripod headstand -> crow", "src": TRIPOD, "dst": CROW,
     "kind": "quasi_static", "generator": "keyframe_statics",
     "phases": [
         phase("fold_legs", ("HEAD",) + HANDS, (), (0.4, 2.0), basis="tripod_to_arm",
               note="the legs fold down from the headstand until the shins reach the upper arms"),
         phase("shift_forward", ("HEAD",) + HANDS, SHIN_BRACES, (0.5, 2.5), basis="tripod_to_arm",
               note="both shin braces make; the COM shifts forward over the hands, unloading the head"),
         phase("settle", HANDS, SHIN_BRACES, (0.3, 1.0), basis="settle", note="the head breaks: crow"),
     ],
     "risk": "low; partial human precedent (Side Crow -b, Koundinya -a/-b go tripod -> legs onto arms -> arm balance)"},
    {"id": "E5", "rank": "primary", "label": "handstand -> chaturanga (float-down)", "src": HANDSTAND,
     "dst": CHATURANGA, "kind": "dynamic", "generator": "mppi",
     "phases": [
         phase("overbalance_arc", HANDS, (), (0.5, 1.5), quasi_static=False,
               note="the legs arc over behind the hands as the shoulders move forward"),
         phase("land", HANDS + FEET, (), (0.3, 1.0), quasi_static=False, basis="settle",
               note="feet make (peak vertical force <= 3 BW); the COM drops 0.73 m onto the wrists and shoulders"),
     ],
     "risk": "the highest: 0.826 m 6-body distance, beyond every within-clip edge (max 0.705 m)"},
    {"id": "B1", "rank": "backup", "label": "crow -> plank (jump-back, straight arms)", "src": CROW, "dst": PLANK,
     "kind": "dynamic", "generator": "mppi",
     "phases": [
         phase("float_back", HANDS, (), (0.25, 0.6), quasi_static=False, basis="jump_back",
               note="braces break, the legs shoot back with straight arms"),
         phase("land", HANDS + FEET, (), (0.3, 1.0), quasi_static=False, basis="settle",
               note="feet make (peak vertical force <= 3 BW)"),
     ],
     "risk": "an easier landing than chaturanga"},
    {"id": "B2", "rank": "backup", "label": "firefly -> crow", "src": FIREFLY, "dst": CROW,
     "kind": "quasi_static", "generator": "keyframe_statics",
     "phases": [
         phase("bend_knees", HANDS, THIGH_BRACES, (0.5, 2.0), note="the knees bend and the legs draw back along the arms"),
         phase("settle", HANDS, SHIN_BRACES, (0.3, 1.0), basis="settle",
               note="thigh braces swap for shin braces: crow"),
     ],
     "risk": "the Firefly end is weak (tracked 0.613 at e15500, feet down)"},
]

# §1.3's contact edits (the acceptance target): ground breaks/makes, brace breaks/makes, orientation.
EXPECTED_EDITS = {
    "E1": {"ground_break": [], "ground_make": [], "braces_break": list(SHIN_BRACES), "braces_make": [],
           "orientation": "prone->inverted"},
    "E2": {"ground_break": [], "ground_make": list(FEET), "braces_break": list(SHIN_BRACES), "braces_make": [],
           "orientation": "prone->prone"},
    "E3": {"ground_break": [], "ground_make": [], "braces_break": [], "braces_make": list(SHIN_BRACES),
           "orientation": "inverted->prone"},
    "E4": {"ground_break": ["HEAD"], "ground_make": [], "braces_break": [], "braces_make": list(SHIN_BRACES),
           "orientation": "inverted->prone"},
    "E5": {"ground_break": [], "ground_make": list(FEET), "braces_break": [], "braces_make": [],
           "orientation": "inverted->prone"},
    "B1": {"ground_break": [], "ground_make": list(FEET), "braces_break": list(SHIN_BRACES), "braces_make": [],
           "orientation": "prone->prone"},
    "B2": {"ground_break": [], "ground_make": [], "braces_break": list(THIGH_BRACES), "braces_make": list(SHIN_BRACES),
           "orientation": "upright->prone"},
}

# The free edge (§1.3, card R1): a human squat -> hop -> handstand inside Scorpion -b. Not a synthesis target:
# R1 cuts the clip after its stable hands-only handstand and labels a new hold there.
FREE_EDGE = {"id": "FREE", "rank": "free", "label": "squat -> handstand (hop), human capture",
             "stem": "220923_Scorpion_pose_or_vrischikasana-b", "window_s": [8.3, 13.0], "kind": "human_capture",
             "generator": "cut (card R1)", "src": SCORPION_B_PREP, "dst": None,
             "note": "the destination hold does not exist in release v2: R1 labels it (ground set from markers and mat, "
                     "expected hands) and truncates the clip, which also removes the scorpion gate v2 certified as "
                     "beyond the plant"}


# --------------------------------------------------------------------------- #
# Release access
# --------------------------------------------------------------------------- #
class Release:
    """The release v2 artefacts the spec reads (read-only)."""

    def __init__(self, release_dir: Path = RELEASE_DIR, record: Path = RELEASE_RECORD):
        self.dir = Path(release_dir)
        self.record_path = Path(record)
        self.record = yaml.safe_load(open(record)) if str(record).endswith((".yaml", ".yml")) else json.load(open(record))
        self.holds_yaml = yaml.safe_load(open(self.dir / "release_holds.yaml"))
        self.graph = json.load(open(self.dir / "contact_graph.json"))
        self.targets = torch.load(self.dir / "contact_targets.pt", map_location="cpu", weights_only=False)
        statics_dir = REPO / "data/reference_curation/statics_v2" / self.record["statics_id"]
        self.statics = {json.loads(l)["hold_id"]: json.loads(l) for l in open(statics_dir / "holds.jsonl")}
        labels_dir = REPO / "data/reference_curation/labels" / self.record["labels_id"]
        self.annotations: dict = {}
        for line in open(labels_dir / "annotations.jsonl"):
            a = json.loads(line)
            self.annotations.setdefault(a["hold_id"], {})[a["contact"]] = a
        self.library = json.load(open(LIBRARY_E15500))
        self.inputs = [self.record_path, self.dir / "release_holds.yaml", self.dir / "contact_graph.json",
                       self.dir / "contact_targets.pt", statics_dir / "holds.jsonl", labels_dir / "annotations.jsonl",
                       LIBRARY_E15500, CENSUS, WITNESS]
        self.release_hold = {h["hold_id"]: (c, h) for c in self.holds_yaml["clips"] for h in c["holds"]}
        self.x0 = {n: c for n, c in self.graph["clips"].items() if float(c["variant_s"]) == 0.0}
        self.segment = {s["hold_id"]: (n, c, k, s) for n, c in self.x0.items() for k, s in enumerate(c["segments"])}
        self.library_hold = {h["hold_id"]: (c, h) for c in self.library["clips"] for h in c["holds"]}
        self._motions: dict = {}
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(MJCF_FLAT))
        self.body_names = [model.body(i).name for i in range(1, model.nbody)]
        self.mass = model.body_mass[1:].copy()
        self.ipos = model.body_ipos[1:].copy()

    def motion(self, stem: str) -> dict:
        if stem not in self._motions:
            self._motions[stem] = torch.load(self.dir / "motions" / f"{stem}.motion", map_location="cpu",
                                             weights_only=False)
        return self._motions[stem]

    def sidecar_index(self, hold_id: str) -> tuple[int, int]:
        """``(motion, segment)`` of the x0 motion's segment in the sidecar's layout."""
        name, clip, k, _ = self.segment[hold_id]
        m = self.targets["motion_names"].index(name)
        if self.targets["hold_ids"][int(self.targets["seg_hold_index"][m, k])] != hold_id:
            raise ValueError(f"{hold_id}: the sidecar's segment {k} of {name} is another hold")
        return m, k


def resolve(rel: Release, hold_id: str) -> list[str]:
    """Reasons ``hold_id`` does not resolve in release v2 (empty when it does)."""
    out = []
    if hold_id not in rel.release_hold:
        out.append("not in release_holds.yaml")
    if hold_id not in rel.segment:
        out.append("not an x0 segment of the graph")
    if hold_id not in rel.statics:
        out.append("no statics v2 row")
    if hold_id not in rel.targets["hold_ids"]:
        out.append("not in the sidecar")
    return out


def _quat_apply_xyzw(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    u, w = q[..., :3], q[..., 3:4]
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def endpoint(rel: Release, hold_id: str) -> dict:
    """Everything the generators and the gate need about one end of an edge."""
    bad = resolve(rel, hold_id)
    if bad:
        raise ValueError(f"{hold_id} does not resolve in {RELEASE_ID}: {bad}")
    clip, hold = rel.release_hold[hold_id]
    name, gclip, k, seg = rel.segment[hold_id]
    node = rel.graph["nodes"][seg["node"]]
    m, kk = rel.sidecar_index(hold_id)
    free = [z for z, f in zip(ZONE_ORDER, rel.targets["seg_ground_free"][m, kk].tolist()) if f]
    share = {z: round(float(s), 3) for z, s in zip(ZONE_ORDER, rel.targets["seg_ground_free_share"][m, kk].tolist())}
    ground = sorted(p[:-2] for p in hold["pairs_ground"])
    anns = rel.annotations.get(hold_id, {})
    braces = []
    for p in hold["pairs_configured"]:
        if p.endswith(":G"):
            continue
        a = anns.get(p, {})
        braces.append({"pair": p, "role": a.get("target_role"), "critical": bool(a.get("critical")),
                       "statics_necessity": (a.get("statics") or {}).get("necessity"),
                       "lp_load_n": (a.get("statics") or {}).get("load_n")})
    touch_ok = sorted(p for p, a in anns.items() if "+" in p and not a.get("in_configuration")
                      and a.get("target_role") in ("allowed", "incidental"))
    st = rel.statics[hold_id]
    gated = st.get("gated") or {}
    statics = {"statics_id": st["statics_id"], "verdict": st["verdict"], "frame": st["frame"],
               "s_star": gated.get("s_star"), "beyond_plant": gated.get("beyond_plant"),
               "top_joints": gated.get("top_joints"), "group_util": gated.get("group_util"),
               "loads_n": {c: v.get("load_n") for c, v in (gated.get("contacts") or {}).items()},
               "limit_violations_deg": gated.get("limit_violations_deg"),
               "witness_passed": (st.get("witness") or {}).get("passed")}
    lib = rel.library_hold.get(hold_id, (None, {}))[1]
    e15500 = {k_: lib.get(k_) for k_ in ("tracked_share", "support_min", "err6_p50", "substitutions")} if lib else None
    # the exemplar frame's bodies (release .motion, 60 fps, COMMON order = MJCF order)
    mot = rel.motion(clip["stem"])
    f = int(hold["frame_hold"])
    pos = mot["rigid_body_pos"][f].double().numpy()
    rot = mot["rigid_body_rot"][f].double().numpy()
    b = {n: i for i, n in enumerate(rel.body_names)}
    com = (rel.mass[:, None] * (pos + _quat_apply_xyzw(rot, rel.ipos))).sum(0) / rel.mass.sum()
    lh, rh = pos[b["L_Hand"]], pos[b["R_Hand"]]
    mid = 0.5 * (lh + rh)
    line = rh - lh
    anchor = {"L_Wrist": pos[b["L_Wrist"]].round(4).tolist(), "L_Hand": lh.round(4).tolist(),
              "R_Wrist": pos[b["R_Wrist"]].round(4).tolist(), "R_Hand": rh.round(4).tolist(),
              "hands_mid": mid.round(4).tolist(), "hands_width_m": round(float(np.linalg.norm(line[:2])), 4),
              "hand_line_heading_deg": round(float(np.degrees(np.arctan2(line[1], line[0]))), 2),
              "com": com.round(4).tolist(), "com_minus_hands_mid_xy_m": round(float(np.linalg.norm((com - mid)[:2])), 4),
              "pelvis": pos[0].round(4).tolist()}
    fps = int(mot["fps"])
    return {
        "hold_id": hold_id, "stem": clip["stem"], "group": clip["group"], "name": hold["name"],
        "fps": fps, "frame_hold": f, "t_hold": round(f / fps, 4), "frame_start": int(hold["frame_start"]),
        "frame_end": int(hold["frame_end"]), "t_start": float(hold["t_start"]), "t_end": float(hold["t_end"]),
        "exemplar_moved": not hold_id.endswith(f"@{f}"),
        "node": int(seg["node"]), "node_key": node["key"], "orientation": hold["orientation"],
        "ground": ground, "braces": braces, "touch_allowed": touch_ok,
        "known_free": free, "known_free_share": share,
        "unknown_zones": sorted(set(ZONE_ORDER) - set(ground) - set(free)),
        "pose_role": hold["labels"]["pose_role"], "gate": hold["gate"]["decision"], "gate_flags": hold["gate"]["flags"],
        "statics": statics, "e15500": e15500, "anchor": anchor,
    }


# --------------------------------------------------------------------------- #
# Phases and events
# --------------------------------------------------------------------------- #
def config_of(ep: dict) -> dict:
    return {"ground": sorted(ep["ground"]), "braces": sorted(b["pair"] for b in ep["braces"])}


def events_between(a: dict, b: dict) -> dict:
    return {"ground_break": sorted(set(a["ground"]) - set(b["ground"])),
            "ground_make": sorted(set(b["ground"]) - set(a["ground"])),
            "braces_break": sorted(set(a["braces"]) - set(b["braces"])),
            "braces_make": sorted(set(b["braces"]) - set(a["braces"]))}


def _empty(ev: dict) -> bool:
    return not any(ev.values())


def build_edge(rel: Release, e: dict) -> dict:
    src, dst = endpoint(rel, e["src"]), endpoint(rel, e["dst"])
    cs, cd = config_of(src), config_of(dst)
    known = (set(src["ground"]) | set(src["known_free"])) & (set(dst["ground"]) | set(dst["known_free"]))
    phases, events = [], []
    prev, t_lo, t_hi = cs, 0.0, 0.0
    for i, p in enumerate(e["phases"]):
        cfg = {"ground": p["ground"], "braces": p["braces"]}
        ev = events_between(prev, cfg)
        if not _empty(ev):
            events.append({"at": "start" if i == 0 else f"{e['phases'][i - 1]['name']}->{p['name']}",
                           "phase_index": i, "t_range_s": [round(t_lo, 3), round(t_hi, 3)], **ev})
        free = sorted(z for z in ZONE_ORDER if z in known and z not in p["ground"])
        phases.append({**p, "free": free})
        t_lo += p["duration_s"][0]
        t_hi += p["duration_s"][1]
        prev = cfg
    end = events_between(prev, cd)
    net = events_between(cs, cd)
    edit = {**net, "orientation": f"{src['orientation']}->{dst['orientation']}"}
    return {"id": e["id"], "rank": e["rank"], "label": e["label"], "kind": e["kind"], "generator": e["generator"],
            "risk": e.get("risk"), "source": src, "destination": dst, "contact_edit": edit,
            "planted_anchor": sorted(set(src["ground"]) & set(dst["ground"])),
            "phases": phases, "events": events, "end_event": end,
            "duration_s": [round(t_lo, 3), round(t_hi, 3)]}


# --------------------------------------------------------------------------- #
# Evidence: the corpus timing behind the duration ranges, and §1.3's distances
# --------------------------------------------------------------------------- #
def corpus_timing(rel: Release) -> dict:
    """Inter-hold times of the x0 transitions the ranges lean on."""
    def pct(x):
        x = np.asarray(x, float)
        return {"n": int(len(x)), "p10": round(float(np.percentile(x, 10)), 3),
                "p50": round(float(np.percentile(x, 50)), 3), "p90": round(float(np.percentile(x, 90)), 3)} \
            if len(x) else {"n": 0}

    hands = {"L_HAND:G", "R_HAND:G"}
    inv = []
    for name, c in rel.x0.items():
        for a, b in zip(c["segments"], c["segments"][1:]):
            ga, gb = set(a["pairs"]), set(b["pairs"])
            if hands <= ga and hands <= gb and "inverted" in (a["orientation_bin"], b["orientation_bin"]):
                inv.append(float(b["t_start"]) - float(a["t_end"]))
    passages = {}
    for stem, pairs in (("220923_Pose_Dedicated_to_the_Sage_Koundinya_or_Eka_Pada_Koundinyanasana_I_and_II-a",
                         ((920, 1033), (1033, 1221))),
                        ("220923_Pose_Dedicated_to_the_Sage_Koundinya_or_Eka_Pada_Koundinyanasana_I_and_II-b",
                         ((578, 774), (774, 888), (888, 1060))),
                        ("220923_Side_Crane_Crow_Pose_or_Parsva_Bakasana_-b", ((931, 1046), (1046, 1230), (1230, 1493))),
                        ("220923_Handstand_pose_or_Adho_Mukha_Vrksasana_-a", ((718, 1193), (1193, 1632), (2025, 2314)))):
        segs = {int(s["frame_hold"]): s for s in rel.x0[stem]["segments"]}
        for a, b in pairs:
            sa, sb = segs[a], segs[b]
            passages[f"{stem.replace('220923_', '')}@{a}->@{b}"] = {
                "from": f"{sa['name']} {sorted(sa['pairs'])}", "to": f"{sb['name']} {sorted(sb['pairs'])}",
                "transition_s": round(float(sb["t_start"]) - float(sa["t_end"]), 3),
                "hold_to_hold_s": round(float(sb["t_hold"]) - float(sa["t_hold"]), 3),
                "dwell_to_s": round(float(sb["t_end"]) - float(sb["t_start"]), 3)}
    return {"hands_inverted_transition_s": pct(inv), "passages": passages}


def _census_name(hold_id: str) -> str:
    """``census.py``'s short endpoint name."""
    stem, frame = hold_id.split("@")
    return stem.replace("220923_", "").replace("220926_", "")[:40] + "@" + frame


def distances() -> dict:
    """§1.3's evidence: the census candidates (6-body distances, COM offsets) and the support-checked witnesses."""
    census = json.load(open(CENSUS))
    return {"cands": census["cands"], "witness_support_checked": json.load(open(WITNESS)),
            "hold_keys": {v: k for k, v in census["H"].items()}}


def census_row(dist: dict, e: dict) -> dict | None:
    """The census candidate with the same endpoints."""
    for c in dist["cands"]:
        if c["src"] == _census_name(e["src"]) and c["dst"] == _census_name(e["dst"]):
            return {k: c[k] for k in ("label", "d6_bestyaw", "d6_heading", "com_hands_off_src", "com_hands_off_dst",
                                      "pelvis_dz", "com_dz", "direct_edge")}
    return None


# --------------------------------------------------------------------------- #
# Build, check, write
# --------------------------------------------------------------------------- #
def build(rel: Release | None = None) -> dict:
    rel = rel or Release()
    edges = []
    dist = distances()
    for e in EDGES:
        spec = build_edge(rel, e)
        spec["census"] = census_row(dist, e)
        key = f"{dist['hold_keys'].get(e['src'])}->{dist['hold_keys'].get(e['dst'])}"
        spec["support_checked_witness"] = dist["witness_support_checked"].get(key)
        edges.append(spec)
    free = dict(FREE_EDGE)
    free["source"] = endpoint(rel, FREE_EDGE["src"])
    return {
        **ids.provenance(SCHEMA_VERSION, MODULE, __file__, rel.inputs),
        "kind": "edge_spec", "release_id": RELEASE_ID, "plant_sha256": rel.record["plant"]["plant_sha256"],
        "statics_id": rel.record["statics_id"], "labels_id": rel.record["labels_id"],
        "zone_order": list(ZONE_ORDER),
        "conventions": {
            "frames": "release v2 .motion frames at 60 fps (COMMON body order = MJCF order); t = frame / fps",
            "ground": "zones planted (supports, labels v2 pairs_ground)",
            "braces": "body-body contacts kept closed (labels v2 configured pairs)",
            "free": "zones that must stay off the floor in the phase (not planted, state known at both ends: planted "
                    "or the sidecar's known free); a zone in an event's make set may touch down inside that event's "
                    "window only",
            "touch_allowed": "body-body touches the reviewer allows at the endpoint (not braces; never charged)",
            "events": "contact changes between consecutive configurations; t_range_s = the event's admissible time "
                      "since the departure, from the phases' duration ranges",
            "planted_anchor": "zones planted at both ends: the generator keeps them fixed (T2 re-anchors D on them)",
        },
        "duration_bases": BASIS, "corpus_timing": corpus_timing(rel),
        "edges": edges, "free_edge": free,
    }


def check(spec: dict) -> list[str]:
    errors = []
    for e in spec["edges"]:
        exp = EXPECTED_EDITS[e["id"]]
        got = e["contact_edit"]
        for k, v in exp.items():
            if (sorted(got[k]) if isinstance(v, list) else got[k]) != (sorted(v) if isinstance(v, list) else v):
                errors.append(f"{e['id']}: {k} is {got[k]}, §1.3 says {v}")
        if not _empty(e["end_event"]):
            errors.append(f"{e['id']}: the last phase is not D's configuration: {e['end_event']}")
        for p in e["phases"]:
            both = set(p["ground"]) & set(p["free"])
            if both:
                errors.append(f"{e['id']}/{p['name']}: planted and free: {sorted(both)}")
            lo, hi = p["duration_s"]
            if not 0 < lo <= hi:
                errors.append(f"{e['id']}/{p['name']}: bad duration {p['duration_s']}")
        if not e["planted_anchor"]:
            errors.append(f"{e['id']}: no zone is planted at both ends")
        if e["source"]["unknown_zones"] or e["destination"]["unknown_zones"]:
            errors.append(f"{e['id']}: endpoint zones of unknown state {e['source']['unknown_zones']} / "
                          f"{e['destination']['unknown_zones']}")
    return errors


def table(spec: dict) -> str:
    rows = ["| id | edge | S -> D (node) | contact edit | phases (s) | d6 best-yaw | kind |", "|---|---|---|---|---|---|---|"]
    for e in spec["edges"]:
        s, d, c = e["source"], e["destination"], e["contact_edit"]
        edit = "; ".join(f"{k} {', '.join(v)}" for k, v in c.items() if k != "orientation" and v) or "ground unchanged"
        ph = " -> ".join(f"{p['name']} [{p['duration_s'][0]}, {p['duration_s'][1]}]" for p in e["phases"])
        d6 = e["census"]["d6_bestyaw"] if e["census"] else None
        rows.append(f"| {e['id']} | {e['label']} | @{s['frame_hold']} ({s['node']}) -> @{d['frame_hold']} ({d['node']}) "
                    f"| {edit}; {c['orientation']} | {ph} | {d6} | {e['kind']} |")
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="re-derive and compare with the written file")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    spec = build()
    errors = check(spec)
    for err in errors:
        print("FAILED", err, file=sys.stderr)
    if args.check:
        old = json.load(open(args.out))
        strip = lambda d: {k: v for k, v in d.items() if k not in ("generator", "inputs")}  # noqa: E731
        same = json.dumps(strip(old), sort_keys=True) == json.dumps(strip(json.loads(json.dumps(spec))), sort_keys=True)
        print("edges.json", "matches" if same else "DIFFERS from", "the re-derived spec")
        return 1 if errors or not same else 0
    if errors:
        return 1
    args.out.write_text(json.dumps(spec, indent=1) + "\n")
    print(table(spec))
    print(f"wrote {ids.display_path(args.out)}: {len(spec['edges'])} edges + the free edge; every hold id resolves in "
          f"{RELEASE_ID}; contact edits equal §1.3's")
    return 0


if __name__ == "__main__":
    sys.exit(main())
