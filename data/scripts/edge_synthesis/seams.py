"""Where a synthesised variant meets the real clips R2 splices it into: card T6's seam group (PLAN.MD §1.4, "What the
Step 3 build must handle"). ``evidence/seam_offsets.py``'s measure lives here now (``legacy``).

The group, on the exported ``.motion`` alone (60 fps, plant v2; S and D from ``edges.json``):

==========  ==========================================================================================================
check       rule
==========  ==========================================================================================================
start       the first frame against S's exemplar frame in S's own clip frame (the export does not move it): every body
            within ``START_ALL_CM``, every body of S's planted zones (the hands) within ``START_PLANTED_CM``
end         the last frame's planted bodies (the bodies of D's ground zones) against D's exemplar placed by the
            generator's hand-anchor transform (``sketch.hand_anchor_transform``) within ``END_PLANTED_CM``
drift       a zone planted at S and at D (the hands) moves at most ``DRIFT_CM`` horizontally from the first frame, on
            every frame; a zone that makes during the variant (a landing's feet) at most ``DRIFT_CM`` from its
            position ``IMPACT_S`` after its touchdown (its lowest collider point first within ``TOUCH_M`` of the
            floor), to the end
==========  ==========================================================================================================

**Why ``end`` can fail by construction** (measured 2026-10-04 for T6). The exemplars of crow, the handstand, the plank
and chaturanga place the palms differently: after the hand-anchor transform the knuckle bodies (``L_Hand``,
``R_Hand``) meet within 0.4-1.0 cm, but each palm (``L_Wrist``/``R_Wrist``, whose origin is the wrist joint at the
heel of the palm) is turned 8-14 deg about the vertical, so the wrist origins sit 1.8-3.0 cm apart (E1/E3 2.9 / 2.7,
B1 1.8 / 2.2, E2 2.5 / 3.0; E5 0.5 / 0.3). No rigid placement of D removes a mirror-symmetric yaw of the two palms
(a four-body planar Procrustes leaves the same), and a loaded palm cannot turn 12 deg without its heel sliding about
3 cm, so a variant whose hands keep S's pose (T6's rule) meets D's palms only to within the exemplars' own
difference. ``end`` reports that literally; ``end_attainable`` passes a body whose end offset is the exemplars' own
(the variant within ``START_PLANTED_CM`` of S's pose for that body, and S's exemplar beyond ``END_PLANTED_CM`` from
D's) and records it in ``end_exemplar_bodies``. Which of the two R2 needs is the user's decision.
"""

from __future__ import annotations

import numpy as np

from extract_contact_configs import ZONE_ORDER, ZONES
from edge_synthesis import sketch as SK

START_ALL_CM = 1.0
START_PLANTED_CM = 0.5
END_PLANTED_CM = 1.5
DRIFT_CM = 0.5
TOUCH_M = 0.01             # the motion files' geometric contact rule
IMPACT_S = 0.1             # admit.IMPACT_S
NAMES = ("Pelvis L_Hip L_Knee L_Ankle L_Toe R_Hip R_Knee R_Ankle R_Toe Torso Spine Chest Neck Head L_Thorax "
         "L_Shoulder L_Elbow L_Wrist L_Hand R_Thorax R_Shoulder R_Elbow R_Wrist R_Hand").split()
BI = {n: i for i, n in enumerate(NAMES)}
HANDS = [BI["L_Hand"], BI["R_Hand"]]
FEET = [BI["L_Toe"], BI["R_Toe"], BI["L_Ankle"], BI["R_Ankle"]]
LANDING_DESTINATIONS = ("Plank_Pose", "Four-Limbed_Staff")


def endpoints(e: dict) -> dict:
    """S's exemplar (its clip frame) and D's placed by the hand-anchor transform: ``{sp, sr, dp, dr}`` ([24, 3],
    [24, 4] xyzw)."""
    from edge_synthesis import plant_mj as pm

    s, d = e["source"], e["destination"]
    sm = pm.load_motion(s["stem"])
    sp = sm["rigid_body_pos"][s["frame_hold"]].double().numpy()
    sr = sm["rigid_body_rot"][s["frame_hold"]].double().numpy()
    dm = pm.load_motion(d["stem"])
    dp = dm["rigid_body_pos"][d["frame_hold"]].double().numpy()
    dr = dm["rigid_body_rot"][d["frame_hold"]].double().numpy()
    yaw, t = SK.hand_anchor_transform(sp, dp, BI)
    dp2, dr2 = SK.apply_planar(dp, dr, yaw, t)
    return {"sp": sp, "sr": sr, "dp": dp2, "dr": dr2, "yaw_deg": float(np.degrees(yaw))}


def legacy(motion: dict, e: dict) -> dict:
    """``evidence/seam_offsets.py``'s measure (the numbers in PLAN.MD's tables before T6): the hands (``L_Hand``,
    ``R_Hand``) at both ends, the four foot bodies at the end of a landing, the hand widths, the hands' drift."""
    ep = endpoints(e)
    sp, dp2 = ep["sp"], ep["dp"]
    pos = motion["rigid_body_pos"].double().numpy()
    landing = any(k in e["destination"]["stem"] for k in LANDING_DESTINATIONS)

    def width(p):
        return round(float(np.linalg.norm(p[BI["L_Hand"], :2] - p[BI["R_Hand"], :2])), 4)
    from edge_synthesis import plant_mj as pm
    dp_raw = pm.load_motion(e["destination"]["stem"])["rigid_body_pos"][e["destination"]["frame_hold"]].double().numpy()
    return {"hands_start_cm": round(100 * float(np.linalg.norm(pos[0, HANDS] - sp[HANDS], axis=1).max()), 2),
            "hands_end_cm": round(100 * float(np.linalg.norm(pos[-1, HANDS] - dp2[HANDS], axis=1).max()), 2),
            "feet_end_cm": (round(100 * float(np.linalg.norm(pos[-1, FEET] - dp2[FEET], axis=1).max()), 2)
                            if landing else None),
            "hand_width_m": {"synthetic": width(pos[0]), "S": width(sp), "D": width(dp_raw)},
            "hand_drift_cm": round(100 * float(np.linalg.norm(pos[:, HANDS, :2] - pos[:1, HANDS, :2], axis=2).max()), 2)}


def zone_lowest(motion: dict) -> np.ndarray:
    """``[T, 15]`` every zone's lowest collider point above the floor (plant v2's colliders)."""
    from scipy.spatial.transform import Rotation

    from edge_synthesis import costs as C
    from reference_curation import mosh_replay as mr

    sk = mr.skeleton_for(mr.V2_XML)
    pos = motion["rigid_body_pos"].double().numpy()
    rot = motion["rigid_body_rot"].double().numpy()
    R = Rotation.from_quat(rot.reshape(-1, 4)).as_matrix().reshape(rot.shape[:2] + (3, 3))
    return C.zone_lowest(sk, pos, R)[0]


def check(motion: dict, e: dict) -> dict:
    """The seam group of one exported variant (``motion``: the ``.motion`` dict; ``e``: its edge in edges.json)."""
    ep = endpoints(e)
    sp, dp2 = ep["sp"], ep["dp"]
    pos = motion["rigid_body_pos"].double().numpy()
    fps = int(motion["fps"])
    src_g, dst_g = list(e["source"]["ground"]), list(e["destination"]["ground"])
    start_planted = [BI[b] for z in src_g for b in ZONES[z]]
    end_planted = [BI[b] for z in dst_g for b in ZONES[z]]
    # start
    d_all = np.linalg.norm(pos[0] - sp, axis=-1)
    start = {"all_max_cm": round(100 * float(d_all.max()), 3), "all_worst": NAMES[int(d_all.argmax())],
             "planted_max_cm": round(100 * float(d_all[start_planted].max()), 3),
             "planted_worst": NAMES[start_planted[int(d_all[start_planted].argmax())]]}
    start["pass"] = start["all_max_cm"] <= START_ALL_CM and start["planted_max_cm"] <= START_PLANTED_CM
    # end: literal, and the bodies whose offset is the exemplars' own
    d_end = np.linalg.norm(pos[-1, end_planted] - dp2[end_planted], axis=-1)
    d_ex = np.linalg.norm(sp[end_planted] - dp2[end_planted], axis=-1)            # S's exemplar vs D's, per body
    d_to_s = np.linalg.norm(pos[-1, end_planted] - sp[end_planted], axis=-1)
    by_body = {NAMES[b]: {"to_D_cm": round(100 * float(d_end[i]), 3),
                          "exemplars_S_to_D_cm": round(100 * float(d_ex[i]), 3) if b in start_planted else None,
                          "to_S_cm": round(100 * float(d_to_s[i]), 3) if b in start_planted else None}
               for i, b in enumerate(end_planted)}
    beyond = [NAMES[b] for i, b in enumerate(end_planted) if d_end[i] > END_PLANTED_CM / 100]
    exemplar = [NAMES[b] for i, b in enumerate(end_planted) if d_end[i] > END_PLANTED_CM / 100 and b in start_planted
                and d_to_s[i] <= START_PLANTED_CM / 100 and d_ex[i] > END_PLANTED_CM / 100]
    end = {"planted_max_cm": round(100 * float(d_end.max()), 3), "planted_worst": NAMES[end_planted[int(d_end.argmax())]],
           "beyond": beyond, "by_body": by_body, "pass": not beyond,
           "end_exemplar_bodies": exemplar, "pass_attainable": not (set(beyond) - set(exemplar))}
    if beyond:
        end["reason"] = ("the exemplars' own: these bodies keep S's pose and S's exemplar is beyond the tolerance "
                         "from D's" if not set(beyond) - set(exemplar) else
                         "generator: " + ", ".join(sorted(set(beyond) - set(exemplar))) + " end off D's exemplar")
    # drift
    low = zone_lowest(motion)
    rows = {}
    for z in sorted(set(src_g) | set(dst_g)):
        if z not in dst_g:
            continue
        bs = [BI[b] for b in ZONES[z]]
        if z in src_g:
            f0, ref = 0, pos[0, bs, :2]
            kind = "planted throughout"
        else:
            zi = ZONE_ORDER.index(z)
            touch = np.nonzero(low[:, zi] < TOUCH_M)[0]
            if not len(touch):
                rows[z] = {"kind": "makes", "touchdown_frame": None, "max_cm": None}
                continue
            f0 = min(int(touch[0]) + int(round(IMPACT_S * fps)), len(pos) - 1)
            ref = pos[f0, bs, :2]
            kind = "makes"
        dd = np.linalg.norm(pos[f0:, bs, :2] - ref[None], axis=-1)
        rows[z] = {"kind": kind, "from_frame": f0, "max_cm": round(100 * float(dd.max()), 3),
                   "worst": NAMES[bs[int(np.unravel_index(dd.argmax(), dd.shape)[1])]]}
        if kind == "makes":
            rows[z]["touchdown_frame"] = int(f0 - int(round(IMPACT_S * fps)))
    worst = max((r["max_cm"] for r in rows.values() if r.get("max_cm") is not None), default=0.0)
    missing = [z for z, r in rows.items() if r.get("max_cm") is None]
    drift = {"zones": rows, "max_cm": worst, "never_touched": missing, "pass": worst <= DRIFT_CM and not missing}
    out = {"start": start, "end": end, "drift": drift,
           "thresholds_cm": {"start_all": START_ALL_CM, "start_planted": START_PLANTED_CM,
                             "end_planted": END_PLANTED_CM, "drift": DRIFT_CM},
           "pass": start["pass"] and end["pass"] and drift["pass"],
           "pass_attainable": start["pass"] and end["pass_attainable"] and drift["pass"]}
    out["failed"] = [k for k in ("start", "end", "drift") if not out[k]["pass"]]
    return out
