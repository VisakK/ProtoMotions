# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Package a chosen set of ``.motion`` clips into a training ``.pt``.

``MotionLib`` can already package a whole *directory*, but directory order is
whatever the filesystem returns and there is no way to pick a subset.  This
writes an explicit ``.yaml`` manifest (kept next to the ``.pt`` as the record of
what went in, and reusable as a ``--motion-file`` in its own right), packages it,
and then verifies the result frame-by-frame against the source clips.

The verification matters for the contact-matching reward: the reward reads
``motion_lib.contacts``, so a packaging step that dropped or reordered the
per-frame labels would silently train against the wrong contact targets.

Usage::

    PYTHONPATH=. python data/scripts/package_motion_subset.py \
      --out data/smpl/yoga_yogi_crow_pair_v2_contacts.pt \
      data/smpl/yoga_yogi_balance_subset_v2_contacts/220923_Crane_Crow_Pose_or_Bakasana_-a.motion \
      data/smpl/yoga_yogi_balance_subset_v2_contacts/220923_Side_Crane_Crow_Pose_or_Parsva_Bakasana_-a.motion
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("motions", nargs="+", help="Paths to the .motion files, in order.")
    parser.add_argument("--out", type=str, required=True, help="Output .pt path.")
    parser.add_argument(
        "--weights",
        type=float,
        nargs="*",
        default=None,
        help="Sampling weight per motion (default: 1.0 each).",
    )
    parser.add_argument(
        "--yaml",
        type=str,
        default=None,
        help="Manifest path (default: the .pt path with a .yaml suffix).",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing .pt.")
    return parser


def write_manifest(manifest: Path, motions: list[Path], weights: list[float]) -> None:
    """MotionLib resolves manifest entries relative to the manifest's directory."""
    manifest.parent.mkdir(parents=True, exist_ok=True)
    lines = ["motions:"]
    for path, weight in zip(motions, weights):
        rel = os.path.relpath(path.resolve(), manifest.parent.resolve())
        lines.append(f"  - file: {rel}")
        lines.append(f"    weight: {weight}")
    manifest.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = create_parser().parse_args()
    motions = [Path(m) for m in args.motions]
    for path in motions:
        if not path.is_file() or path.suffix != ".motion":
            raise SystemExit(f"not a .motion file: {path}")
    weights = args.weights or [1.0] * len(motions)
    if len(weights) != len(motions):
        raise SystemExit(f"got {len(weights)} weights for {len(motions)} motions")

    out = Path(args.out)
    if out.suffix != ".pt":
        raise SystemExit("--out must end in .pt")
    if out.exists() and not args.force:
        raise SystemExit(f"{out} exists; pass --force to overwrite")

    manifest = Path(args.yaml) if args.yaml else out.with_suffix(".yaml")
    write_manifest(manifest, motions, weights)
    print(f"manifest: {manifest}")

    from protomotions.components.motion_lib import MotionLib, MotionLibConfig

    lib = MotionLib(config=MotionLibConfig(motion_file=str(manifest)), device=args.device)
    lib.save_to_file(out)

    # Verify against the sources rather than trusting the packer.
    packaged = torch.load(out, map_location="cpu", weights_only=False)
    assert len(packaged["motion_files"]) == len(motions), "motion count mismatch"
    starts = packaged["length_starts"]
    frames = packaged["motion_num_frames"]
    print(f"\npackaged {len(motions)} motions, {int(frames.sum())} frames total")
    ok = True
    for i, path in enumerate(motions):
        source = torch.load(path, map_location="cpu", weights_only=False)
        lo = int(starts[i])
        hi = lo + int(frames[i])
        n = source["rigid_body_pos"].shape[0]
        checks = {
            "frame count": n == int(frames[i]),
            "order": Path(packaged["motion_files"][i]).name == path.name,
            "rigid_body_pos": torch.equal(packaged["gts"][lo:hi], source["rigid_body_pos"]),
            "rigid_body_rot": torch.equal(packaged["grs"][lo:hi], source["rigid_body_rot"]),
            "dof_pos": torch.equal(packaged["dps"][lo:hi], source["dof_pos"]),
        }
        if "contacts" in packaged and "rigid_body_contacts" in source:
            checks["contacts"] = torch.equal(
                packaged["contacts"][lo:hi].bool(), source["rigid_body_contacts"].bool()
            )
        failed = [k for k, v in checks.items() if not v]
        ok &= not failed
        contact_frac = (
            float(packaged["contacts"][lo:hi].float().mean())
            if "contacts" in packaged
            else float("nan")
        )
        print(
            f"  [{i}] {path.name}\n"
            f"      {n} frames, {float(packaged['motion_lengths'][i]):.2f}s, "
            f"weight {float(packaged['motion_weights'][i]):.2f}, "
            f"contact labels on {100 * contact_frac:.1f}% of body-frames"
            + (f"\n      FAILED: {failed}" if failed else "")
        )
    if not ok:
        raise SystemExit("verification failed")
    print(f"\nwrote {out} (verified against sources)")


if __name__ == "__main__":
    main()
