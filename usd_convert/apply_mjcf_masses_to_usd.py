# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Restore MJCF geom densities onto a converted USD robot.

The Isaac Sim MJCF importer authors ``physics:density = 0.0`` on every rigid
body regardless of what the MJCF said.  Zero means "unspecified" to UsdPhysics,
so PhysX silently falls back to its 1000 kg/m^3 default and the articulation
simulates at the wrong mass -- for ``smpl_yogi03596_lowtorque`` that is 37.78 kg
instead of the 74.00 kg the asset was deliberately built for, which halves every
contact force in the simulation.

This script reads the source MJCF and writes the real density back onto each
matching rigid-body prim.  Because MuJoCo and UsdPhysics both compute mass and
inertia analytically from the same primitives (sphere / capsule / box), setting
density reproduces MuJoCo's mass *and* inertia tensor exactly -- which authoring
an explicit ``physics:mass`` would not, since that only rescales whatever
inertia PhysX already computed.

A body whose geoms disagree on density cannot be expressed as one body-level
density; those fall back to an explicit ``physics:mass`` and are reported.

Usage::

    python usd_convert/apply_mjcf_masses_to_usd.py \
      --mjcf data/assets/smpl/smpl_yogi03596_lowtorque_flat.xml \
      --usd  data/assets/smpl/smpl_yogi03596_lowtorque_usd

Verify afterwards by loading the robot and summing the simulated masses::

    env.simulator._robot.root_physx_view.get_masses()[0].sum()
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


DEFAULT_DENSITY = 1000.0  # MuJoCo's default when a geom omits both mass/density


def geom_volume(geom: ET.Element) -> float:
    """Analytic volume of a MuJoCo collision primitive, in m^3."""
    kind = geom.get("type", "sphere")
    if kind == "sphere":
        r = float(geom.get("size"))
        return 4 / 3 * math.pi * r**3
    if kind == "capsule":
        r = float(geom.get("size").split()[0])
        ends = [float(v) for v in geom.get("fromto").split()]
        length = math.dist(ends[:3], ends[3:])
        return math.pi * r * r * length + 4 / 3 * math.pi * r**3
    if kind == "box":
        s = [float(v) for v in geom.get("size").split()]
        return 8 * s[0] * s[1] * s[2]
    if kind == "cylinder":
        parts = geom.get("size").split()
        r = float(parts[0])
        if geom.get("fromto"):
            ends = [float(v) for v in geom.get("fromto").split()]
            length = math.dist(ends[:3], ends[3:])
        else:
            length = 2 * float(parts[1])
        return math.pi * r * r * length
    raise ValueError(f"unsupported geom type {kind!r} for mass computation")


def read_mjcf_bodies(mjcf_path: Path) -> dict[str, dict]:
    """Map body name -> {density, mass, uniform} from a flattened MJCF."""
    bodies: dict[str, dict] = {}
    for body in ET.parse(mjcf_path).getroot().iter("body"):
        name = body.get("name")
        if name is None:
            continue
        densities, mass = [], 0.0
        for geom in body.findall("geom"):
            volume = geom_volume(geom)
            if geom.get("mass") is not None:
                explicit = float(geom.get("mass"))
                density = explicit / volume if volume > 0 else 0.0
                mass += explicit
            else:
                density = float(geom.get("density", DEFAULT_DENSITY))
                mass += density * volume
            densities.append(density)
        if not densities:
            continue
        uniform = max(densities) - min(densities) <= 1e-6 * max(densities)
        bodies[name] = {
            "density": densities[0],
            "mass": mass,
            "uniform": uniform,
        }
    return bodies


def find_physics_layer(usd_arg: Path) -> Path:
    """Accept either the physics .usd itself or the converted package directory."""
    if usd_arg.is_file():
        return usd_arg
    candidates = sorted((usd_arg / "configuration").glob("*_physics.usd"))
    if len(candidates) != 1:
        raise SystemExit(
            f"expected exactly one *_physics.usd under {usd_arg / 'configuration'}, "
            f"found {[c.name for c in candidates]}"
        )
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mjcf", required=True, help="Source (flattened) MJCF.")
    parser.add_argument(
        "--usd",
        required=True,
        help="Converted USD package directory, or the *_physics.usd layer itself.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without writing.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip writing a .bak copy of the physics layer.",
    )
    args = parser.parse_args()

    from pxr import Usd, UsdPhysics

    mjcf_path = Path(args.mjcf)
    layer_path = find_physics_layer(Path(args.usd))
    bodies = read_mjcf_bodies(mjcf_path)
    if not bodies:
        raise SystemExit(f"no bodies with geoms found in {mjcf_path}")

    stage = Usd.Stage.Open(str(layer_path))
    rigid_bodies = {
        prim.GetName(): prim
        for prim in stage.Traverse()
        if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    }
    if not rigid_bodies:
        raise SystemExit(f"no rigid bodies found in {layer_path}")

    missing = sorted(set(bodies) - set(rigid_bodies))
    extra = sorted(set(rigid_bodies) - set(bodies))
    if missing:
        print(f"WARNING: MJCF bodies with no matching prim: {missing}", file=sys.stderr)
    if extra:
        print(f"WARNING: prims with no matching MJCF body: {extra}", file=sys.stderr)

    print(f"{'body':<12} {'was':>18}  {'now':>22}   {'mass (kg)':>9}")
    print("-" * 68)
    total, changed, by_mass = 0.0, 0, []
    for name, prim in rigid_bodies.items():
        spec = bodies.get(name)
        if spec is None:
            continue
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        density_attr = mass_api.CreateDensityAttr()
        mass_attr = mass_api.CreateMassAttr()
        before = (
            f"density={density_attr.Get() or 0.0:.1f}"
            if not mass_attr.HasAuthoredValue() or not mass_attr.Get()
            else f"mass={mass_attr.Get():.3f}"
        )

        if spec["uniform"]:
            after = f"density={spec['density']:.3f}"
            if not args.dry_run:
                density_attr.Set(spec["density"])
                mass_attr.Set(0.0)  # 0 = unset, so density governs
        else:
            after = f"mass={spec['mass']:.3f} (mixed densities)"
            by_mass.append(name)
            if not args.dry_run:
                mass_attr.Set(spec["mass"])
                density_attr.Set(0.0)

        total += spec["mass"]
        changed += 1
        print(f"{name:<12} {before:>18}  ->  {after:<22} {spec['mass']:9.3f}")

    print("-" * 68)
    print(f"{changed} bodies, total mass {total:.2f} kg")
    if by_mass:
        print(
            f"NOTE: {len(by_mass)} body/bodies had geoms of differing density and got an "
            f"explicit mass instead, so their inertia is PhysX's scaled by mass: {by_mass}"
        )

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return

    backup = layer_path.with_suffix(layer_path.suffix + ".bak")
    if not args.no_backup and not backup.exists():
        shutil.copy2(layer_path, backup)
        print(f"backup written to {backup}")
    stage.GetRootLayer().Save()
    print(f"updated {layer_path}")


if __name__ == "__main__":
    main()
