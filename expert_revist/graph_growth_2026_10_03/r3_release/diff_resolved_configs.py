# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pre-launch checklist item 4 on release v3 (card R3): diff a ``CONFIG_ONLY`` run's ``resolved_configs.pt`` against
another run's, field by field, on the CPU.

Walks dataclasses, plain objects (``__dict__``), dicts and lists. A field missing from the reference pickle's ``__dict__`` is a *new* field (its
value and its class default are printed; ``getattr`` would return the default and hide it), as E6's diffs did.

    PYTHONPATH=. ../env_isaaclab/bin/python expert_revist/graph_growth_2026_10_03/r3_release/diff_resolved_configs.py \\
        results/<new>/resolved_configs.pt results/<reference>/resolved_configs.pt
"""

from __future__ import annotations

import dataclasses
import sys

import torch


def same(a, b) -> bool:
    """Leaf equality that never asks a tensor for its truth value."""
    if torch.is_tensor(a) or torch.is_tensor(b):
        return torch.is_tensor(a) and torch.is_tensor(b) and a.shape == b.shape and a.dtype == b.dtype and bool(
            torch.equal(a, b))
    try:
        return bool(a == b)
    except Exception:  # noqa: BLE001 -- an object whose == is not a bool
        return repr(a) == repr(b)


def walk(a, b, path: str, out: list) -> None:
    if dataclasses.is_dataclass(a) and dataclasses.is_dataclass(b) and type(a) is type(b):
        da, db = vars(a), vars(b)
        for k in sorted(set(da) | set(db)):
            if k not in db:
                default = next((f.default for f in dataclasses.fields(a) if f.name == k), None)
                out.append(("new", f"{path}.{k}", da[k], default))
            elif k not in da:
                out.append(("gone", f"{path}.{k}", None, db[k]))
            else:
                walk(da[k], db[k], f"{path}.{k}", out)
    elif (type(a) is type(b) and hasattr(a, "__dict__") and not callable(a) and not torch.is_tensor(a)
          and not isinstance(a, type)):
        walk(vars(a), vars(b), f"{path}<{type(a).__name__}>", out)            # plain objects: MdpComponent, ...
    elif isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k not in b or k not in a:
                out.append(("key", f"{path}[{k!r}]", a.get(k), b.get(k)))
            else:
                walk(a[k], b[k], f"{path}[{k!r}]", out)
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and len(a) == len(b) and any(
            dataclasses.is_dataclass(v) or isinstance(v, (dict, list, tuple)) for v in a):
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{path}[{i}]", out)
    elif not same(a, b):
        out.append(("value", path, a, b))


def main() -> int:
    new = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
    ref = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
    out: list = []
    walk(new, ref, "", out)
    for kind, path, a, b in out:
        sa, sb = repr(a), repr(b)
        print(f"{kind:5s} {path}: {sa[:300]}{'...' if len(sa) > 300 else ''}  <-  {sb[:300]}{'...' if len(sb) > 300 else ''}")
    print(f"{len(out)} differences")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
