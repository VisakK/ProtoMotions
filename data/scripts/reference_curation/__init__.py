# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The offline reference reviewer: builds the curated dataset release the next expert trains on.

Built step by step from ``expert_revist/reference_curation_review_2026_09_28/BUILD_PLAN.MD``.
One module per step, pure functions plus a thin ``main()``; import with
``PYTHONPATH=.:data/scripts``.

``ids``      stem <-> MOYO recording id <-> file paths, hold ids, provenance records
``capture``  Step 1: per clip/frame/zone evidence from the MoSh markers, the shipped
             reference and the pressure mat
``audit``    Step 2: every hold of a manifest against that evidence: the fix list, the
             review queue and the progress metrics
``render``   Step 3: the evidence scene and its auto-framed views of any clip frame
             (bit-exact, on OSMesa)
``packets``  Step 3: queue items -> blind (Pass A) and informed (Pass B) review packets
             outside the repo, with a private index
``verdicts`` Step 4: the Pass-A answer contract (``prompts/``, ``schemas/``), its validator,
             the verdict ledger, the truth and the calibration table of each claim class
``review``   Step 4: the headless reviewer (``claude -p``, blind, resumable, budgeted)
             that fills the ledger
``human_mesh`` Step 5: the performer's registered MoSh SMPL-X mesh, its floor and self
             contacts per zone (capture store v2 = v1 + ``human_*``), and its render layer
``sources``  Step 6: capture store v3 (v2 + the seam-arbitrated floor contact of the limb
             zones) and the evidence hierarchy: mesh, then markers, then the mat
``labels``   Step 6: labels v1, reconciled deterministically from the capture (window,
             exemplar, ground set, Tier-1 load paths, roles), with the Pass-B claims attached
``informed`` Step 6: the informed review (Pass B): packets with the labels and measurements,
             the contract (``prompts/pass_b.md``, ``schemas/pass_b.json``) and its calibration
"""
