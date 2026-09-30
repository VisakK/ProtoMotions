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
``statics``  Step 7: the gated static LP (contacts only inside their bands, typed statuses, the
             min/max-load necessity test, joint stops, the counterfactual closure) over every hold
             of a labels folder, written to ``data/reference_curation/statics/``
``witness``  Step 7: the MuJoCo statue witness: does the plant, simulated, hold the pose on the
             contacts the gated LP balanced it on
``plant``    Step 8 (Step 7's open checks): the training plant from recorded PhysX rollouts: its
             exp-map joint coordinates, hard limits, the references outside them, and the gated
             LP against PhysX's measured contact forces
``retarget`` Step 8: the contact-constrained retarget: every frame of a clip re-solved on the
             plant's own coordinates so the human's supports rest flat in the floor band, the
             human's body-body contacts close, nothing the plant collides overlaps and every
             joint stays in its box; all ``.motion`` fields regenerated, with lineage
``edits``    Step 8: Pass C, the reviewer's blind before/after check of the retarget's edits
             over the performer's mesh, with its machine-truth controls and calibration
``roles``    Step 8: labels v1.1 = labels v1 with the statics of the retargeted references
             merged in (``required_touch`` ground supports statics requires -> ``required_support``)
``isaac_statue`` Step 8 (Step 7's open checks): the statue test in IsaacLab (a script, run as its
             own process; not a usable witness yet, see the card)
"""
