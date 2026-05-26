"""Offline master-VLAD-codebook exporter for DL-VINS loop_fusion.

The C++ ``loop_fusion`` package fits a VLAD codebook **online** during a run
(seeded from a master codebook if one is provided, then refit at
KF=``VLAD_WARMUP_KFS``). This package trains that master from one or more
held-out VPR benchmark image roots (AnyLoc's ``custom_datasets/`` and
``datasets_vg/``) — never the DL-VINS evaluation datasets, to keep
evaluation honest.

Entry point: ``python -m vpr.export_codebook --image-roots ... --extractors ...``
"""

__all__ = ["config"]
