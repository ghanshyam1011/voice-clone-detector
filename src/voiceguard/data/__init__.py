"""Manifest-driven data access. A manifest is the single source of truth for
what audio exists and how it is split -- built once from the official
protocol files, never from a directory walk or a home-made random split
(that leaks speakers and attack types across partitions).
"""

from voiceguard.data.manifests import (
    build_asvspoof_la,
    build_in_the_wild,
    load_manifest,
)

__all__ = ["build_asvspoof_la", "build_in_the_wild", "load_manifest"]
