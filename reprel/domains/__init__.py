"""Concrete domains. Importing this package registers them in ``reprel.core.DOMAINS``."""

from .boxworld import BoxWorldConfig, BoxWorldDomain
from .office import OfficeConfig, OfficeDomain
from .taxi import TaxiConfig, TaxiDomain

__all__ = [
    "BoxWorldConfig",
    "BoxWorldDomain",
    "OfficeConfig",
    "OfficeDomain",
    "TaxiConfig",
    "TaxiDomain",
]
