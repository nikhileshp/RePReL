"""Concrete domains. Importing this package registers them in ``reprel.core.DOMAINS``."""

from .office import OfficeConfig, OfficeDomain
from .taxi import TaxiConfig, TaxiDomain

__all__ = ["OfficeConfig", "OfficeDomain", "TaxiConfig", "TaxiDomain"]
