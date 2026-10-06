"""Concrete domains. Importing this package registers them in ``reprel.core.DOMAINS``."""

from .taxi import TaxiConfig, TaxiDomain

__all__ = ["TaxiConfig", "TaxiDomain"]
