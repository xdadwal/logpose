"""Fixture package for tool discovery.

Re-exports ``alpha`` from :mod:`.first` on purpose: the same ``ToolDef`` object then
appears in two module namespaces, which is the identity-dedupe case.
"""

from tests.fixtures_discovery.first import alpha

__all__ = ["alpha"]
