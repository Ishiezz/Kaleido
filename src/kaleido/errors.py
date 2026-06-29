from __future__ import annotations


class KaleidoError(Exception):
    """Base for all Kaleido runtime errors."""


class FacetNotFoundError(KaleidoError):
    """Raised when a facet_id is absent from the registry."""

    def __init__(self, facet_id: str) -> None:
        super().__init__(f"Facet not found: {facet_id!r}")
        self.facet_id = facet_id


class RegistryNotLoadedError(KaleidoError):
    """Raised when the registry has not been seeded before use."""


class ScoringError(KaleidoError):
    """Raised when the scorer backend returns an unexpected response."""


class SynthesisError(KaleidoError):
    """Raised when facet contract synthesis fails validation."""


class EmbeddingError(KaleidoError):
    """Raised on failure to produce embeddings."""


class ConfigurationError(KaleidoError):
    """Raised when required configuration is missing or invalid."""
