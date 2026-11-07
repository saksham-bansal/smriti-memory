"""Cross-session ingestion, temporal normalization, atomic claim extraction, and offline Dream consolidation."""
from smriti.ingestion.dreamer import DreamEngine
from smriti.ingestion.extractor import SessionIngestor
from smriti.ingestion.temporal import TemporalNormalizer

__all__ = ["TemporalNormalizer", "SessionIngestor", "DreamEngine"]
