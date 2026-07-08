"""Platform-tier OpenRouter embedding config resolution.

Extracted out of ``_platform.py`` (not inlined like the pre-existing OPENAI
branch) so adding a second provider branch doesn't push that file over the
project's per-file line budget. Mirrors the OPENAI branch's config shape;
``init_platform_embedding`` in ``_platform.py`` owns the actual singleton
assignment + init-error list mutation, so this module stays a pure resolver
and every provider branch still funnels through one place that touches
that module-level state.
"""

from __future__ import annotations

import logging
import os

from common.constants import VECTOR_DIM
from common.embedding.constants import (
    OPENROUTER_EMBEDDING_BASE_URL,
    OPENROUTER_EMBEDDING_MODEL,
)
from common.embedding.protocols import EmbeddingProvider
from common.embedding.providers.openrouter import OpenRouterEmbeddingProvider

logger = logging.getLogger(__name__)

_ERROR_TAG = "openrouter-embedding-config"


def _parse_send_dimensions() -> tuple[bool, bool]:
    """Returns ``(value, had_error)``."""
    raw = (os.environ.get("PLATFORM_EMBEDDING_SEND_DIMENSIONS") or "false").lower()
    if raw not in ("true", "false"):
        logger.warning(
            "PLATFORM_EMBEDDING_SEND_DIMENSIONS=%r must be 'true' or 'false'", raw
        )
        return False, True
    return raw == "true", False


def _parse_truncate_to_dim() -> tuple[int, bool]:
    """Returns ``(value, had_error)``."""
    raw = os.environ.get("PLATFORM_EMBEDDING_TRUNCATE_TO_DIM")
    try:
        truncate_to_dim = int(raw) if raw else VECTOR_DIM
    except ValueError:
        logger.warning("PLATFORM_EMBEDDING_TRUNCATE_TO_DIM=%r must be an integer", raw)
        return 0, True
    if truncate_to_dim != VECTOR_DIM:
        logger.warning(
            "PLATFORM_EMBEDDING_TRUNCATE_TO_DIM=%r must equal VECTOR_DIM=%d",
            truncate_to_dim,
            VECTOR_DIM,
        )
        return 0, True
    return truncate_to_dim, False


def resolve_platform_openrouter_embedding() -> tuple[
    EmbeddingProvider | None, list[str]
]:
    """Build the OpenRouter platform singleton from ``PLATFORM_EMBEDDING_*`` env vars.

    Returns ``(provider, error_tags)``. A non-empty ``error_tags`` means the
    config was rejected — the caller must leave the singleton ``None`` so
    the worker fails loud (F-7) rather than silently embedding against the
    wrong endpoint.

    No dual ``base_url ⊕ send_dimensions`` misconfig guard here (unlike
    OPENAI): OpenRouter's endpoint shape is fixed and
    ``truncate_to_dim=VECTOR_DIM`` below already guarantees schema-correct
    output regardless of whether the routed model honours ``dimensions`` —
    see ``common.embedding._registry_openrouter`` for the same reasoning on
    the tenant-tier path.
    """
    api_key = os.environ.get("PLATFORM_EMBEDDING_API_KEY", "")
    if not api_key:
        logger.warning(
            "PLATFORM_EMBEDDING_PROVIDER=openrouter but no PLATFORM_EMBEDDING_API_KEY"
        )
        return None, [_ERROR_TAG]

    base_url = (
        os.environ.get("PLATFORM_EMBEDDING_BASE_URL") or OPENROUTER_EMBEDDING_BASE_URL
    )
    send_dimensions, error = _parse_send_dimensions()
    if error:
        return None, [_ERROR_TAG]
    truncate_to_dim, error = _parse_truncate_to_dim()
    if error:
        return None, [_ERROR_TAG]

    try:
        embed_model = (
            os.environ.get("PLATFORM_EMBEDDING_MODEL") or OPENROUTER_EMBEDDING_MODEL
        )
        provider = OpenRouterEmbeddingProvider(
            api_key=api_key,
            model=embed_model,
            base_url=base_url,
            send_dimensions=send_dimensions,
            truncate_to_dim=truncate_to_dim,
        )
        logger.info(
            "Platform embedding: openrouter/%s (base_url=%s, send_dimensions=%s)",
            embed_model,
            base_url,
            send_dimensions,
        )
        return provider, []
    except Exception:
        logger.exception("Failed to initialize platform OpenRouter embedding provider")
        return None, ["openrouter-embedding"]
