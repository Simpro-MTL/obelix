"""Tenant-tier OpenRouter embedding provider resolution.

Extracted out of ``_registry.py`` (not inlined like the pre-existing OPENAI
branch) to keep that already-large file from growing further. Mirrors the
OPENAI branch's shape — LRU cache, tenant→platform→fake fallback tier,
env-driven config — see ``_get_or_create_openai_provider`` and the OPENAI
branch of ``get_embedding_provider`` in ``_registry.py`` for the shared
rationale.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections import OrderedDict

from common.constants import VECTOR_DIM
from common.embedding.constants import (
    OPENROUTER_EMBEDDING_BASE_URL,
    OPENROUTER_EMBEDDING_MODEL,
)
from common.embedding.protocols import EmbeddingProvider
from common.embedding.providers.fake import FakeEmbeddingProvider
from common.embedding.providers.openrouter import OpenRouterEmbeddingProvider

logger = logging.getLogger(__name__)

# Separate cache/namespace from ``_openai_provider_cache`` in ``_registry.py``
# so an OpenRouter api_key can never alias an OpenAI client. Own
# background-task set too — see ``_get_or_create_openai_provider`` in
# ``_registry.py`` for the full eviction rationale this mirrors.
_OPENROUTER_CACHE_MAX = 256
_openrouter_provider_cache: OrderedDict[
    tuple[str, str, str, bool, str | None, int | None],
    OpenRouterEmbeddingProvider,
] = OrderedDict()
_background_tasks: set[asyncio.Task[None]] = set()


def _get_or_create_openrouter_provider(
    api_key: str,
    model: str,
    base_url: str,
    send_dimensions: bool,
    query_instruction: str | None,
    truncate_to_dim: int | None,
) -> OpenRouterEmbeddingProvider:
    """LRU-bounded ``OpenRouterEmbeddingProvider`` lookup keyed on the full
    client config tuple. Same eviction + background-``aclose()`` logic as
    ``_get_or_create_openai_provider`` in ``_registry.py``.
    """
    cache_key = (
        api_key,
        model,
        base_url,
        send_dimensions,
        query_instruction,
        truncate_to_dim,
    )
    cached = _openrouter_provider_cache.get(cache_key)
    if cached is not None:
        _openrouter_provider_cache.move_to_end(cache_key)
        return cached
    provider = OpenRouterEmbeddingProvider(
        api_key=api_key,
        model=model,
        base_url=base_url,
        send_dimensions=send_dimensions,
        query_instruction=query_instruction,
        truncate_to_dim=truncate_to_dim,
    )
    _openrouter_provider_cache[cache_key] = provider
    if len(_openrouter_provider_cache) > _OPENROUTER_CACHE_MAX:
        _, evicted = _openrouter_provider_cache.popitem(last=False)
        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(evicted.aclose())
            _background_tasks.add(task)
            task.add_done_callback(_background_tasks.discard)
        except RuntimeError:
            pass
    return provider


def _resolve_openrouter_api_key(tenant_config: object | None) -> str:
    """Tenant override first, then ``OPENROUTER_API_KEY`` env. Empty string if neither.

    ``OPENROUTER_API_KEY`` is already bridged into ``os.environ`` for the
    LLM path (``core_api.config.bridge_credentials_to_environ``), so no
    additional bridge wiring is needed for the embedding registry to read it.
    """
    if tenant_config is not None:
        key = getattr(tenant_config, "openrouter_api_key", None)
        if key:
            return key
    return os.environ.get("OPENROUTER_API_KEY", "")


def _openrouter_fallback() -> EmbeddingProvider:
    """No usable key → platform singleton, else ``FakeEmbeddingProvider`` (F-2).

    Tenant-agnostic: the platform singleton is process-wide, not scoped to
    any particular tenant, so there is no ``tenant_config`` to read here.
    """
    from common.embedding._platform import get_platform_embedding

    platform = get_platform_embedding()
    if platform is not None:
        logger.info(
            "No tenant key for OpenRouter embedding, using platform embedding (%s)",
            platform.model,
        )
        return platform
    logger.warning(
        "No API key for OpenRouter embedding provider, returning FakeEmbeddingProvider",
    )
    return FakeEmbeddingProvider()


def _parse_openrouter_send_dimensions() -> bool:
    """Strict bool parse for ``OPENROUTER_EMBEDDING_SEND_DIMENSIONS``.

    ``or "false"`` (not ``os.environ.get(K, "false")``) so an empty-string
    value — the docker-compose unset-var shape (``"${VAR:-}"`` → ``""``) —
    takes the same safe default as a genuinely unset var (F-5), rather than
    reaching the ``not in ("true", "false")`` branch and raising. A
    non-canonical non-empty value ("yes", "1") still raises (F-6).
    """
    raw = (os.environ.get("OPENROUTER_EMBEDDING_SEND_DIMENSIONS") or "false").lower()
    if raw not in ("true", "false"):
        raise ValueError(
            f"OPENROUTER_EMBEDDING_SEND_DIMENSIONS={raw!r} must be 'true' or 'false'."
        )
    return raw == "true"


def _parse_openrouter_truncate_to_dim() -> int:
    """Parse + validate ``OPENROUTER_EMBEDDING_TRUNCATE_TO_DIM``.

    Defaults to ``VECTOR_DIM`` (unlike OpenAI's default-``None``):
    OpenRouter doesn't document a ``dimensions`` request param, so
    client-side truncation is the only deterministic way to guarantee
    schema-dim output regardless of the routed model's native width (U-3).
    Only ``VECTOR_DIM`` itself is a valid override value — anything else
    would either under- or over-fill the pgvector column.
    """
    raw = os.environ.get("OPENROUTER_EMBEDDING_TRUNCATE_TO_DIM")
    try:
        truncate_to_dim = int(raw) if raw else VECTOR_DIM
    except ValueError:
        raise ValueError(
            f"OPENROUTER_EMBEDDING_TRUNCATE_TO_DIM={raw!r} must be an integer"
        ) from None
    if truncate_to_dim != VECTOR_DIM:
        raise ValueError(
            f"OPENROUTER_EMBEDDING_TRUNCATE_TO_DIM={truncate_to_dim} must equal "
            f"VECTOR_DIM={VECTOR_DIM}; this knob is only for truncating a "
            "wider model's native output to the schema dimension."
        )
    return truncate_to_dim


def resolve_openrouter_provider(tenant_config: object | None) -> EmbeddingProvider:
    """Resolve (or construct) the OpenRouter provider for ``get_embedding_provider``.

    Called from the ``ProviderName.OPENROUTER`` branch in
    ``_registry.get_embedding_provider``.

    Deliberately does NOT reproduce the OPENAI branch's dual
    ``base_url ⊕ send_dimensions`` misconfig guards: those exist because a
    self-hosted OpenAI-compatible endpoint may reject or require the
    ``dimensions=`` kwarg depending on whether it's hosted OpenAI or
    TEI/vLLM. OpenRouter has one fixed endpoint shape and the
    ``truncate_to_dim=VECTOR_DIM`` default already guarantees schema-correct
    output regardless of whether the routed model honours ``dimensions`` —
    there's no failure-guaranteeing combination left to guard against.
    """
    api_key = _resolve_openrouter_api_key(tenant_config)
    if not api_key:
        return _openrouter_fallback()

    embed_model = (
        getattr(tenant_config, "embedding_model", None)
        if tenant_config is not None
        else None
    ) or OPENROUTER_EMBEDDING_MODEL
    base_url = (
        os.environ.get("OPENROUTER_EMBEDDING_BASE_URL") or OPENROUTER_EMBEDDING_BASE_URL
    )
    send_dimensions = _parse_openrouter_send_dimensions()
    query_instruction = os.environ.get("EMBEDDING_QUERY_INSTRUCTION") or None
    truncate_to_dim = _parse_openrouter_truncate_to_dim()

    return _get_or_create_openrouter_provider(
        api_key,
        embed_model,
        base_url,
        send_dimensions,
        query_instruction,
        truncate_to_dim,
    )
