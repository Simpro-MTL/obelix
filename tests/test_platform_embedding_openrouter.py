"""Platform-tier tests for ``PLATFORM_EMBEDDING_PROVIDER=openrouter``.

New file rather than an extension of ``tests/test_platform_providers.py``:
that file is already 650+ lines (pre-existing, grandfathered) and adding
another provider's worth of cases there would push it further past the
project's test-file size budget. Mirrors its
``TestPlatformEmbeddingSelfHosted`` style and its module-level singleton
reset fixture, scoped to just the embedding singleton this file touches.

Calls ``common.embedding._platform.init_platform_embedding()`` directly —
the exact entry point core-worker's lifespan uses — rather than the
core-api ``init_platform_providers`` wrapper.
"""

from __future__ import annotations

import pytest


def _reset_platform_embedding() -> None:
    import common.embedding._platform as embedding_mod

    embedding_mod._platform_embedding = None
    embedding_mod._platform_init_errors.clear()


@pytest.fixture(autouse=True)
def _clean_platform_embedding():
    """Ensure the platform embedding singleton is reset before and after
    each test — a leftover ``"openrouter-embedding-config"`` init-error
    entry would otherwise leak into unrelated ``/status`` health tests."""
    _reset_platform_embedding()
    yield
    _reset_platform_embedding()


@pytest.mark.unit
def test_platform_openrouter_builds_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """``PLATFORM_EMBEDDING_PROVIDER=openrouter`` + a key builds a real
    ``OpenRouterEmbeddingProvider`` singleton with no init errors."""
    monkeypatch.setenv("PLATFORM_EMBEDDING_PROVIDER", "openrouter")
    monkeypatch.setenv("PLATFORM_EMBEDDING_API_KEY", "sk-or-platform")
    monkeypatch.setenv("PLATFORM_EMBEDDING_MODEL", "openai/text-embedding-3-large")

    from common.embedding._platform import (
        get_platform_embedding,
        get_platform_init_errors,
        init_platform_embedding,
    )
    from common.embedding.providers.openrouter import OpenRouterEmbeddingProvider

    init_platform_embedding()
    emb = get_platform_embedding()
    assert isinstance(emb, OpenRouterEmbeddingProvider)
    assert emb.model == "openai/text-embedding-3-large"
    assert get_platform_init_errors() == []


@pytest.mark.unit
def test_platform_openrouter_missing_key_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F-7: ``PLATFORM_EMBEDDING_PROVIDER=openrouter`` with no
    ``PLATFORM_EMBEDDING_API_KEY`` must reject the config — singleton
    stays ``None`` and ``"openrouter-embedding-config"`` is recorded so
    the worker fails loud (visible on ``/status`` as degraded) instead of
    silently embedding against the wrong endpoint."""
    monkeypatch.setenv("PLATFORM_EMBEDDING_PROVIDER", "openrouter")
    monkeypatch.delenv("PLATFORM_EMBEDDING_API_KEY", raising=False)

    from common.embedding._platform import (
        get_platform_embedding,
        get_platform_init_errors,
        init_platform_embedding,
    )

    init_platform_embedding()
    assert get_platform_embedding() is None
    assert "openrouter-embedding-config" in get_platform_init_errors()


@pytest.mark.unit
def test_platform_openrouter_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """No ``PLATFORM_EMBEDDING_BASE_URL`` / ``SEND_DIMENSIONS`` /
    ``TRUNCATE_TO_DIM`` set → OpenRouter's own defaults apply: the
    OpenRouter endpoint, ``send_dimensions=False``, ``truncate_to_dim=
    VECTOR_DIM``."""
    from common.constants import VECTOR_DIM
    from common.embedding.constants import OPENROUTER_EMBEDDING_BASE_URL

    monkeypatch.setenv("PLATFORM_EMBEDDING_PROVIDER", "openrouter")
    monkeypatch.setenv("PLATFORM_EMBEDDING_API_KEY", "sk-or-platform")
    monkeypatch.delenv("PLATFORM_EMBEDDING_BASE_URL", raising=False)
    monkeypatch.delenv("PLATFORM_EMBEDDING_SEND_DIMENSIONS", raising=False)
    monkeypatch.delenv("PLATFORM_EMBEDDING_TRUNCATE_TO_DIM", raising=False)
    monkeypatch.delenv("PLATFORM_EMBEDDING_MODEL", raising=False)

    from common.embedding._platform import (
        get_platform_embedding,
        init_platform_embedding,
    )
    from common.embedding.providers.openrouter import OpenRouterEmbeddingProvider

    init_platform_embedding()
    emb = get_platform_embedding()
    # Narrow from the ``EmbeddingProvider`` protocol to the concrete class
    # before touching its private attrs below — the protocol only exposes
    # ``embed``/``embed_batch``/``provider_name``/``model``.
    assert isinstance(emb, OpenRouterEmbeddingProvider)
    assert emb._send_dimensions is False
    assert emb._truncate_to_dim == VECTOR_DIM
    assert str(emb._client.base_url).rstrip(
        "/"
    ) == OPENROUTER_EMBEDDING_BASE_URL.rstrip("/")


@pytest.mark.unit
def test_platform_openrouter_invalid_send_dimensions_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-canonical ``PLATFORM_EMBEDDING_SEND_DIMENSIONS`` value must
    reject the config rather than silently coercing."""
    monkeypatch.setenv("PLATFORM_EMBEDDING_PROVIDER", "openrouter")
    monkeypatch.setenv("PLATFORM_EMBEDDING_API_KEY", "sk-or-platform")
    monkeypatch.setenv("PLATFORM_EMBEDDING_SEND_DIMENSIONS", "yes")

    from common.embedding._platform import (
        get_platform_embedding,
        get_platform_init_errors,
        init_platform_embedding,
    )

    init_platform_embedding()
    assert get_platform_embedding() is None
    assert "openrouter-embedding-config" in get_platform_init_errors()


@pytest.mark.unit
def test_platform_openrouter_truncate_not_vector_dim_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``PLATFORM_EMBEDDING_TRUNCATE_TO_DIM`` other than ``VECTOR_DIM``
    must reject the config — the knob only exists to truncate a wider
    model's output down to the schema dimension."""
    from common.constants import VECTOR_DIM

    monkeypatch.setenv("PLATFORM_EMBEDDING_PROVIDER", "openrouter")
    monkeypatch.setenv("PLATFORM_EMBEDDING_API_KEY", "sk-or-platform")
    monkeypatch.setenv("PLATFORM_EMBEDDING_TRUNCATE_TO_DIM", str(VECTOR_DIM + 1))

    from common.embedding._platform import (
        get_platform_embedding,
        get_platform_init_errors,
        init_platform_embedding,
    )

    init_platform_embedding()
    assert get_platform_embedding() is None
    assert "openrouter-embedding-config" in get_platform_init_errors()
