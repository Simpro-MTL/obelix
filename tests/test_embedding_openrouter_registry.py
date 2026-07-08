"""Registry-side tests for the OpenRouter embedding provider.

Split out from ``test_embedding_openrouter.py`` (provider-level tests) to
stay under the per-file line budget — mirrors the registry-test section of
``test_embedding_openai_compat.py``, adapted for OpenRouter's env vars and
defaults (``OPENROUTER_EMBEDDING_SEND_DIMENSIONS`` defaults ``false``,
``OPENROUTER_EMBEDDING_TRUNCATE_TO_DIM`` defaults ``VECTOR_DIM``).

No network: constructing ``OpenRouterEmbeddingProvider`` only builds an
``openai.AsyncOpenAI`` client object (no I/O), so these tests call
``get_embedding_provider`` directly without mocking the SDK, matching the
existing OpenAI registry tests' style.
"""

from __future__ import annotations

from collections import OrderedDict
from types import SimpleNamespace

import pytest

from common.constants import VECTOR_DIM
from common.embedding._registry import get_embedding_provider
from common.embedding.constants import OPENROUTER_EMBEDDING_MODEL
from common.embedding.providers.openrouter import OpenRouterEmbeddingProvider


def _reset_registry_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Wipe the registry's process-level OpenRouter provider LRU so a test
    sees a fresh cache-miss path. The cache lives in ``_registry_openrouter``
    (extracted out of ``_registry.py`` to keep that file's size in check)."""
    import common.embedding._registry_openrouter as openrouter_registry_mod

    monkeypatch.setattr(openrouter_registry_mod, "_openrouter_provider_cache", OrderedDict())


@pytest.mark.unit
def test_registry_openrouter_reads_openrouter_api_key_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No tenant config → falls back to ``OPENROUTER_API_KEY`` env, and
    constructs a real ``OpenRouterEmbeddingProvider``."""
    _reset_registry_state(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

    provider = get_embedding_provider("openrouter")
    assert isinstance(provider, OpenRouterEmbeddingProvider)


@pytest.mark.unit
def test_registry_openrouter_tenant_key_overrides_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tenant-supplied ``openrouter_api_key`` wins over the env var."""
    _reset_registry_state(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-env")
    tenant_config = SimpleNamespace(openrouter_api_key="sk-or-tenant", embedding_model=None)

    provider = get_embedding_provider("openrouter", tenant_config)
    assert isinstance(provider, OpenRouterEmbeddingProvider)
    # Providers are cached by api_key — a different call with the env key
    # alone must NOT return this tenant-keyed instance.
    other = get_embedding_provider("openrouter")
    assert other is not provider


@pytest.mark.unit
def test_registry_openrouter_no_key_falls_back_to_fake(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F-2: no tenant key, no env key, no platform singleton configured →
    falls back to ``FakeEmbeddingProvider`` rather than crashing."""
    from common.embedding.providers.fake import FakeEmbeddingProvider

    _reset_registry_state(monkeypatch)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    import common.embedding._platform as platform_mod

    monkeypatch.setattr(platform_mod, "_platform_embedding", None)

    provider = get_embedding_provider("openrouter")
    assert isinstance(provider, FakeEmbeddingProvider)


@pytest.mark.unit
def test_registry_openrouter_default_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """U-1: with no tenant override and no ``OPENROUTER_EMBEDDING_MODEL``
    env, the default model is ``openai/text-embedding-3-small``."""
    _reset_registry_state(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.delenv("OPENROUTER_EMBEDDING_MODEL", raising=False)

    provider = get_embedding_provider("openrouter")
    assert isinstance(provider, OpenRouterEmbeddingProvider)
    assert provider.model == OPENROUTER_EMBEDDING_MODEL == "openai/text-embedding-3-small"


@pytest.mark.unit
def test_registry_openrouter_empty_send_dimensions_defaults_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F-5: docker-compose passes unset vars as ``""`` — an empty
    ``OPENROUTER_EMBEDDING_SEND_DIMENSIONS`` must default to ``false``,
    NOT raise."""
    _reset_registry_state(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("OPENROUTER_EMBEDDING_SEND_DIMENSIONS", "")

    provider = get_embedding_provider("openrouter")
    assert isinstance(provider, OpenRouterEmbeddingProvider)


@pytest.mark.unit
@pytest.mark.parametrize("val", ["yes", "1", "trun", "TRU"])
def test_registry_openrouter_rejects_non_canonical_send_dimensions(
    monkeypatch: pytest.MonkeyPatch,
    val: str,
) -> None:
    """F-6: a typo or non-canonical truthy spelling must raise, not
    silently coerce. Empty string is covered separately (F-5) — it must
    NOT raise."""
    _reset_registry_state(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("OPENROUTER_EMBEDDING_SEND_DIMENSIONS", val)

    with pytest.raises(
        ValueError,
        match=r"OPENROUTER_EMBEDDING_SEND_DIMENSIONS=.*must be 'true' or 'false'",
    ):
        get_embedding_provider("openrouter")


@pytest.mark.unit
def test_registry_openrouter_rejects_truncate_not_vector_dim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``OPENROUTER_EMBEDDING_TRUNCATE_TO_DIM`` other than ``VECTOR_DIM``
    is nonsensical — pgvector would reject the mismatched vector at write
    time. The registry catches it up front."""
    _reset_registry_state(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("OPENROUTER_EMBEDDING_TRUNCATE_TO_DIM", str(VECTOR_DIM + 256))

    with pytest.raises(ValueError, match="must equal VECTOR_DIM"):
        get_embedding_provider("openrouter")


@pytest.mark.unit
def test_registry_openrouter_rejects_non_integer_truncate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-integer ``OPENROUTER_EMBEDDING_TRUNCATE_TO_DIM`` must raise a
    ``ValueError`` naming the env var, not the bare CPython parse error."""
    _reset_registry_state(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("OPENROUTER_EMBEDDING_TRUNCATE_TO_DIM", "not-an-int")

    with pytest.raises(
        ValueError,
        match=r"OPENROUTER_EMBEDDING_TRUNCATE_TO_DIM=.*must be an integer",
    ):
        get_embedding_provider("openrouter")


@pytest.mark.unit
def test_registry_openrouter_caches_and_returns_same_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two calls with identical config resolve to the same cached
    provider instance rather than constructing a fresh client + httpx
    pool each time."""
    _reset_registry_state(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

    first = get_embedding_provider("openrouter")
    second = get_embedding_provider("openrouter")
    assert first is second
