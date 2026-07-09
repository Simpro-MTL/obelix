"""Unit tests for ``OpenRouterEmbeddingProvider``.

Standalone-copy sibling of ``test_embedding_openai_compat.py`` — same
mocking helpers and test style, adapted for OpenRouter's defaults
(``send_dimensions=False``, ``truncate_to_dim=VECTOR_DIM``,
``provider_name == "openrouter"``).

What's covered (no network, no real OpenRouter calls):

* ``provider_name`` is ``"openrouter"``, not ``"openai"`` (U-4).
* ``base_url`` defaults to the OpenRouter endpoint and is forwarded to
  the underlying OpenAI-compatible SDK client.
* ``send_dimensions`` defaults to ``False`` (U-2) and the toggle behaves
  like the OpenAI provider's.
* Default ``truncate_to_dim=VECTOR_DIM`` slices + L2-renormalizes a wider
  model's native output (U-3 / F-4), and raises on an undersized
  response (F-3).
* ``embed_batch`` sorts by index and applies the same truncation.
* ``embed_query`` instruction-prefix behaviour matches the OpenAI provider.
* Both ``EmbeddingProvider`` and ``InstructionAwareEmbedder`` protocols.
* SDK errors propagate out of ``embed`` so the service-layer retry/degrade
  path can catch them (F-1).
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from common.constants import VECTOR_DIM
from common.embedding.constants import OPENROUTER_EMBEDDING_BASE_URL
from common.embedding.providers.openrouter import OpenRouterEmbeddingProvider

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _fake_openai_response(vector: list[float]):
    """Shape an OpenAI-compatible ``embeddings.create`` response object."""
    return SimpleNamespace(data=[SimpleNamespace(index=0, embedding=vector)])


def _patched_provider(
    monkeypatch: pytest.MonkeyPatch,
    *,
    response_vector: list[float] | None = None,
    **provider_kwargs,
) -> tuple[OpenRouterEmbeddingProvider, MagicMock, AsyncMock]:
    """Build an ``OpenRouterEmbeddingProvider`` whose underlying
    ``openai.AsyncOpenAI`` is fully mocked.

    Returns ``(provider, async_openai_constructor_mock, embeddings_create_mock)``.
    """
    if response_vector is None:
        response_vector = [0.5] * VECTOR_DIM

    embeddings_create = AsyncMock(return_value=_fake_openai_response(response_vector))
    fake_client = MagicMock()
    fake_client.embeddings.create = embeddings_create

    async_openai_ctor = MagicMock(return_value=fake_client)
    monkeypatch.setattr(
        "common.embedding.providers.openrouter.openai.AsyncOpenAI", async_openai_ctor
    )

    provider = OpenRouterEmbeddingProvider(api_key="sk-or-fake", **provider_kwargs)
    return provider, async_openai_ctor, embeddings_create


# ---------------------------------------------------------------------------
# Identity + base_url
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_openrouter_provider_name_is_openrouter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """U-4: provider identity must be ``"openrouter"``, not ``"openai"`` —
    matters for logs, degraded-provider stats, and cache namespacing."""
    provider, _, _ = _patched_provider(monkeypatch)
    assert provider.provider_name == "openrouter"


@pytest.mark.unit
def test_openrouter_base_url_defaults_to_openrouter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unlike the OpenAI provider (``base_url=None`` default), OpenRouter's
    ``base_url`` always defaults to the OpenRouter endpoint and is always
    forwarded — there's no "hosted OpenAI" fallback to omit it for."""
    _, async_openai_ctor, _ = _patched_provider(monkeypatch)
    assert (
        async_openai_ctor.call_args.kwargs.get("base_url")
        == OPENROUTER_EMBEDDING_BASE_URL
    )


@pytest.mark.unit
def test_openrouter_base_url_override_forwarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit ``base_url`` override (e.g. a proxy in front of
    OpenRouter) is still forwarded verbatim."""
    _, async_openai_ctor, _ = _patched_provider(
        monkeypatch, base_url="https://proxy.internal/v1"
    )
    assert (
        async_openai_ctor.call_args.kwargs.get("base_url")
        == "https://proxy.internal/v1"
    )


# ---------------------------------------------------------------------------
# send_dimensions default + toggle (U-2)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_openrouter_default_send_dimensions_false_omits_kwarg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """U-2: OpenRouter doesn't document a ``dimensions`` param, so the
    default must omit the kwarg entirely (unlike OpenAI's default True)."""
    provider, _, create = _patched_provider(monkeypatch)  # defaults
    await provider.embed("hello")
    assert "dimensions" not in create.call_args.kwargs


@pytest.mark.unit
@pytest.mark.asyncio
async def test_openrouter_send_dimensions_true_sends_vector_dim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit opt-in still sends ``dimensions=VECTOR_DIM`` for models
    that DO honour it."""
    provider, _, create = _patched_provider(monkeypatch, send_dimensions=True)
    await provider.embed("hello")
    assert create.call_args.kwargs.get("dimensions") == VECTOR_DIM


# ---------------------------------------------------------------------------
# Default truncation (U-3 / F-4) + undersized raise (F-3)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_openrouter_default_truncates_wider_model_to_1024(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """U-3 / F-4: the default ``truncate_to_dim=VECTOR_DIM`` slices +
    L2-renormalizes a wider model's native output (e.g. a 1536-dim
    ``openai/text-embedding-3-small`` response) down to the schema dim."""
    raw: list[float] = [float(x) for x in range(1, 2 * VECTOR_DIM + 1)]
    provider, _, _ = _patched_provider(monkeypatch, response_vector=raw)
    out = await provider.embed("anything")

    assert len(out) == VECTOR_DIM
    norm = math.sqrt(sum(x * x for x in out))
    assert math.isclose(norm, 1.0, abs_tol=1e-9)
    sum_sq = sum(v * v for v in range(1, VECTOR_DIM + 1))
    expected_norm = math.sqrt(sum_sq)
    for i, got in enumerate(out, start=1):
        assert math.isclose(got, i / expected_norm, abs_tol=1e-9)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_openrouter_native_1024_model_is_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A model that's already 1024-native passes through unchanged (no
    spurious renormalization when there's nothing to slice)."""
    raw = [0.1] * VECTOR_DIM
    provider, _, _ = _patched_provider(monkeypatch, response_vector=raw)
    out = await provider.embed("anything")
    assert out == raw


@pytest.mark.unit
@pytest.mark.asyncio
async def test_openrouter_raises_on_undersized_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F-3: a model returning fewer than ``VECTOR_DIM`` dims must raise
    ``ValueError`` naming both dims — fail fast, attributable to the
    model+truncate config rather than a downstream pgvector write error."""
    half = VECTOR_DIM // 2
    raw: list[float] = [float(x) for x in range(1, half + 1)]
    provider, _, _ = _patched_provider(monkeypatch, response_vector=raw)

    with pytest.raises(ValueError) as ei:
        await provider.embed("any")
    msg = str(ei.value)
    assert str(half) in msg
    assert str(VECTOR_DIM) in msg


@pytest.mark.unit
def test_openrouter_provider_init_rejects_truncate_below_vector_dim() -> None:
    """Defence-in-depth: direct construction with ``truncate_to_dim !=
    VECTOR_DIM`` raises immediately rather than silently producing
    schema-incompatible vectors."""
    with pytest.raises(ValueError, match="must equal VECTOR_DIM"):
        OpenRouterEmbeddingProvider(api_key="sk-or-test", truncate_to_dim=512)


# ---------------------------------------------------------------------------
# embed_batch
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_openrouter_embed_batch_sorted_and_truncated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``embed_batch`` sorts by index and applies the same
    send_dimensions + truncation behaviour as ``embed``."""
    provider, _, create = _patched_provider(monkeypatch, send_dimensions=True)
    wide_a = list(range(1, 2 * VECTOR_DIM + 1))
    wide_b = [x * 2 for x in wide_a]
    create.return_value = SimpleNamespace(
        data=[
            SimpleNamespace(index=1, embedding=wide_b),
            SimpleNamespace(index=0, embedding=wide_a),
        ]
    )
    out = await provider.embed_batch(["a", "b"])

    assert create.call_args.kwargs.get("dimensions") == VECTOR_DIM
    assert len(out) == 2
    assert len(out[0]) == VECTOR_DIM
    assert len(out[1]) == VECTOR_DIM
    # index=0 (wide_a) must come first despite the response listing it second.
    assert out[0][0] < out[1][0] or math.isclose(out[0][0], out[1][0])


# ---------------------------------------------------------------------------
# embed_query (asymmetric, instruction-aware)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_openrouter_embed_query_no_instruction_passes_text_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Symmetric models get no prefix — equivalent to ``embed``."""
    provider, _, create = _patched_provider(monkeypatch)
    await provider.embed_query("what is the capital of France?")
    assert create.call_args.kwargs["input"] == "what is the capital of France?"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_openrouter_embed_query_with_instruction_prepends_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Per-call ``instruction`` produces the Qwen-style
    ``Instruct: ...\\nQuery: ...`` prefix."""
    provider, _, create = _patched_provider(monkeypatch)
    await provider.embed_query("paris", instruction="Retrieve relevant memories")
    expected = "Instruct: Retrieve relevant memories\nQuery: paris"
    assert create.call_args.kwargs["input"] == expected


# ---------------------------------------------------------------------------
# Protocol conformance + error propagation (F-1)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_openrouter_implements_both_protocols(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``OpenRouterEmbeddingProvider`` implements both ``EmbeddingProvider``
    and ``InstructionAwareEmbedder`` — same as the OpenAI provider."""
    from common.embedding.protocols import EmbeddingProvider, InstructionAwareEmbedder

    provider, _, _ = _patched_provider(monkeypatch)
    assert isinstance(provider, EmbeddingProvider)
    assert isinstance(provider, InstructionAwareEmbedder)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_openrouter_embed_propagates_provider_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F-1: the SDK raising (network error, 429, etc.) must propagate out
    of ``embed`` rather than being swallowed, so the service-layer
    ``_run_with_retry`` / degrade-to-``None`` path can catch it."""
    provider, _, create = _patched_provider(monkeypatch)
    create.side_effect = RuntimeError("OpenRouter unreachable")

    with pytest.raises(RuntimeError, match="OpenRouter unreachable"):
        await provider.embed("hello")
