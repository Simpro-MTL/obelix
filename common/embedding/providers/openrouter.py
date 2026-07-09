"""OpenRouter embedding provider.

Standalone copy of :class:`~common.embedding.providers.openai.OpenAIEmbeddingProvider`
(user-selected: a thin subclass would DRY up the ~90% shared logic, but the
provider identity divergence — ``provider_name`` plus the OpenRouter-specific
defaults below — is treated as a first-class fork point rather than an
inheritance hook). Talks to OpenRouter's OpenAI-compatible embeddings
endpoint (``https://openrouter.ai/api/v1/embeddings``).
"""

from __future__ import annotations

import math

import httpx
import openai

from common.constants import VECTOR_DIM
from common.embedding.constants import (
    OPENAI_HTTPX_MAX_CONNECTIONS,
    OPENAI_HTTPX_MAX_KEEPALIVE_CONNECTIONS,
    OPENAI_REQUEST_TIMEOUT_SECONDS,
    OPENROUTER_EMBEDDING_BASE_URL,
    OPENROUTER_EMBEDDING_MODEL,
)


class OpenRouterEmbeddingProvider:
    """Embedding provider using OpenRouter's OpenAI-compatible embeddings API.

    OpenRouter does not document a ``dimensions`` request parameter (unlike
    hosted ``api.openai.com``), so this provider defaults
    ``send_dimensions=False`` and ``truncate_to_dim=VECTOR_DIM``: whatever
    dimensionality the routed model natively returns gets sliced + L2
    -renormalized down to the schema dimension client-side. That makes the
    provider schema-correct regardless of whether a given OpenRouter-hosted
    model happens to honour a client-supplied ``dimensions`` value — see
    :meth:`_postprocess`.
    """

    def __init__(
        self,
        api_key: str,
        model: str = OPENROUTER_EMBEDDING_MODEL,
        base_url: str = OPENROUTER_EMBEDDING_BASE_URL,
        send_dimensions: bool = False,
        query_instruction: str | None = None,
        truncate_to_dim: int | None = VECTOR_DIM,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._send_dimensions = send_dimensions
        self._query_instruction = query_instruction
        self._truncate_to_dim = truncate_to_dim
        # Defence in depth: the registry's ``get_embedding_provider``
        # already validates this knob before constructing a provider,
        # but direct construction (a one-off script, a migration helper,
        # a test that mocks the registry) bypasses that path. Without
        # this guard, ``truncate_to_dim=512`` against a 1024-dim schema
        # silently produces 512-element vectors that pgvector rejects
        # at write time with a dim-mismatch error far from the original
        # constructor call. Fail fast at construction instead.
        if truncate_to_dim is not None and truncate_to_dim != VECTOR_DIM:
            raise ValueError(
                f"truncate_to_dim={truncate_to_dim} must equal "
                f"VECTOR_DIM={VECTOR_DIM}; this knob is only for truncating a "
                "wider model's native output to the schema dimension."
            )
        # Explicit per-call timeout + httpx pool sizing, mirroring
        # ``OpenAIEmbeddingProvider``: reuse the same env-driven constants
        # rather than inventing OpenRouter-specific ones, since the
        # underlying SDK/httpx mechanics are identical.
        self._client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=OPENAI_REQUEST_TIMEOUT_SECONDS,
            http_client=httpx.AsyncClient(
                limits=httpx.Limits(
                    max_connections=OPENAI_HTTPX_MAX_CONNECTIONS,
                    max_keepalive_connections=OPENAI_HTTPX_MAX_KEEPALIVE_CONNECTIONS,
                ),
            ),
        )

    @property
    def provider_name(self) -> str:
        # NOT "openai" — this is the one hard divergence from the copy.
        # Matters for logs, degraded-provider stats, and cache namespacing.
        return "openrouter"

    @property
    def model(self) -> str:
        return self._model

    async def aclose(self) -> None:
        """Close the underlying httpx pool cleanly.

        Without this, ``asyncio`` debug mode emits ``ResourceWarning:
        Unclosed <httpx.AsyncClient>`` when the provider is GC'd —
        noisy in tests and a leak in long-lived processes that rotate
        client instances. Idempotent; safe to call multiple times.
        """
        await self._client.close()

    def _postprocess(self, emb: list[float]) -> list[float]:
        # Matryoshka truncation: slice to ``truncate_to_dim`` and L2-renormalize.
        # Models trained with MRL produce vectors that stay coherent under
        # truncation, but the resulting vector is no longer unit-norm — we
        # re-normalize so cosine similarity at the pgvector layer stays correct.
        if self._truncate_to_dim is not None and len(emb) > self._truncate_to_dim:
            emb = emb[: self._truncate_to_dim]
            n = math.sqrt(sum(x * x for x in emb))
            if n > 0:
                emb = [x / n for x in emb]
        elif self._truncate_to_dim is not None and len(emb) < self._truncate_to_dim:
            # Undersized passthrough is silently broken: returning a
            # vector with fewer than ``truncate_to_dim`` elements would
            # produce inserts that pgvector rejects with
            # ``expected N dimensions, not M``. Fail fast here so the
            # mismatch is attributable to the model+truncate config,
            # not surfaced obliquely as a write error far downstream.
            raise ValueError(
                f"Model returned {len(emb)}-dim vector but truncate_to_dim="
                f"{self._truncate_to_dim}; the configured model must produce "
                f"at least {self._truncate_to_dim} native dimensions."
            )
        return emb

    async def embed(self, text: str) -> list[float]:
        """Generate an embedding vector for a single text (document/ingest path)."""
        kwargs: dict = {"model": self._model, "input": text}
        if self._send_dimensions:
            kwargs["dimensions"] = VECTOR_DIM
        response = await self._client.embeddings.create(**kwargs)
        return self._postprocess(response.data[0].embedding)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embedding vectors for multiple texts in one call."""
        kwargs: dict = {"model": self._model, "input": texts}
        if self._send_dimensions:
            kwargs["dimensions"] = VECTOR_DIM
        response = await self._client.embeddings.create(**kwargs)
        # OpenRouter (like OpenAI) returns embeddings sorted by index.
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [self._postprocess(item.embedding) for item in sorted_data]

    async def embed_query(
        self, text: str, instruction: str | None = None
    ) -> list[float]:
        """Generate an embedding vector for a search-side query.

        For instruction-aware models routed through OpenRouter, prepends
        the resolved instruction in the convention these models were
        trained with: ``"Instruct: <task>\\nQuery: <text>"``. The
        instruction is resolved as: per-call *instruction* arg →
        constructor *query_instruction* → no prefix.

        For symmetric models (the default ``openai/text-embedding-3-small``
        routed via OpenRouter), pass no instruction at construction time
        and this is equivalent to :meth:`embed`.
        """
        instr = instruction or self._query_instruction
        if instr:
            text = f"Instruct: {instr}\nQuery: {text}"
        return await self.embed(text)
