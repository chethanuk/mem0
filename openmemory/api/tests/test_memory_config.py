"""Tests for OpenMemory's memory client configuration (environment -> mem0 config).

Covers LM Studio provider support (issue #6246) end-to-end through
``get_memory_client`` / ``get_default_memory_config`` -- the same path the API
takes on startup -- alongside regression coverage for the existing Ollama and
OpenAI paths.
"""

import os

# Set dummy keys before any imports that trigger client initialization
os.environ.setdefault("OPENAI_API_KEY", "test-key")

import pytest

from app.utils import memory as memory_utils
from app.utils.memory import get_default_memory_config, get_memory_client

# Provider-selection variables, cleared before every test so that the developer's
# own shell environment cannot change the outcome.
PROVIDER_ENV_VARS = (
    "LLM_PROVIDER",
    "LLM_MODEL",
    "LLM_API_KEY",
    "LLM_BASE_URL",
    "EMBEDDER_PROVIDER",
    "EMBEDDER_MODEL",
    "EMBEDDER_API_KEY",
    "EMBEDDER_BASE_URL",
    "OLLAMA_BASE_URL",
    "OLLAMA_HOST",
    "LMSTUDIO_BASE_URL",
    "LMSTUDIO_HOST",
)

# Vector-store auto-detection variables; cleared so every test lands on the
# default Qdrant branch regardless of the host environment.
VECTOR_STORE_ENV_VARS = (
    "CHROMA_HOST",
    "CHROMA_PORT",
    "QDRANT_HOST",
    "QDRANT_PORT",
    "WEAVIATE_CLUSTER_URL",
    "WEAVIATE_HOST",
    "WEAVIATE_PORT",
    "REDIS_URL",
    "PG_HOST",
    "PG_PORT",
    "MILVUS_HOST",
    "MILVUS_PORT",
    "ELASTICSEARCH_HOST",
    "ELASTICSEARCH_PORT",
    "OPENSEARCH_HOST",
    "OPENSEARCH_PORT",
    "FAISS_PATH",
)

LMSTUDIO_DEFAULT_URL = "http://localhost:1234/v1"
DOCKER_HOST = "host.docker.internal"


@pytest.fixture(autouse=True)
def clean_provider_env(monkeypatch):
    """Isolate every test from ambient provider configuration."""
    for var in PROVIDER_ENV_VARS + VECTOR_STORE_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


# ---------------------------------------------------------------------------
# THE failing test: a user pointing OpenMemory at LM Studio via env vars
# ---------------------------------------------------------------------------

def test_lmstudio_llm_config_uses_lmstudio_base_url(monkeypatch):
    """LLM_PROVIDER=lmstudio must produce a config the mem0 SDK can dial."""
    monkeypatch.setenv("LLM_PROVIDER", "lmstudio")
    monkeypatch.setenv("LLM_MODEL", "qwen2.5-7b-instruct")
    monkeypatch.setenv("LMSTUDIO_BASE_URL", LMSTUDIO_DEFAULT_URL)

    config = get_default_memory_config()

    assert config["llm"]["provider"] == "lmstudio"
    assert config["llm"]["config"]["lmstudio_base_url"] == LMSTUDIO_DEFAULT_URL
    assert config["llm"]["config"]["model"] == "qwen2.5-7b-instruct"


# ---------------------------------------------------------------------------
# Provider selection
# ---------------------------------------------------------------------------

def test_lmstudio_embedder_follows_llm_provider(monkeypatch):
    """A fully-local LM Studio stack must not silently fall back to OpenAI."""
    monkeypatch.setenv("LLM_PROVIDER", "lmstudio")
    monkeypatch.setenv("LLM_MODEL", "qwen2.5-7b-instruct")
    monkeypatch.setenv("EMBEDDER_MODEL", "text-embedding-nomic-embed-text-v1.5")
    monkeypatch.setenv("LMSTUDIO_BASE_URL", LMSTUDIO_DEFAULT_URL)

    config = get_default_memory_config()

    assert config["embedder"]["provider"] == "lmstudio"
    assert config["embedder"]["config"]["lmstudio_base_url"] == LMSTUDIO_DEFAULT_URL
    assert config["embedder"]["config"]["model"] == "text-embedding-nomic-embed-text-v1.5"
    assert "api_key" not in config["embedder"]["config"]


@pytest.mark.parametrize(
    "lmstudio_base_url, llm_base_url, expected",
    [
        ("http://lms:1234/v1", "http://other:9999/v1", "http://lms:1234/v1"),
        (None, "http://other:9999/v1", "http://other:9999/v1"),
        (None, None, LMSTUDIO_DEFAULT_URL),
    ],
)
def test_lmstudio_llm_base_url_precedence(monkeypatch, lmstudio_base_url, llm_base_url, expected):
    """LMSTUDIO_BASE_URL > LLM_BASE_URL > default (mirrors the Ollama chain)."""
    monkeypatch.setenv("LLM_PROVIDER", "lmstudio")
    monkeypatch.setenv("LLM_MODEL", "qwen2.5-7b-instruct")
    if lmstudio_base_url:
        monkeypatch.setenv("LMSTUDIO_BASE_URL", lmstudio_base_url)
    if llm_base_url:
        monkeypatch.setenv("LLM_BASE_URL", llm_base_url)

    config = get_default_memory_config()

    assert config["llm"]["config"]["lmstudio_base_url"] == expected


@pytest.mark.parametrize(
    "embedder_base_url, lmstudio_base_url, llm_base_url, expected",
    [
        ("http://emb:1234/v1", "http://lms:1234/v1", "http://llm:1234/v1", "http://emb:1234/v1"),
        (None, "http://lms:1234/v1", "http://llm:1234/v1", "http://lms:1234/v1"),
        (None, None, "http://llm:1234/v1", "http://llm:1234/v1"),
        (None, None, None, LMSTUDIO_DEFAULT_URL),
    ],
)
def test_lmstudio_embedder_base_url_precedence(
    monkeypatch, embedder_base_url, lmstudio_base_url, llm_base_url, expected
):
    """EMBEDDER_BASE_URL > LMSTUDIO_BASE_URL > LLM_BASE_URL > default."""
    monkeypatch.setenv("LLM_PROVIDER", "lmstudio")
    monkeypatch.setenv("LLM_MODEL", "qwen2.5-7b-instruct")
    monkeypatch.setenv("EMBEDDER_MODEL", "text-embedding-nomic-embed-text-v1.5")
    if embedder_base_url:
        monkeypatch.setenv("EMBEDDER_BASE_URL", embedder_base_url)
    if lmstudio_base_url:
        monkeypatch.setenv("LMSTUDIO_BASE_URL", lmstudio_base_url)
    if llm_base_url:
        monkeypatch.setenv("LLM_BASE_URL", llm_base_url)

    config = get_default_memory_config()

    assert config["embedder"]["config"]["lmstudio_base_url"] == expected


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("http://h:1234", "http://h:1234/v1"),
        ("http://h:1234/", "http://h:1234/v1"),
        ("http://h:1234/v1", "http://h:1234/v1"),
        ("http://h:1234/v1/", "http://h:1234/v1"),
    ],
)
def test_lmstudio_url_v1_normalization(raw, expected):
    """LM Studio serves an OpenAI-compatible API under /v1; accept either form."""
    assert memory_utils._normalize_lmstudio_url(raw) == expected


@pytest.mark.parametrize("provider_env, model_env", [("LLM", "LLM_MODEL"), ("EMBEDDER", "EMBEDDER_MODEL")])
def test_lmstudio_requires_model(monkeypatch, provider_env, model_env):
    """LM Studio model ids are machine-specific, so there is no sane default."""
    monkeypatch.setenv("LLM_PROVIDER", "lmstudio")
    monkeypatch.setenv("EMBEDDER_PROVIDER", "lmstudio")
    # Supply every model except the one under test.
    if model_env != "LLM_MODEL":
        monkeypatch.setenv("LLM_MODEL", "qwen2.5-7b-instruct")
    if model_env != "EMBEDDER_MODEL":
        monkeypatch.setenv("EMBEDDER_MODEL", "text-embedding-nomic-embed-text-v1.5")

    with pytest.raises(ValueError, match=model_env):
        get_default_memory_config()


@pytest.mark.parametrize(
    "api_key, expected",
    [
        (None, None),
        ("proxy-key", "proxy-key"),
    ],
)
def test_lmstudio_api_key_optional(monkeypatch, api_key, expected):
    """Stock LM Studio ignores the key; the SDK fills the "lm-studio" placeholder."""
    monkeypatch.setenv("LLM_PROVIDER", "lmstudio")
    monkeypatch.setenv("LLM_MODEL", "qwen2.5-7b-instruct")
    if api_key:
        monkeypatch.setenv("LLM_API_KEY", api_key)

    llm_config = get_default_memory_config()["llm"]["config"]

    assert llm_config.get("api_key") == expected


# ---------------------------------------------------------------------------
# Docker localhost -> host rewrite
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "provider, url_key, supplied, expected",
    [
        # LM Studio: localhost/127.0.0.1 must be redirected at the Docker host.
        ("lmstudio", "lmstudio_base_url", "http://localhost:1234/v1", f"http://{DOCKER_HOST}:1234/v1"),
        ("lmstudio", "lmstudio_base_url", "http://127.0.0.1:1234/v1", f"http://{DOCKER_HOST}:1234/v1"),
        # DB/config-router supplied URL without the /v1 suffix gets canonicalized.
        ("lmstudio", "lmstudio_base_url", "http://localhost:1234", f"http://{DOCKER_HOST}:1234/v1"),
        # Key absent entirely -> Docker-reachable default injected.
        ("lmstudio", "lmstudio_base_url", None, f"http://{DOCKER_HOST}:1234/v1"),
        # A remote LM Studio must be left alone.
        ("lmstudio", "lmstudio_base_url", "http://192.168.1.50:1234/v1", "http://192.168.1.50:1234/v1"),
        # Ollama behaviour must be unchanged.
        ("ollama", "ollama_base_url", "http://localhost:11434", f"http://{DOCKER_HOST}:11434"),
        ("ollama", "ollama_base_url", None, f"http://{DOCKER_HOST}:11434"),
        ("ollama", "ollama_base_url", "http://192.168.1.50:11434", "http://192.168.1.50:11434"),
    ],
)
def test_local_provider_docker_url_rewrite(monkeypatch, provider, url_key, supplied, expected):
    """localhost inside the API container cannot reach a host-run model server."""
    monkeypatch.setattr(memory_utils, "_get_docker_host_url", lambda *args, **kwargs: DOCKER_HOST)

    config_section = {"provider": provider, "config": {"model": "some-model"}}
    if supplied is not None:
        config_section["config"][url_key] = supplied

    entry = memory_utils._LOCAL_PROVIDER_URLS[provider]
    fixed = memory_utils._fix_local_base_url(config_section, *entry)

    assert fixed["config"][url_key] == expected


# ---------------------------------------------------------------------------
# Entry point: the config the API actually hands to the mem0 SDK
# ---------------------------------------------------------------------------

def test_get_memory_client_passes_lmstudio_config_to_sdk(monkeypatch):
    """End-to-end: env vars -> the config dict Memory.from_config() receives."""
    monkeypatch.setenv("LLM_PROVIDER", "lmstudio")
    monkeypatch.setenv("LLM_MODEL", "qwen2.5-7b-instruct")
    monkeypatch.setenv("EMBEDDER_MODEL", "text-embedding-nomic-embed-text-v1.5")
    monkeypatch.setenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
    monkeypatch.setattr(memory_utils, "_get_docker_host_url", lambda *args, **kwargs: DOCKER_HOST)

    # No database in a unit test run: force the "use defaults" branch.
    def _no_db():
        raise RuntimeError("no database in tests")

    monkeypatch.setattr(memory_utils, "SessionLocal", _no_db)

    captured = {}

    class _StubMemory:
        @staticmethod
        def from_config(config_dict):
            captured.update(config_dict)
            return object()

    monkeypatch.setattr(memory_utils, "Memory", _StubMemory)
    memory_utils.reset_memory_client()

    client = get_memory_client()

    assert client is not None, "memory client failed to initialize"
    assert captured["llm"]["provider"] == "lmstudio"
    assert captured["embedder"]["provider"] == "lmstudio"
    # localhost was rewritten for the container, on both sections.
    assert captured["llm"]["config"]["lmstudio_base_url"] == f"http://{DOCKER_HOST}:1234/v1"
    assert captured["embedder"]["config"]["lmstudio_base_url"] == f"http://{DOCKER_HOST}:1234/v1"
    # A fully-local stack must not require an OpenAI key anywhere.
    assert "api_key" not in captured["llm"]["config"]
    assert "api_key" not in captured["embedder"]["config"]

    memory_utils.reset_memory_client()


# ---------------------------------------------------------------------------
# Regression: existing providers
# ---------------------------------------------------------------------------

def test_ollama_config_unchanged(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")

    config = get_default_memory_config()

    assert config["llm"]["provider"] == "ollama"
    assert config["llm"]["config"]["ollama_base_url"] == "http://localhost:11434"
    assert config["llm"]["config"]["model"] == "llama3.1:latest"
    assert config["embedder"]["provider"] == "ollama"
    assert config["embedder"]["config"]["ollama_base_url"] == "http://localhost:11434"


def test_openai_default_config_unchanged():
    config = get_default_memory_config()

    assert config["llm"]["provider"] == "openai"
    assert config["llm"]["config"]["model"] == "gpt-4o-mini"
    assert config["llm"]["config"]["api_key"] == "env:OPENAI_API_KEY"
    assert config["embedder"]["provider"] == "openai"
    assert config["embedder"]["config"]["model"] == "text-embedding-3-small"
