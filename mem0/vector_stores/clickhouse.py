"""ClickHouse vector store integration for mem0.

Uses ``clickhouse-connect`` against the ClickHouse HTTP interface and a
``ReplacingMergeTree`` table keyed on ``id`` for upsert/update semantics.
"""

import json
import logging
import re
import uuid
from typing import Any, Dict, List, Optional

try:
    import clickhouse_connect
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError("clickhouse-connect is not installed. Install it with 'pip install clickhouse-connect'") from exc

from pydantic import BaseModel

from mem0.configs.vector_stores.clickhouse import ClickhouseConfig
from mem0.vector_stores.base import VectorStoreBase

logger = logging.getLogger(__name__)

# Distance metric name -> ClickHouse SQL function.
_DISTANCE_FUNCS = {
    "cosine": "cosineDistance",
    "l2": "L2Distance",
    "dot": "dotProduct",
}

# Identifiers are interpolated into DDL, so they must be strict.
_SAFE_IDENTIFIER_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
# Filter keys are interpolated into JSONExtractString paths.
_SAFE_FILTER_KEY_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def _validate_identifier(name: str, label: str) -> str:
    if not _SAFE_IDENTIFIER_RE.match(name):
        raise ValueError(
            f"Invalid {label} {name!r}: only letters, digits, and underscores are allowed, "
            "must start with a letter or underscore."
        )
    return name


class OutputData(BaseModel):
    """Standard output structure returned from vector operations."""

    id: Optional[str]
    score: Optional[float]
    payload: Optional[Dict[str, Any]]


class ClickhouseDB(VectorStoreBase):
    """Vector store backed by ClickHouse."""

    def __init__(self, config: Optional[ClickhouseConfig] = None, **kwargs):
        """Initialize the ClickHouse connection and ensure the collection exists.

        Args:
            config: Pre-built ClickhouseConfig instance. If omitted, kwargs are
                used to construct one (the form VectorStoreFactory uses).
        """
        if isinstance(config, ClickhouseConfig):
            self.config = config
        else:
            self.config = ClickhouseConfig(**(kwargs or {}))

        _validate_identifier(self.config.database, "database")
        _validate_identifier(self.config.collection_name, "collection_name")

        self._distance_func = _DISTANCE_FUNCS.get(self.config.distance_metric, "cosineDistance")

        self.client = clickhouse_connect.get_client(
            host=self.config.host,
            port=self.config.port,
            username=self.config.username,
            password=self.config.password,
            database=self.config.database,
            secure=self.config.secure,
        )
        self.create_col(name=self.config.collection_name)

    def _table(self, name: Optional[str] = None) -> str:
        col = _validate_identifier(name or self.config.collection_name, "collection_name")
        return f"{self.config.database}.{col}"

    def create_col(
        self,
        name: Optional[str] = None,
        vector_size: Optional[int] = None,
        distance: str = "cosine",
    ):
        """Create a ClickHouse table for vectors if it does not already exist.

        Args:
            name: Table/collection name. Defaults to the configured collection.
            vector_size: Embedding dimensions (used for validation/docs only).
            distance: Similarity metric: cosine, l2 or dot. Selects the SQL
                distance function used by search.
        """
        table = self._table(name)
        self._distance_func = _DISTANCE_FUNCS.get(distance, "cosineDistance")
        self.client.command(
            f"CREATE TABLE IF NOT EXISTS {table} "
            "(id String, vector Array(Float32), payload String) "
            "ENGINE = ReplacingMergeTree() PRIMARY KEY id ORDER BY id"
        )
        logger.debug("Created ClickHouse collection %s", table)

    def insert(
        self,
        vectors: List[List[float]],
        payloads: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
    ):
        """Insert vectors with optional payloads and ids.

        Args:
            vectors: List of embedding vectors.
            payloads: List of payload dicts (defaults to empty dicts).
            ids: List of string ids (defaults to generated UUIDs).
        """
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in vectors]
        if payloads is None:
            payloads = [{} for _ in vectors]
        data = [[i, v, json.dumps(p)] for i, v, p in zip(ids, vectors, payloads)]
        self.client.insert(
            self._table(),
            data,
            column_names=["id", "vector", "payload"],
        )

    def search(
        self,
        query: str,
        vectors: List[float],
        top_k: int = 5,
        filters: Optional[dict] = None,
    ) -> List[OutputData]:
        """Search for the most similar vectors.

        Returns similarity scores where higher is better (base.py contract):
        cosine/l2 distance is converted with ``max(0.0, 1.0 - dist)``; dot
        product is returned as-is.
        """
        where, params = self._build_filter_conditions(filters)
        params["query_vector"] = vectors
        params["top_k"] = top_k
        sql = (
            f"SELECT id, vector, payload, {self._distance_func}(vector, %(query_vector)s) AS dist "
            f"FROM {self._table()} FINAL"
        )
        if where:
            sql += f" WHERE {where}"
        sql += " ORDER BY dist ASC LIMIT %(top_k)s"
        res = self.client.query(sql, parameters=params)

        out = []
        for row in res.result_rows:
            dist = float(row[3])
            if self._distance_func == "dotProduct":
                score = dist
            else:
                score = max(0.0, 1.0 - dist)
            out.append(
                OutputData(
                    id=str(row[0]),
                    score=score,
                    payload=json.loads(row[2]) if row[2] else {},
                )
            )
        return out

    def _build_filter_conditions(self, filters: Optional[dict]) -> tuple[str, Dict[str, Any]]:
        """Translate a filter dict into a ClickHouse WHERE clause with parameters."""
        if not filters:
            return "", {}
        clauses = []
        params: Dict[str, Any] = {}
        for idx, (key, value) in enumerate(filters.items()):
            if not _SAFE_FILTER_KEY_RE.match(key):
                logger.warning("Skipping invalid filter key %r", key)
                continue
            k_param, v_param = f"k{idx}", f"v{idx}"
            params[k_param] = key
            params[v_param] = value
            clauses.append(f"JSONExtractString(payload, %({k_param})s) = %({v_param})s")
        return " AND ".join(clauses), params

    def delete(self, vector_id: str):
        """Delete a vector by id."""
        self.client.command(
            f"ALTER TABLE {self._table()} DELETE WHERE id = %(id)s",
            parameters={"id": vector_id},
        )

    def update(
        self,
        vector_id: str,
        vector: Optional[List[float]] = None,
        payload: Optional[dict] = None,
    ):
        """Update a vector and/or its payload by re-inserting the same id.

        ReplacingMergeTree deduplicates rows with the same primary key, so the
        re-inserted row replaces the old one on the next FINAL read.
        """
        existing = self.get(vector_id)
        if existing is None:
            logger.warning("Cannot update missing vector %s", vector_id)
            return
        new_vector = vector if vector is not None else []
        new_payload = payload if payload is not None else (existing.payload or {})
        self.insert(vectors=[new_vector], payloads=[new_payload], ids=[vector_id])

    def get(self, vector_id: str) -> Optional[OutputData]:
        """Retrieve a vector by id."""
        res = self.client.query(
            f"SELECT id, vector, payload FROM {self._table()} FINAL WHERE id = %(id)s",
            parameters={"id": vector_id},
        )
        if not res.result_rows:
            return None
        row = res.result_rows[0]
        return OutputData(
            id=str(row[0]),
            score=None,
            payload=json.loads(row[2]) if row[2] else {},
        )

    def list_cols(self) -> List[str]:
        """List all tables (collections) in the configured database."""
        res = self.client.query(f"SHOW TABLES FROM {self.config.database}")
        return [str(row[0]) for row in res.result_rows]

    def delete_col(self):
        """Drop the collection table."""
        self.client.command(f"DROP TABLE IF EXISTS {self._table()}")

    def col_info(self) -> dict:
        """Return collection info: name and row count."""
        res = self.client.query(f"SELECT count() FROM {self._table()} FINAL")
        count = res.result_rows[0][0] if res.result_rows else 0
        return {"name": self.config.collection_name, "count": count}

    def list(self, filters: Optional[dict] = None, top_k: int = 100) -> List[OutputData]:
        """List vectors, optionally filtered."""
        where, params = self._build_filter_conditions(filters)
        params["top_k"] = top_k
        sql = f"SELECT id, vector, payload FROM {self._table()} FINAL"
        if where:
            sql += f" WHERE {where}"
        sql += " LIMIT %(top_k)s"
        res = self.client.query(sql, parameters=params)
        return [
            OutputData(
                id=str(row[0]),
                score=None,
                payload=json.loads(row[2]) if row[2] else {},
            )
            for row in res.result_rows
        ]

    def reset(self):
        """Drop and recreate the collection."""
        self.delete_col()
        self.create_col()
