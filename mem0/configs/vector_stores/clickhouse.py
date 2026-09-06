"""Pydantic configuration for the ClickHouse vector store integration."""

import os
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


class ClickhouseConfig(BaseModel):
    """Configuration required to connect to a ClickHouse server.

    All fields can be overridden by environment variables
    (``CLICKHOUSE_HOST``, ``CLICKHOUSE_PORT``, ``CLICKHOUSE_USERNAME``,
    ``CLICKHOUSE_PASSWORD``, ``CLICKHOUSE_DATABASE``, ``CLICKHOUSE_SECURE``),
    which take precedence over explicit values only when the explicit value
    is not provided.
    """

    database: Optional[str] = Field("default", description="Database name")
    collection_name: str = Field("mem0", description="Default collection name")
    embedding_model_dims: Optional[int] = Field(1536, description="Embedding dimensions")
    host: Optional[str] = Field("localhost", description="ClickHouse host")
    port: Optional[int] = Field(8123, description="ClickHouse HTTP port")
    username: Optional[str] = Field("default", description="ClickHouse username")
    password: Optional[str] = Field("", description="ClickHouse password")
    secure: Optional[bool] = Field(False, description="HTTPS/SSL connection flag")
    distance_metric: Literal["cosine", "l2", "dot"] = Field(
        "cosine", description="Similarity metric used for vector search"
    )

    @model_validator(mode="before")
    @classmethod
    def check_env_vars(cls, values):
        """Pull connection settings from environment variables when not set explicitly."""
        env_map = {
            "host": "CLICKHOUSE_HOST",
            "port": "CLICKHOUSE_PORT",
            "username": "CLICKHOUSE_USERNAME",
            "password": "CLICKHOUSE_PASSWORD",
            "database": "CLICKHOUSE_DATABASE",
            "secure": "CLICKHOUSE_SECURE",
        }
        for field, env_name in env_map.items():
            if field not in values or values.get(field) in (None, ""):
                env_value = os.getenv(env_name)
                if env_value is not None and env_value != "":
                    if field == "port":
                        values[field] = int(env_value)
                    elif field == "secure":
                        values[field] = env_value.lower() in ("1", "true", "yes", "on")
                    else:
                        values[field] = env_value
        return values
