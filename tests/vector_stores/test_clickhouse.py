"""Tests for the ClickHouse vector store provider (mem0#6991)."""

import json
import sys
import unittest
import uuid
from unittest.mock import MagicMock, patch

import pytest

from mem0.configs.vector_stores.clickhouse import ClickhouseConfig
from mem0.utils.factory import VectorStoreFactory


class TestClickhouseDB(unittest.TestCase):
    """Unit tests for ClickhouseDB with a mocked clickhouse_connect client."""

    def setUp(self):
        self.mock_client = MagicMock()
        patcher = patch("mem0.vector_stores.clickhouse.clickhouse_connect.get_client")
        self.mock_get_client = patcher.start()
        self.mock_get_client.return_value = self.mock_client
        self.addCleanup(patcher.stop)

        # Re-import so the module-level import guard resolves to the mock
        import importlib

        self.module = importlib.import_module("mem0.vector_stores.clickhouse")
        self.ClickhouseDB = self.module.ClickhouseDB

        self.db = self.ClickhouseDB(
            host="localhost",
            port=8123,
            username="default",
            password="",
            database="default",
            collection_name="mem0",
            embedding_model_dims=1536,
        )

    def test_init_and_config_object(self):
        cfg = ClickhouseConfig(host="ch.example.com", collection_name="col2")
        self.mock_get_client.reset_mock()
        db = self.ClickhouseDB(config=cfg)
        self.assertEqual(db.config.collection_name, "col2")
        self.mock_get_client.assert_called_once_with(
            host="ch.example.com",
            port=8123,
            username="default",
            password="",
            database="default",
            secure=False,
        )

    def test_init_kwargs_form(self):
        db = self.ClickhouseDB(
            host="h",
            port=8123,
            username="u",
            password="p",
            database="d",
            collection_name="c",
            embedding_model_dims=4,
        )
        self.assertEqual(db.config.database, "d")
        self.assertEqual(db.config.collection_name, "c")

    def test_create_col_cosine_default(self):
        self.mock_client.command.reset_mock()
        self.db.create_col(name="col_a", vector_size=4, distance="cosine")
        sql = self.mock_client.command.call_args[0][0]
        self.assertIn("CREATE TABLE IF NOT EXISTS", sql)
        self.assertIn("default.col_a", sql)
        self.assertIn("ReplacingMergeTree()", sql)

    def test_create_col_uses_configured_name(self):
        self.mock_client.command.reset_mock()
        self.db.create_col()
        sql = self.mock_client.command.call_args[0][0]
        self.assertIn("default.mem0", sql)

    def test_insert_defaults(self):
        self.mock_client.insert.reset_mock()
        vectors = [[0.1, 0.2], [0.3, 0.4]]
        self.db.insert(vectors)
        args = self.mock_client.insert.call_args
        data = args[0][1]
        self.assertEqual(len(data), 2)
        for row in data:
            self.assertEqual(len(row), 3)
            uuid.UUID(row[0])  # valid uuid string
            self.assertEqual(json.loads(row[2]), {})
        self.assertEqual(args[1]["column_names"], ["id", "vector", "payload"])

    def test_insert_with_ids_and_payloads(self):
        self.mock_client.insert.reset_mock()
        vectors = [[0.1]]
        ids = ["abc"]
        payloads = [{"k": "v"}]
        self.db.insert(vectors, payloads=payloads, ids=ids)
        data = self.mock_client.insert.call_args[0][1]
        self.assertEqual(data[0][0], "abc")
        self.assertEqual(json.loads(data[0][2]), {"k": "v"})

    def test_search_cosine(self):
        self.mock_client.query.reset_mock()
        row = ("id1", [0.1, 0.2], json.dumps({"a": 1}), 0.25)
        res = MagicMock()
        res.result_rows = [row]
        self.mock_client.query.return_value = res
        out = self.db.search("q", [0.1, 0.2], top_k=5)
        sql = self.mock_client.query.call_args[0][0]
        self.assertIn("cosineDistance", sql)
        self.assertIn("FINAL", sql)
        self.assertIn("%(query_vector)s", sql)
        self.assertEqual(out[0].id, "id1")
        self.assertAlmostEqual(out[0].score, 0.75)
        self.assertEqual(out[0].payload, {"a": 1})

    def test_search_l2_mapping(self):
        self.mock_client.query.reset_mock()
        res = MagicMock()
        res.result_rows = [("id1", [0.1], "{}", 2.0)]
        self.mock_client.query.return_value = res
        self.db.create_col(name="mem0", distance="l2")
        self.db.search("q", [0.1])
        sql = self.mock_client.query.call_args[0][0]
        self.assertIn("L2Distance", sql)

    def test_search_with_filters(self):
        self.mock_client.query.reset_mock()
        res = MagicMock()
        res.result_rows = [("id1", [0.1], json.dumps({"tag": "x"}), 0.1)]
        self.mock_client.query.return_value = res
        self.db.search("q", [0.1], filters={"tag": "x"})
        sql = self.mock_client.query.call_args[0][0]
        params = self.mock_client.query.call_args[1]["parameters"]
        self.assertIn("JSONExtractString(payload, %(k0)s) = %(v0)s", sql)
        self.assertEqual(params["k0"], "tag")
        self.assertEqual(params["v0"], "x")

    def test_get_found_and_missing(self):
        res = MagicMock()
        res.result_rows = [("id1", [0.1], json.dumps({"a": 1}))]
        self.mock_client.query.return_value = res
        got = self.db.get("id1")
        self.assertEqual(got.id, "id1")
        self.assertEqual(got.payload, {"a": 1})

        self.mock_client.query.return_value = MagicMock(result_rows=[])
        self.assertIsNone(self.db.get("missing"))

    def test_list(self):
        res = MagicMock()
        res.result_rows = [("id1", [0.1], "{}"), ("id2", [0.2], json.dumps({"b": 2}))]
        self.mock_client.query.return_value = res
        out = self.db.list(top_k=10)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[1].payload, {"b": 2})

    def test_list_cols(self):
        res = MagicMock()
        res.result_rows = [("mem0",), ("other",)]
        self.mock_client.query.return_value = res
        cols = self.db.list_cols()
        self.assertEqual(cols, ["mem0", "other"])

    def test_delete(self):
        self.mock_client.command.reset_mock()
        self.db.delete("id1")
        sql = self.mock_client.command.call_args[0][0]
        self.assertIn("DELETE WHERE id = %(id)s", sql)

    def test_update(self):
        self.mock_client.query.reset_mock()
        res = MagicMock()
        res.result_rows = [("id1", [0.1], json.dumps({"a": 1}))]
        self.mock_client.query.return_value = res
        self.mock_client.insert.reset_mock()
        self.db.update("id1", vector=[0.9], payload={"b": 2})
        data = self.mock_client.insert.call_args[0][1]
        self.assertEqual(data[0][0], "id1")
        self.assertEqual(json.loads(data[0][2]), {"b": 2})

    def test_col_info(self):
        res = MagicMock()
        res.result_rows = [[5]]
        self.mock_client.query.return_value = res
        info = self.db.col_info()
        self.assertEqual(info["count"], 5)
        self.assertEqual(info["name"], "mem0")

    def test_reset(self):
        self.mock_client.command.reset_mock()
        self.db.reset()
        calls = [c[0][0] for c in self.mock_client.command.call_args_list]
        self.assertTrue(any("DROP TABLE IF EXISTS" in c for c in calls))
        self.assertTrue(any("CREATE TABLE IF NOT EXISTS" in c for c in calls))

    def test_identifier_validation(self):
        with self.assertRaises(ValueError):
            self.ClickhouseDB(
                host="h",
                port=8123,
                database="bad-db; DROP",
                collection_name="mem0",
                embedding_model_dims=4,
            )

    def test_factory_creation(self):
        store = VectorStoreFactory.create(
            "clickhouse",
            {
                "host": "localhost",
                "port": 8123,
                "username": "default",
                "password": "",
                "database": "default",
                "collection_name": "mem0",
                "embedding_model_dims": 1536,
            },
        )
        self.assertIsInstance(store, self.ClickhouseDB)


class TestClickhouseImportError(unittest.TestCase):
    def test_import_error_when_package_missing(self):
        # Simulate clickhouse_connect absent; importing the module must raise
        # ImportError. Remove both the package and the already-imported module
        # from sys.modules so the import guard actually runs.
        import importlib

        saved_pkg = sys.modules.get("clickhouse_connect")
        saved_module = sys.modules.get("mem0.vector_stores.clickhouse")
        # None in sys.modules is a "failed import" placeholder: `import
        # clickhouse_connect` raises ImportError even though the package is
        # installed in this environment.
        sys.modules["clickhouse_connect"] = None
        for key in list(sys.modules):
            if key.startswith("clickhouse_connect") or key == "mem0.vector_stores.clickhouse":
                del sys.modules[key]
        sys.modules["clickhouse_connect"] = None
        try:
            with pytest.raises(ImportError, match="clickhouse-connect"):
                importlib.import_module("mem0.vector_stores.clickhouse")
        finally:
            if saved_pkg is not None:
                sys.modules["clickhouse_connect"] = saved_pkg
            if saved_module is not None:
                sys.modules["mem0.vector_stores.clickhouse"] = saved_module
