from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from collections.abc import Iterator
from pathlib import Path
from typing import Any


class SqliteResponseCache:
    """Small WAL-backed cache and counter store shared by worker processes."""

    def __init__(self, path: str | Path, *, namespace: str) -> None:
        self.path = Path(path)
        self.namespace = namespace
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS responses ("
                "namespace TEXT NOT NULL, cache_key TEXT NOT NULL, payload TEXT NOT NULL, "
                "PRIMARY KEY(namespace, cache_key))"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS counters ("
                "namespace TEXT NOT NULL, metric TEXT NOT NULL, value REAL NOT NULL, "
                "PRIMARY KEY(namespace, metric))"
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def get(self, key: str) -> Any | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM responses WHERE namespace=? AND cache_key=?",
                (self.namespace, key),
            ).fetchone()
        return None if row is None else json.loads(row[0])

    def put(self, key: str, payload: Any) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO responses(namespace, cache_key, payload) VALUES(?,?,?)",
                (self.namespace, key, encoded),
            )

    def increment(self, metric: str, value: float = 1.0) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO counters(namespace, metric, value) VALUES(?,?,?) "
                "ON CONFLICT(namespace, metric) DO UPDATE SET "
                "value=counters.value+excluded.value",
                (self.namespace, metric, float(value)),
            )

    def metrics(self) -> dict[str, float]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT metric, value FROM counters WHERE namespace=? ORDER BY metric",
                (self.namespace,),
            ).fetchall()
        return {str(metric): float(value) for metric, value in rows}


__all__ = ["SqliteResponseCache"]
