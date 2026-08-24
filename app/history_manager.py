import csv
import io
import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.config import HISTORY_DB_PATH

# Ensure repository root is known for relative paths
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class HistoryManager:
    """Manages persistent storage and retrieval of search queries in SQLite."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = HISTORY_DB_PATH

        if not os.path.isabs(db_path):
            self.db_path = os.path.join(_REPO_ROOT, db_path)
        else:
            self.db_path = db_path

        # Ensure directory exists
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)

        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        # Enable WAL mode for better concurrency
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
        except sqlite3.DatabaseError:
            pass
        return conn

    def _init_db(self) -> None:
        """Create the search_history table and indexes if they don't exist."""
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS search_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    query TEXT NOT NULL,
                    is_question INTEGER NOT NULL DEFAULT 0,
                    semantic_query TEXT,
                    result_count INTEGER NOT NULL DEFAULT 0,
                    execution_time_ms REAL NOT NULL DEFAULT 0.0,
                    filters_json TEXT,
                    top_patents_json TEXT,
                    extracted_answer TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_search_history_timestamp ON search_history(timestamp DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_search_history_query ON search_history(query)"
            )
            conn.commit()

    def add_search(
        self,
        query: str,
        is_question: bool = False,
        semantic_query: Optional[str] = None,
        result_count: int = 0,
        execution_time_ms: float = 0.0,
        filters: Optional[List[Any]] = None,
        top_patents: Optional[List[Any]] = None,
        extracted_answer: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> int:
        """Record a search query execution and return the new row ID."""
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty")

        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()

        filters_json = json.dumps(filters) if filters is not None else None
        top_patents_json = json.dumps(top_patents) if top_patents is not None else None

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO search_history (
                    timestamp, query, is_question, semantic_query,
                    result_count, execution_time_ms, filters_json,
                    top_patents_json, extracted_answer
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    query.strip(),
                    1 if is_question else 0,
                    semantic_query,
                    result_count,
                    round(execution_time_ms, 2),
                    filters_json,
                    top_patents_json,
                    extracted_answer,
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def get_history(
        self,
        limit: int = 100,
        offset: int = 0,
        search_term: Optional[str] = None,
        is_question_filter: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve historical search queries sorted by timestamp descending."""
        query_sql = "SELECT * FROM search_history"
        params: List[Any] = []
        where_clauses: List[str] = []

        if search_term and search_term.strip():
            where_clauses.append("(query LIKE ? OR semantic_query LIKE ?)")
            term = f"%{search_term.strip()}%"
            params.extend([term, term])

        if is_question_filter is not None:
            where_clauses.append("is_question = ?")
            params.append(1 if is_question_filter else 0)

        if where_clauses:
            query_sql += " WHERE " + " AND ".join(where_clauses)

        query_sql += " ORDER BY id DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query_sql, params)
            rows = cursor.fetchall()
            return [self._row_to_dict(row) for row in rows]

    def get_search_by_id(self, search_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve a specific search record by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM search_history WHERE id = ?", (search_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_dict(row)
            return None

    def delete_search(self, search_id: int) -> bool:
        """Delete a search record by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM search_history WHERE id = ?", (search_id,))
            conn.commit()
            return cursor.rowcount > 0

    def clear_history(self) -> bool:
        """Delete all search history records."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM search_history")
            conn.commit()
            return True

    def get_stats(self) -> Dict[str, Any]:
        """Compute aggregate statistics over search history."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT
                    COUNT(*) as total_searches,
                    SUM(CASE WHEN is_question = 1 THEN 1 ELSE 0 END) as question_count,
                    SUM(CASE WHEN is_question = 0 THEN 1 ELSE 0 END) as topic_count,
                    AVG(execution_time_ms) as avg_latency_ms,
                    SUM(result_count) as total_results,
                    MAX(timestamp) as latest_search_time
                FROM search_history
                """
            )
            row = cursor.fetchone()
            if not row or row["total_searches"] == 0:
                return {
                    "total_searches": 0,
                    "question_count": 0,
                    "topic_count": 0,
                    "avg_latency_ms": 0.0,
                    "total_results": 0,
                    "latest_search_time": None,
                }

            return {
                "total_searches": row["total_searches"] or 0,
                "question_count": row["question_count"] or 0,
                "topic_count": row["topic_count"] or 0,
                "avg_latency_ms": round(row["avg_latency_ms"] or 0.0, 1),
                "total_results": row["total_results"] or 0,
                "latest_search_time": row["latest_search_time"],
            }

    def export_csv(self) -> str:
        """Export all search history as CSV string."""
        records = self.get_history(limit=10000)
        output = io.StringIO()
        fieldnames = [
            "id",
            "timestamp",
            "query",
            "is_question",
            "semantic_query",
            "result_count",
            "execution_time_ms",
            "extracted_answer",
            "top_patents",
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            top_patents_str = (
                ", ".join(r.get("top_patents", []))
                if isinstance(r.get("top_patents"), list)
                else str(r.get("top_patents") or "")
            )
            writer.writerow(
                {
                    "id": r["id"],
                    "timestamp": r["timestamp"],
                    "query": r["query"],
                    "is_question": "Yes" if r["is_question"] else "No",
                    "semantic_query": r["semantic_query"] or "",
                    "result_count": r["result_count"],
                    "execution_time_ms": r["execution_time_ms"],
                    "extracted_answer": r["extracted_answer"] or "",
                    "top_patents": top_patents_str,
                }
            )
        return output.getvalue()

    def export_json(self) -> str:
        """Export all search history as formatted JSON string."""
        records = self.get_history(limit=10000)
        return json.dumps(records, indent=2)

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        filters = None
        if row["filters_json"]:
            try:
                filters = json.loads(row["filters_json"])
            except Exception:
                filters = row["filters_json"]

        top_patents = None
        if row["top_patents_json"]:
            try:
                top_patents = json.loads(row["top_patents_json"])
            except Exception:
                top_patents = row["top_patents_json"]

        return {
            "id": row["id"],
            "timestamp": row["timestamp"],
            "query": row["query"],
            "is_question": bool(row["is_question"]),
            "semantic_query": row["semantic_query"],
            "result_count": row["result_count"],
            "execution_time_ms": row["execution_time_ms"],
            "filters": filters,
            "top_patents": top_patents,
            "extracted_answer": row["extracted_answer"],
        }
