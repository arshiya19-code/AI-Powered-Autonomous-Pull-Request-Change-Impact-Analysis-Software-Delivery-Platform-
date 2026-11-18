from __future__ import annotations
import hashlib, json, os, sqlite3, threading
from pathlib import Path
from typing import Any
from backend.app.schemas.models import FeedbackRecord

class FeedbackService:
    """Append-safe lifecycle evidence store. It never trains a model."""
    _lock = threading.RLock()
    def __init__(self, path: str | os.PathLike[str] | None = None):
        self.path = Path(path or os.getenv("FEEDBACK_STORE_PATH", "backend/feedback.sqlite3"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS feedback (workflow_id TEXT PRIMARY KEY, timestamp TEXT NOT NULL, payload TEXT NOT NULL)")
            db.commit()
    def _connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.execute("PRAGMA busy_timeout=30000")
        return db
    @staticmethod
    def fingerprint(features: dict[str, Any] | None) -> str | None:
        if features is None: return None
        return hashlib.sha256(json.dumps(features, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    def record(self, event: FeedbackRecord | None = None, **values) -> FeedbackRecord:
        item = event or FeedbackRecord(**values)
        payload = item.model_dump(mode="json")
        payload["features_fingerprint"] = payload["features_fingerprint"] or self.fingerprint(payload.get("features"))
        item = FeedbackRecord.model_validate(payload)
        with self._lock, self._connect() as db:
            db.execute("INSERT OR REPLACE INTO feedback(workflow_id,timestamp,payload) VALUES(?,?,?)", (item.workflow_id, item.analysis_timestamp.isoformat(), json.dumps(item.model_dump(mode="json"), sort_keys=True)))
            db.commit()
        return item
    def get(self, workflow_id: str) -> FeedbackRecord | None:
        with self._connect() as db:
            row = db.execute("SELECT payload FROM feedback WHERE workflow_id=?", (workflow_id,)).fetchone()
        return FeedbackRecord.model_validate_json(row[0]) if row else None
    def list(self, limit: int = 50, offset: int = 0) -> list[FeedbackRecord]:
        limit = max(1, min(int(limit), 100)); offset = max(0, int(offset))
        with self._connect() as db: rows = db.execute("SELECT payload FROM feedback ORDER BY timestamp DESC LIMIT ? OFFSET ?", (limit, offset)).fetchall()
        return [FeedbackRecord.model_validate_json(row[0]) for row in rows]
