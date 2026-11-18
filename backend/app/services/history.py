from __future__ import annotations
from .feedback import FeedbackService
from backend.app.schemas.models import HistoryItem

class HistoryService:
    def __init__(self, feedback: FeedbackService | None = None): self.feedback = feedback or FeedbackService()
    def list(self, limit: int = 50, offset: int = 0) -> list[HistoryItem]:
        return [self._item(x) for x in self.feedback.list(limit, offset)]
    def get(self, workflow_id: str) -> HistoryItem | None:
        x = self.feedback.get(workflow_id)
        return self._item(x) if x else None
    @staticmethod
    def _item(x):
        return HistoryItem(workflow_id=x.workflow_id, repository=x.repository, pull_request=x.pull_request, timestamp=x.analysis_timestamp, risk_status="known" if x.predicted_risk is not None else "unavailable", decision=x.decision, validation_status=x.validation_outcome, deployment_result=x.deployment_result, monitoring_health=x.monitoring_health, rollback_status=x.rollback_occurred)
