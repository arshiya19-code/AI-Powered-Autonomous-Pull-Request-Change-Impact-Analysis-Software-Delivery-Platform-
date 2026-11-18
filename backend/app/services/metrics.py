from __future__ import annotations

try:
    from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
except ImportError:  # Keep the API usable until the optional lightweight dependency is installed.
    _fallback_metrics = []
    class _Metric:
        def __init__(self, name, help, labels=(), registry=None): self.name=name; self.labels_names=labels; self.values={}; _fallback_metrics.append(self)
        def labels(self, *values): return _Child(self, tuple(values))
        def observe(self, value): self.values[()] = self.values.get((), 0) + value
    class _Child:
        def __init__(self, parent, key): self.parent=parent; self.key=key
        def inc(self, value=1): self.parent.values[self.key] = self.parent.values.get(self.key, 0) + value
        def set(self, value): self.parent.values[self.key] = value
    CollectorRegistry = object
    Counter = Gauge = Histogram = _Metric
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"
    def generate_latest(registry):
        lines=[]
        for metric in _fallback_metrics:
            if metric.values:
                for key, value in metric.values.items(): lines.append(f"{metric.name} {value}")
            else:
                lines.append(f"{metric.name} 0")
        return ("\n".join(lines)+"\n").encode()

registry = CollectorRegistry()
analyses = Counter("change_impact_analyses_total", "PR analyses", ["status"], registry=registry)
workflow_runs = Counter("change_impact_workflows_total", "Workflow runs", ["status"], registry=registry)
analysis_duration = Histogram("change_impact_analysis_duration_seconds", "Analysis duration", registry=registry)
workflow_duration = Histogram("change_impact_workflow_duration_seconds", "Workflow duration", registry=registry)
validations = Counter("change_impact_validations_total", "Validation checks", ["status"], registry=registry)
repairs = Counter("change_impact_repairs_total", "Repair attempts", ["status"], registry=registry)
deployments = Counter("change_impact_deployment_simulations_total", "Deployment simulations", ["action"], registry=registry)
recommendations = Counter("change_impact_recommendations_total", "Decision recommendations", ["decision"], registry=registry)
availability = Gauge("change_impact_service_availability", "Availability of bounded platform services", ["service"], registry=registry)
monitoring_health = Gauge("change_impact_monitoring_health", "Current monitoring health", ["health"], registry=registry)

def mark_availability(**states: str) -> None:
    for service, state in states.items():
        availability.labels(service).set(1 if state == "available" else 0)

def render() -> bytes:
    return generate_latest(registry)

METRICS_CONTENT_TYPE = CONTENT_TYPE_LATEST
