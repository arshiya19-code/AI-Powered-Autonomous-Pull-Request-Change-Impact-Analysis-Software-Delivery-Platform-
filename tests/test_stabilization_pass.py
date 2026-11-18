from backend.app.presentation import format_duration
from backend.app.services.analysis import AnalysisPipelineError, AnalysisService, PRAnalysisRequest
from backend.app.services.stages import transition


DIFF = """diff --git a/pkg/mod.py b/pkg/mod.py
--- a/pkg/mod.py
+++ b/pkg/mod.py
@@ -1,2 +1,2 @@
-from old_dep import value
+from new_dep import value
 def changed():
     return value
"""


def test_analysis_uses_distinct_explicit_base_and_head_snapshots():
    request = PRAnalysisRequest(
        repository="fixture/repo", base_sha="base123", head_sha="head456", diff=DIFF,
        sources={"pkg/mod.py": "from new_dep import value\ndef changed():\n    return value\n"},
        base_sources={"pkg/mod.py": "from old_dep import value\ndef changed():\n    return value\n"},
        head_sources={"pkg/mod.py": "from new_dep import value\ndef changed():\n    return value\n"},
    )
    calls = []
    original = AnalysisService._build_graph
    try:
        AnalysisService._build_graph = lambda self, sources, repository_root=None, evidence=None: original(self, sources, repository_root, evidence)
        service = AnalysisService()
        original_analyze = service._acquire
        service._acquire = lambda req: (req.diff, req.sources or {}, [])
        response = service.analyze(request)
        assert response.pr_metadata.base_sha == "base123"
        assert response.pr_metadata.head_sha == "head456"
        assert response.dependency_change_summary.added == 1
        assert response.dependency_change_summary.removed == 1
    finally:
        AnalysisService._build_graph = original


def test_graph_failures_are_not_converted_to_empty_success():
    service = AnalysisService()
    def fail(*args, **kwargs):
        raise RuntimeError("resolver exploded")
    service._build_graph = fail
    request = PRAnalysisRequest(diff=DIFF, sources={"pkg/mod.py": "def changed():\n    return 1\n"})
    try:
        service.analyze(request)
    except RuntimeError as exc:
        assert str(exc) == "resolver exploded"
    else:
        raise AssertionError("graph failure was swallowed")


def test_graph_skips_minified_vendor_assets_but_records_evidence(tmp_path):
    evidence = {}
    graph = AnalysisService()._build_graph(
        {
            "app.py": "def answer():\n    return 42\n",
            "static/js/jquery.min.js": "function(){",
        },
        tmp_path,
        evidence,
    )

    assert graph.nodes
    assert evidence["parsed_source_count"] == 1
    assert evidence["excluded_source_count"] == 1
    assert evidence["excluded_sources"] == [
        {"file": "static/js/jquery.min.js", "reason": "minified generated asset"}
    ]


def test_failed_stage_skips_downstream_and_has_no_fake_duration():
    stages = transition({}, "ANALYSIS_STARTED")
    stages = transition(stages, "GRAPH_STARTED")
    stages = transition(stages, "GRAPH_FAILED", {"message": "resolver failed"})
    assert stages["GRAPH_BUILT"]["status"] == "FAILED"
    assert stages["IMPACT"]["status"] == "SKIPPED"
    assert stages["IMPACT"]["duration"] is None


def test_duration_formatting_is_measured_or_not_executed():
    assert format_duration(2.83, "PASS") == "Duration: 2.83 s"
    assert format_duration(0.37, "PASS") == "Duration: 0.37 s"
    assert format_duration(0.006, "PASS") == "Duration: 6 ms"
    assert format_duration(None, "SKIPPED") == "Not executed"
    assert format_duration(0, "NOT_CONFIGURED") == "Not executed"
