from pathlib import Path


FRONTEND = Path(__file__).parents[1] / "frontend" / "src" / "main.tsx"


def test_dashboard_binds_repair_and_test_panels_to_persisted_evidence():
    source = FRONTEND.read_text(encoding="utf-8")
    for field in (
        "repair?.candidate?.files",
        "repair.patch",
        "repair.provider",
        "repair.model",
        "workflow.repair_attempts",
        "generated.generated_tests",
        "generated.generated_integration_tests",
        "t.generated_code",
        "workflow.candidate_comparison",
        "workflow.selection",
    ):
        assert field in source


def test_dashboard_does_not_render_empty_generated_test_code_as_executable():
    source = FRONTEND.read_text(encoding="utf-8")
    assert "typeof t.generated_code==='string'&&t.generated_code.trim()" in source
    assert "No valid AI candidate has been created yet." in source
    assert "AI candidate disabled:" in source
