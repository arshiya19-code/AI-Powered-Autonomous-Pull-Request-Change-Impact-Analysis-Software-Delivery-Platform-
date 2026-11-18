import pytest

from backend.app.services.test_generation import TestGenerationService


def test_direct_callee_cannot_be_labeled_gui_integration(tmp_path):
    item = {'test_kind': 'integration', 'dependency_path': 'calculator.ui.gui.CalculatorApp.on_button_press -> calculator.Calculator.percent',
            'content': 'from calculator import Calculator\n\ndef test_percent():\n    calc = Calculator()\n    calc.percent()\n    assert calc.expression == "/100"\n'}
    with pytest.raises(ValueError, match='does not execute its stated caller'):
        TestGenerationService._integration_preflight(tmp_path, [item])


def test_real_imported_caller_is_accepted(tmp_path):
    item = {'test_kind': 'integration', 'dependency_path': 'checkout.checkout_total -> calculator.total_with_tax',
            'content': 'from checkout import checkout_total\n\ndef test_checkout():\n    assert checkout_total(100, 0.08) == 108\n'}
    TestGenerationService._integration_preflight(tmp_path, [item])


@pytest.mark.parametrize('route', ['/question/{position}', '/question/<int:position>'])
def test_http_route_must_match_actual_caller_decorator(tmp_path, route):
    (tmp_path / 'app').mkdir()
    (tmp_path / 'app/main.py').write_text(f'@app.get("{route}")\ndef read_questions(position):\n    return service.read_questions(position)\n')
    item = {'test_kind': 'integration', 'dependency_path': 'app.main.read_questions -> app.api.api.read_questions',
            'content': 'def test_question(client):\n    assert client.get("/question/1").status_code == 200\n'}
    TestGenerationService._integration_preflight(tmp_path, [item])
    item['content'] = 'def test_question(client):\n    assert client.get("/unrelated/1").status_code == 200\n'
    with pytest.raises(ValueError, match='does not execute its stated caller'):
        TestGenerationService._integration_preflight(tmp_path, [item])


def test_module_caller_requires_route_to_actual_callee(tmp_path):
    (tmp_path / 'app').mkdir()
    (tmp_path / 'app/main.py').write_text('@app.get("/question/{position}")\ndef route(position):\n    return service.read_questions(position)\n\n@app.get("/health")\ndef health():\n    return True\n')
    item = {'test_kind': 'integration', 'dependency_path': 'app.main -> app.api.api.read_questions',
            'content': 'def test_question(client):\n    assert client.get("/question/1").status_code == 200\n'}
    TestGenerationService._integration_preflight(tmp_path, [item])
    item['content'] = 'def test_health(client):\n    assert client.get("/health").status_code == 200\n'
    with pytest.raises(ValueError, match='does not execute its stated caller'):
        TestGenerationService._integration_preflight(tmp_path, [item])


def test_short_graph_module_resolves_only_unique_package_source(tmp_path):
    (tmp_path / 'package').mkdir()
    source = '@app.route("/primes/<int:count>")\ndef primes(count):\n    return service.primes(count)\n'
    (tmp_path / 'package/app.py').write_text(source)
    item = {'test_kind': 'integration', 'dependency_path': 'app.primes -> service.primes',
            'content': 'def test_primes(client):\n    assert client.get("/primes/5").status_code == 200\n'}
    TestGenerationService._integration_preflight(tmp_path, [item])
    (tmp_path / 'other').mkdir()
    (tmp_path / 'other/app.py').write_text(source)
    with pytest.raises(ValueError, match='does not execute its stated caller'):
        TestGenerationService._integration_preflight(tmp_path, [item])
