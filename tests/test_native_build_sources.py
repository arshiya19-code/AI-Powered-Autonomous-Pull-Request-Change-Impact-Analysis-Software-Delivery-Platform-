from types import SimpleNamespace
import os

import pytest
import setuptools  # noqa: F401 - initialize distutils compatibility

from backend.app.services import native_build


@pytest.mark.skipif(os.name != 'nt', reason='Windows compiler wrapper')
@pytest.mark.parametrize('existing', [False, True])
def test_native_build_preserves_input_source_tree(tmp_path, monkeypatch, existing):
    import distutils.core

    monkeypatch.chdir(tmp_path)
    source = tmp_path / 'extension.c'
    original = b'/* original repository C source */\n'
    if existing:
        source.write_bytes(original)
    extension = SimpleNamespace(sources=['extension.c'], include_dirs=[], extra_objects=[])
    command = SimpleNamespace(ensure_finalized=lambda: None, build_extensions=lambda: None)

    def setup(*args, **kwargs):
        source.write_text('/* generated checkout-specific C source */\n')
        return SimpleNamespace(ext_modules=[extension], get_command_obj=lambda name: command)

    monkeypatch.setattr(distutils.core, 'run_setup', setup)
    monkeypatch.setattr(native_build.importlib.util, 'find_spec', lambda name: True)
    assert native_build.main() == 0
    assert source.read_bytes() == original if existing else not source.exists()
