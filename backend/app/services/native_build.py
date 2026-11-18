"""Build repository-declared native extensions in the disposable checkout."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import tempfile


def main():
    # Prefer the project's normal build; a configured Windows Zig installation
    # supports the same Extension metadata when MSVC is not installed.
    if os.name != 'nt' or not importlib.util.find_spec('ziglang'):
        return subprocess.call([sys.executable, 'setup.py', 'build_ext', '--inplace'])
    import setuptools  # initialize distutils compatibility before loading metadata
    from distutils.core import run_setup
    original_sources = {path.resolve(): path.read_bytes() for suffix in ("*.c", "*.cpp") for path in Path.cwd().rglob(suffix)
                        if 'build' not in path.parts}
    distribution = run_setup('setup.py', stop_after='config')
    from distutils.unixccompiler import UnixCCompiler
    compiler = UnixCCompiler()
    cc = [sys.executable, '-m', 'ziglang', 'cc']
    compiler.set_executables(compiler=cc, compiler_so=cc, compiler_cxx=cc,
                             linker_so=[*cc, '-shared'], linker_exe=cc)
    command = distribution.get_command_obj('build_ext')
    command.inplace = True
    command.ensure_finalized()
    compiler.set_library_dirs([str(Path(sys.base_prefix) / "libs")])
    command.compiler = compiler
    python_library = Path(sys.base_prefix) / 'libs' / f'python{sys.version_info.major}{sys.version_info.minor}.lib'
    for extension in distribution.ext_modules or []:
        extension.include_dirs = [*extension.include_dirs, sysconfig.get_paths()['include']]
        extension.extra_objects = [*extension.extra_objects, str(python_library)]
    os.environ.setdefault('ZIG_GLOBAL_CACHE_DIR', str(Path(tempfile.gettempdir()) / 'impact-zig-toolchain-cache'))
    generated_sources = [Path(source) for extension in distribution.ext_modules or []
                         for source in extension.sources if Path(source).suffix in {".c", ".cpp"}
                         and Path(source).resolve() not in original_sources]
    try:
        command.build_extensions()
    finally:
        # Cython may regenerate an existing C source with checkout-specific paths.
        # Preserve the exact input tree after compiling, including tracked C files.
        for source, content in original_sources.items():
            if source.read_bytes() != content:
                source.write_bytes(content)
        # Keep generated native intermediates out of deterministic source analysis.
        for source in generated_sources:
            if source.resolve().is_relative_to(Path.cwd()) and source.is_file():
                source.unlink()

    return 0


if __name__ == '__main__':
    raise SystemExit(main())



