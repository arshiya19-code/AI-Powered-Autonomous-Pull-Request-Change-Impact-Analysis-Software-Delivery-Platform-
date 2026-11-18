"""Token-based parsers for the supported brace-delimited languages.

The implementation deliberately tokenizes source before analysing declarations.
It is therefore not a line-regex extractor: comments and string literals are
discarded, nested braces are tracked, and declarations/calls are associated
with their lexical owners.  The resulting public boundary is always the
language-neutral :class:`ParsedSource` representation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from analyzer.languages.base import (
    Dependency, LanguageDependencyType, ParsedSource, ProjectContext, Symbol,
    SymbolKind,
)


@dataclass(frozen=True)
class _Token:
    text: str
    line: int


def _tokenize(source: str) -> Tuple[List[_Token], List[str]]:
    """Lex source while removing comments and literals and retaining lines."""
    tokens: List[_Token] = []
    errors: List[str] = []
    i = 0
    line = 1
    n = len(source)
    while i < n:
        ch = source[i]
        if ch.isspace():
            line += ch == "\n"
            i += 1
        elif source.startswith("//", i):
            end = source.find("\n", i)
            i = n if end < 0 else end
        elif source.startswith("/*", i):
            end = source.find("*/", i + 2)
            if end < 0:
                errors.append("unterminated block comment")
                break
            line += source[i:end + 2].count("\n")
            i = end + 2
        elif ch in "\"'`":
            quote = ch
            start_line = line
            i += 1
            value: List[str] = []
            while i < n and source[i] != quote:
                if source[i] == "\\" and i + 1 < n:
                    value.append(source[i + 1])
                    i += 2
                    continue
                line += source[i] == "\n"
                value.append(source[i])
                i += 1
            if i == n:
                errors.append(f"unterminated string at line {start_line}")
                break
            tokens.append(_Token("@str:" + "".join(value), start_line))
            i += 1
        elif ch.isalpha() or ch == "_":
            j = i + 1
            while j < n and (source[j].isalnum() or source[j] == "_"):
                j += 1
            tokens.append(_Token(source[i:j], line))
            i = j
        elif ch.isdigit():
            j = i + 1
            while j < n and (source[j].isalnum() or source[j] in "._"):
                j += 1
            tokens.append(_Token(source[i:j], line))
            i = j
        else:
            tokens.append(_Token(ch, line))
            i += 1
    return tokens, errors


def _module_name(path: str) -> str:
    p = Path(path).with_suffix("")
    return ".".join(part for part in p.parts if part not in (".", ""))


class BraceLanguageAdapter:
    """Shared concrete parser for C-family and Rust syntax families."""
    _language = ""
    _extensions: Set[str] = set()
    _class_words: Set[str] = {"class", "interface", "struct", "enum"}
    _function_words: Set[str] = {"function", "fn"}

    @property
    def language(self) -> str:
        return self._language

    def supports_file(self, path: str) -> bool:
        return Path(path).suffix.lower() in self._extensions

    def parse(self, source: str, file_path: str) -> ParsedSource:
        tokens, errors = _tokenize(source)
        module = _module_name(file_path)
        lines = source.count("\n") + (1 if source else 0)
        symbols: List[Symbol] = [Symbol(
            name=module.split(".")[-1], qualified_name=module,
            symbol_kind=SymbolKind.MODULE, file_path=file_path,
            start_line=1, end_line=lines, language=self.language,
        )]
        deps: List[Dependency] = []
        brace_depth = 0
        scopes: List[Tuple[str, SymbolKind, int, int]] = []
        declaration_indices: Set[int] = set()

        def owner() -> str:
            return scopes[-1][0] if scopes else module

        def add_dep(target: str, kind: LanguageDependencyType, at: _Token) -> None:
            deps.append(Dependency(owner(), target, kind, file_path, at.line,
                                   language=self.language))

        i = 0
        while i < len(tokens):
            token = tokens[i]
            text = token.text
            if text == "{":
                brace_depth += 1
                i += 1
                continue
            if text == "}":
                while scopes and scopes[-1][3] == brace_depth:
                    qname, _kind, start, _depth = scopes.pop()
                    for symbol in symbols:
                        if symbol.qualified_name == qname:
                            symbol.end_line = token.line
                            break
                brace_depth = max(0, brace_depth - 1)
                i += 1
                continue
            if text == "import":
                target, end = self._read_import(tokens, i)
                if target:
                    add_dep(target, LanguageDependencyType.IMPORTS, token)
                i = max(i + 1, end)
                continue
            if text == "require" and i + 2 < len(tokens) and tokens[i + 1].text == "(":
                if tokens[i + 2].text.startswith("@str:"):
                    add_dep(tokens[i + 2].text[5:], LanguageDependencyType.IMPORTS, token)
                i += 1
                continue
            if text == "#" and i + 2 < len(tokens) and tokens[i + 1].text == "include":
                target, end = self._read_include(tokens, i + 2)
                if target:
                    add_dep(target, LanguageDependencyType.IMPORTS, token)
                i = end
                continue
            if text in self._class_words and i + 1 < len(tokens) and self._identifier(tokens[i + 1].text):
                name = tokens[i + 1].text
                qname = owner() + "." + name
                open_at = self._next(tokens, i + 2, "{")
                if open_at is not None:
                    symbols.append(Symbol(name, qname, SymbolKind.CLASS, file_path,
                        token.line, 0, self.language, parent=owner(),
                        is_public=not name.startswith("_")))
                    scopes.append((qname, SymbolKind.CLASS, token.line, brace_depth + 1))
                    declaration_indices.add(i + 1)
                    # inheritance after ':' (C++) or ':'/extends (others)
                    for j in range(i + 2, open_at):
                        if self._identifier(tokens[j].text) and tokens[j].text not in {"extends", "implements", "public", "private", "protected"}:
                            if j > i + 2:
                                add_dep(tokens[j].text, LanguageDependencyType.INHERITS, tokens[j])
                    i = open_at
                    continue
            func = self._function_declaration(tokens, i)
            if func is not None:
                name_index, open_at = func
                name = tokens[name_index].text
                parent_class = scopes[-1][0] if scopes and scopes[-1][1] == SymbolKind.CLASS else None
                kind = SymbolKind.METHOD if parent_class else SymbolKind.FUNCTION
                qname = (parent_class or module) + "." + name
                symbols.append(Symbol(name, qname, kind, file_path, token.line, 0,
                    self.language, parent=parent_class, is_public=not name.startswith("_")))
                scopes.append((qname, kind, token.line, brace_depth + 1))
                declaration_indices.add(name_index)
                i = open_at
                continue
            if (self._identifier(text) and i not in declaration_indices and i + 1 < len(tokens)
                    and tokens[i + 1].text == "(" and text not in self._keywords):
                add_dep(text, LanguageDependencyType.CALLS, token)
            i += 1
        if brace_depth:
            errors.append("unbalanced braces")
        return ParsedSource(file_path=file_path, language=self.language, symbols=symbols,
                            dependencies=deps, errors=errors,
                            metadata={"module_name": module, "source_lines": lines,
                                      "parser": "brace-token-parser"})

    _keywords = {"if", "for", "while", "switch", "catch", "return", "sizeof", "new", "delete", "match"}

    @staticmethod
    def _identifier(text: str) -> bool:
        return bool(text) and (text[0].isalpha() or text[0] == "_")

    @staticmethod
    def _next(tokens: Sequence[_Token], start: int, wanted: str) -> Optional[int]:
        for i in range(start, len(tokens)):
            if tokens[i].text == wanted:
                return i
            if tokens[i].text in {";", "}"}:
                return None
        return None

    def _function_declaration(self, tokens: Sequence[_Token], i: int) -> Optional[Tuple[int, int]]:
        if tokens[i].text in self._function_words and i + 2 < len(tokens):
            name = i + 1
            if self._identifier(tokens[name].text) and tokens[name + 1].text == "(":
                close = self._next(tokens, name + 1, ")")
                if close is not None:
                    opened = self._next(tokens, close + 1, "{")
                    if opened is not None:
                        return name, opened
        # C#/C++ methods have a type/qualifier before their name.
        if self._identifier(tokens[i].text) and i + 2 < len(tokens) and tokens[i + 1].text == "(":
            close = self._next(tokens, i + 1, ")")
            if close is not None:
                opened = self._next(tokens, close + 1, "{")
                if opened is not None and i > 0 and tokens[i - 1].text not in {".", "::"}:
                    return i, opened
        return None

    def _read_import(self, tokens: Sequence[_Token], i: int) -> Tuple[str, int]:
        j = i + 1
        # ES import ... from "x" and Rust use foo::bar are both handled.
        parts: List[str] = []
        while j < len(tokens) and tokens[j].text not in {";", "\n"}:
            if tokens[j].text.startswith("@str:"):
                return tokens[j].text[5:], j + 1
            if tokens[j].text == "from" and j + 1 < len(tokens) and tokens[j + 1].text.startswith("@str:"):
                return tokens[j + 1].text[5:], j + 2
            if self._identifier(tokens[j].text) or tokens[j].text in {".", ":"}:
                parts.append(tokens[j].text)
            j += 1
        return "".join(parts), j

    def _read_include(self, tokens: Sequence[_Token], i: int) -> Tuple[str, int]:
        if i < len(tokens) and tokens[i].text == "<":
            parts: List[str] = []
            i += 1
            while i < len(tokens) and tokens[i].text != ">":
                parts.append(tokens[i].text)
                i += 1
            return "".join(parts), i + 1
        if i < len(tokens) and tokens[i].text.startswith("@str:"):
            return tokens[i].text[5:], i + 1
        return "", i + 1

    def extract_dependencies(self, parsed: ParsedSource) -> List[Dependency]:
        return list(parsed.dependencies)

    def resolve_symbols(self, parsed: ParsedSource, context: ProjectContext) -> ParsedSource:
        """Resolve exact names and unambiguous suffixes within this language."""
        known = {s.qualified_name for source in context.all_sources if source.language == self.language for s in source.symbols}
        by_short: Dict[str, List[str]] = {}
        for name in known:
            by_short.setdefault(name.split(".")[-1], []).append(name)
        for dep in parsed.dependencies:
            if dep.target in known:
                continue
            matches = by_short.get(dep.target.split(".")[-1], [])
            if len(matches) == 1:
                dep.target = matches[0]
                dep.metadata["resolved"] = True
        return parsed


class TypeScriptLanguageAdapter(BraceLanguageAdapter):
    _language, _extensions = "typescript", {".ts", ".tsx", ".mts", ".cts"}
    _function_words = {"function"}


class JavaScriptLanguageAdapter(BraceLanguageAdapter):
    _language, _extensions = "javascript", {".js", ".jsx", ".mjs", ".cjs"}
    _function_words = {"function"}


class CSharpLanguageAdapter(BraceLanguageAdapter):
    _language, _extensions = "csharp", {".cs"}

    def parse(self, source: str, file_path: str) -> ParsedSource:
        # C# namespace imports use the spelling `using`; the remaining
        # declaration grammar is shared with the C-family parser.
        return super().parse(source.replace("using ", "import "), file_path)


class RustLanguageAdapter(BraceLanguageAdapter):
    _language, _extensions = "rust", {".rs"}
    _function_words = {"fn"}
    _class_words = {"struct", "enum", "trait", "impl"}

    def _read_import(self, tokens: Sequence[_Token], i: int) -> Tuple[str, int]:
        # Rust uses `use`, which shares the token path until a semicolon.
        return super()._read_import(tokens, i)

    def parse(self, source: str, file_path: str) -> ParsedSource:
        # Treat Rust `use` declarations as imports with the same lexical parser.
        parsed = super().parse(source.replace("use ", "import "), file_path)
        return parsed


class CppLanguageAdapter(BraceLanguageAdapter):
    _language, _extensions = "cpp", {".cpp", ".hpp", ".cc", ".cxx", ".hh"}
    _function_words = set()
