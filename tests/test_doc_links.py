"""The guides link to what exists, and name only commands and code that exist.

Checked in docs/, README.md and gsql/README.md: every relative link reaches a file and,
with an anchor, a heading of it (named as GitHub names headings); every `mule` command
in code, and the analysis given to `mule diagnose`, is one the command line has; every
name given to scripts/run_experiments.py is a suite or variant, every option of the
scripts is one they take, and every pytest marker is declared. Outside the records
(RECORDS), which keep what was true when they were written, every repository path in
code exists and every dotted name of the package in a code span resolves: to a module
or what it defines, or else to a setting of the built-in run (`training.seed`).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from dataclasses import fields, is_dataclass
import html
import importlib
from pathlib import Path
import re
import subprocess
import tomllib
from typing import Any

from mule_pattern_learner import cli
from mule_pattern_learner.config import DEFAULT_CONFIG
from mule_pattern_learner.diagnostics.study import ANALYSES
from mule_pattern_learner.experiments.variants import SUITES, VARIANTS
from mule_pattern_learner.paths import REPOSITORY_ROOT

PACKAGE = "mule_pattern_learner"
# Records of their time keep the paths and names of the code they describe: the research
# notes.
RECORDS = (REPOSITORY_ROOT / "docs/research",)
FENCE = re.compile(r"^(```|~~~).*?^\1[ \t]*$", re.S | re.M)
LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
HEADING = re.compile(r"^#{1,6}[ \t]+(.+?)[ \t]*#*[ \t]*$", re.M)
SPAN = re.compile(r"(`+)(.+?)\1", re.S)
# The word after a command; a closing bracket ends it, as in a diagram's [mule train].
COMMAND = re.compile(r"(?:(?<![\w./-])mule|python -m mule_pattern_learner)[ \t]+([^\s`\])]+)")
DIAGNOSE = re.compile(r"(?<![\w./-])mule[ \t]+diagnose[ \t]+([^\s`\])]+)")
SCRIPT = re.compile(r"scripts/(run_experiments|render_queries)\.py([^\n`#]*)")
MARKER = re.compile(r"pytest\b[^\n`#]*?-m[ \t]+([\w\"']+)")
REPOSITORY_PATH = re.compile(r"(?<![\w./<-])(?:src|gsql|scripts|tests|docs)/[\w./-]*")
DOTTED = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+")
# The last part of a file name, which makes `metrics.json` a file rather than a name.
EXTENSIONS = frozenset(
    {"py", "md", "gsql", "csv", "json", "jsonl", "parquet", "png", "pt", "txt", "toml", "gz"}
)


def guides() -> list[Path]:
    """The Markdown files of docs/ and the two READMEs that git tracks or would add."""
    listed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "docs"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    paths = [REPOSITORY_ROOT / name for name in listed if name.endswith(".md")]
    return sorted(
        path
        for path in (*paths, REPOSITORY_ROOT / "README.md", REPOSITORY_ROOT / "gsql/README.md")
        if path.exists()
    )


def prose(text: str) -> str:
    """The text outside fenced code blocks."""
    return FENCE.sub("", text)


def code(text: str) -> Iterator[str]:
    """Every fenced code block and every code span outside them."""
    for block in FENCE.finditer(text):
        yield block.group(0)
    for span in SPAN.finditer(prose(text)):
        yield span.group(2)


def spans(text: str) -> Iterator[str]:
    for span in SPAN.finditer(prose(text)):
        yield span.group(2).strip()


def anchors(path: Path) -> set[str]:
    """The anchors GitHub gives the headings of a Markdown file."""
    seen: Counter[str] = Counter()
    found = set()
    for heading in HEADING.finditer(prose(path.read_text())):
        text = html.unescape(re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", heading.group(1)))
        slug = re.sub(r"[^\w\- ]", "", text.strip().lower()).replace(" ", "-")
        found.add(slug if not seen[slug] else f"{slug}-{seen[slug]}")
        seen[slug] += 1
    return found


def relative(path: Path) -> str:
    return path.relative_to(REPOSITORY_ROOT).as_posix()


def test_every_relative_link_reaches_a_file_and_its_heading() -> None:
    problems = []
    for guide in guides():
        for target in LINK.findall(prose(guide.read_text())):
            if re.match(r"[a-z]+:", target):
                continue
            name, _, anchor = target.partition("#")
            linked = (guide.parent / name).resolve() if name else guide
            if not linked.exists() or not linked.is_relative_to(REPOSITORY_ROOT):
                problems.append(f"{relative(guide)}: {target} does not exist")
            elif anchor and (linked.suffix != ".md" or anchor not in anchors(linked)):
                problems.append(f"{relative(guide)}: {target} names no heading")
    assert problems == []


def test_the_commands_in_the_guides_are_the_command_lines() -> None:
    parser = cli.build_parser()
    (commands,) = (
        action.choices
        for action in parser._actions  # pyright: ignore[reportPrivateUsage]
        if action.choices is not None and action.dest == "command"
    )
    project = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text())
    markers = {line.split(":")[0] for line in project["tool"]["pytest"]["ini_options"]["markers"]}
    options = {"run_experiments": {"--help", "-h"}, "render_queries": {"--check", "--help", "-h"}}
    names = set(SUITES) | set(VARIANTS)
    problems = []
    for guide in guides():
        for text in code(guide.read_text()):
            for command in COMMAND.findall(text):
                if command not in commands and not command.startswith(("[", "<", "-")):
                    problems.append(f"{relative(guide)}: mule {command}")
            for analysis in DIAGNOSE.findall(text):
                if analysis not in ANALYSES and not analysis.startswith(("[", "<", "#")):
                    problems.append(f"{relative(guide)}: mule diagnose {analysis}")
            for script, arguments in SCRIPT.findall(text):
                # Placeholders such as [SUITE or VARIANT ...] name no argument.
                for word in re.sub(r"\[[^\]]*\]|<[^>]*>", "", arguments).split():
                    if word.startswith("-"):
                        known = word in options[script]
                    else:
                        known = script != "run_experiments" or word in names
                    if not known:
                        problems.append(f"{relative(guide)}: scripts/{script}.py {word}")
            for marker in MARKER.findall(text):
                if marker.strip("\"'") not in markers | {""}:
                    problems.append(f"{relative(guide)}: pytest -m {marker}")
    assert problems == []


def setting(name: str) -> bool:
    """Whether a dotted name is a setting of the built-in run, such as `training.seed`."""
    value: Any = DEFAULT_CONFIG.to_dict()
    for part in name.split("."):
        if not isinstance(value, dict) or part not in value:
            return False
        value = value[part]
    return True


def resolves(name: str) -> bool:
    """Whether a dotted name of the package names a module or something one defines."""
    parts = name.split(".")
    if parts[0] != PACKAGE:
        parts = [PACKAGE, *parts]
    for cut in range(len(parts), 0, -1):
        try:
            found: Any = importlib.import_module(".".join(parts[:cut]))
        except ModuleNotFoundError:
            continue
        for part in parts[cut:]:
            if is_dataclass(found) and part in {field.name for field in fields(found)}:
                return part == parts[-1]
            if not hasattr(found, part):
                return False
            found = getattr(found, part)
        return True
    return False


def test_the_paths_and_names_in_the_guides_exist() -> None:
    package = REPOSITORY_ROOT / "src" / PACKAGE
    modules = {
        path.stem
        for path in package.iterdir()
        if not path.stem.startswith("_") and (path.is_dir() or path.suffix == ".py")
    }
    problems = []
    for guide in guides():
        if any(guide.is_relative_to(record) for record in RECORDS):
            continue
        text = guide.read_text()
        for snippet in code(text):
            for path in REPOSITORY_PATH.findall(snippet):
                if not (REPOSITORY_ROOT / path.rstrip(".")).exists():
                    problems.append(f"{relative(guide)}: {path}")
        for span in spans(text):
            for name in DOTTED.findall(span):
                parts = name.split(".")
                if parts[0] not in modules | {PACKAGE} or parts[-1] in EXTENSIONS:
                    continue
                if not resolves(name) and not setting(name):
                    problems.append(f"{relative(guide)}: {name}")
    assert problems == []
