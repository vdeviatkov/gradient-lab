#!/usr/bin/env python3
"""Check local inline/reference Markdown destinations without accessing the network."""

from __future__ import annotations

import argparse
import re
import subprocess
from collections.abc import Iterator, Sequence
from pathlib import Path
from urllib.parse import unquote, urlsplit

_DESTINATIONS = re.compile(
    r"!?\[[^\]\n]*\]\(\s*(<[^>\n]+>|[^\s)]+)"
    r"|^ {0,3}\[[^\]\n]+\]:\s*(<[^>\n]+>|\S+)",
    re.MULTILINE,
)


def without_fences(markdown: str) -> str:
    """Ignore fenced examples while retaining line positions for error messages."""
    fence = ""
    output = []
    for line in markdown.splitlines(keepends=True):
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        if marker:
            run = marker[1]
            if not fence:
                fence = run
            elif run[0] == fence[0] and len(run) >= len(fence) and not line[marker.end() :].strip():
                fence = ""
            output.append("\n" if line.endswith("\n") else "")
        elif fence:
            output.append("\n" if line.endswith("\n") else "")
        else:
            output.append(line)
    return "".join(output)


def heading_anchors(markdown: str) -> set[str]:
    """Return GitHub-style slugs for ATX headings, including duplicate suffixes."""
    anchors: set[str] = set()
    counts: dict[str, int] = {}
    for heading in re.finditer(r"^ {0,3}#{1,6}\s+(.+?)\s*#*\s*$", without_fences(markdown), re.M):
        title = re.sub(r"!?\[([^\]]+)\]\([^)]*\)", r"\1", heading[1])
        title = re.sub(r"<[^>]+>", "", title)
        slug = re.sub(r"[^\w\- ]", "", title.lower()).replace(" ", "-")
        number = counts.get(slug, 0)
        candidate = f"{slug}-{number}" if number else slug
        while candidate in anchors:
            number += 1
            candidate = f"{slug}-{number}"
        counts[slug] = number + 1
        anchors.add(candidate)
    return anchors


def destinations(markdown: str) -> Iterator[tuple[int, str]]:
    """Yield line number and destination for inline links/images and reference definitions."""
    content = without_fences(markdown)
    for match in _DESTINATIONS.finditer(content):
        yield content.count("\n", 0, match.start()) + 1, (match[1] or match[2]).strip("<>")


def check_markdown(path: Path, root: Path) -> list[str]:
    """Validate local files/directories and ATX heading anchors in one Markdown file."""
    root, path = root.resolve(), path.resolve()
    errors = []
    for line, destination in destinations(path.read_text(encoding="utf-8")):
        parsed = urlsplit(destination)
        if parsed.scheme or parsed.netloc:
            continue
        target_path = unquote(parsed.path)
        if not target_path:
            target = path
        elif target_path.startswith("/"):
            target = root / target_path.lstrip("/")
        else:
            target = path.parent / target_path
        target = target.resolve()
        problem = ""
        if not target.is_relative_to(root):
            problem = "destination escapes repository"
        elif not target.exists():
            problem = "destination does not exist"
        elif parsed.fragment and target.is_file() and target.suffix.lower() == ".md":
            if unquote(parsed.fragment) not in heading_anchors(target.read_text(encoding="utf-8")):
                problem = "Markdown heading anchor does not exist"
        if problem:
            errors.append(f"{path.relative_to(root)}:{line}: {destination}: {problem}")
    return errors


def markdown_files(root: Path) -> list[Path]:
    """Inspect tracked and non-ignored new docs; skip locally deleted tracked files."""
    result = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
            "*.md",
        ],
        check=True,
        capture_output=True,
    )
    return sorted(
        {
            root / name.decode("utf-8")
            for name in result.stdout.split(b"\0")
            if name and (root / name.decode("utf-8")).is_file()
        }
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    root = parser.parse_args(argv).root.resolve()
    try:
        files = markdown_files(root)
    except subprocess.CalledProcessError as error:
        parser.error(f"cannot list documentation in Git checkout {root}: {error}")
    errors = [error for path in files for error in check_markdown(path, root)]
    for error in errors:
        print(error)
    if errors:
        print(f"Found {len(errors)} broken local documentation destinations.")
        return 1
    print(f"Checked {len(files)} Markdown files: all local destinations resolve.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
