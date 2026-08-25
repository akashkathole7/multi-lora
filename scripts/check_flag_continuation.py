#!/usr/bin/env python3
"""Entry-10 guard: no flag-only continuation line that depends on a fold.

change_log entry 10: a shell command spread over several deeper-indented lines
inside a YAML folded scalar silently became several commands, because a folded
scalar KEEPS the newline of a more-indented line. The container ran
`vllm serve <model>` with none of its flags. Cost: ~3.2 A100-hours.

The signature is a physical line whose first token is a bare `-flag`. In YAML
that is always the bug. In a shell script it is only correct when bash itself
joins the lines - a trailing backslash, or an unclosed `(` (array literal or
subshell) - or when it is a `case` pattern, which is not a command at all.

Check A (YAML): any flag-starting physical line inside a `command:` block
                scalar is a failure, full stop.
Check B (shell): a flag-starting line is a failure UNLESS bash is joining it.
"""
import re
import sys
from pathlib import Path

ROOTS = [Path("serve/azure"), Path("train/azureml"), Path("serve/spark")]
FLAG = re.compile(r"^\s*-{1,2}[A-Za-z0-9]")
CASE_PATTERN = re.compile(r"^\s*-{1,2}[^)]*\)")  # e.g.  --hours)   or  -h|--help)
JOINERS = ("\\", "(", "|", "&&", "||", "then", "do")

failures = []
scanned = []


def strip_quotes_and_comments(line: str) -> str:
    """Crude but adequate: drop quoted spans and a trailing # comment."""
    out, i, quote = [], 0, None
    while i < len(line):
        ch = line[i]
        if quote:
            if ch == "\\" and quote == '"':
                i += 2
                continue
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in ("'", '"'):
            quote = ch
            i += 1
            continue
        if ch == "\\":
            i += 2
            continue
        if ch == "#" and (not out or out[-1].isspace()):
            break
        out.append(ch)
        i += 1
    return "".join(out)


def scan_shell(path: Path) -> None:
    scanned.append(str(path))
    lines = path.read_text(encoding="utf-8").splitlines()
    depth = 0
    prev = ""
    for n, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if FLAG.match(line):
            joined = prev.endswith(JOINERS) or depth > 0
            is_case = CASE_PATTERN.match(line) is not None
            if not joined and not is_case:
                failures.append(
                    f"{path}:{n}: flag-only line that bash will NOT join to the "
                    f"line above: {stripped!r}"
                )
        clean = strip_quotes_and_comments(line)
        depth += clean.count("(") - clean.count(")")
        if depth < 0:
            depth = 0
        if stripped:
            prev = stripped
    if depth != 0:
        failures.append(f"{path}: unbalanced parentheses (depth {depth}) - guard is unreliable here")


def scan_yaml_command(path: Path) -> None:
    scanned.append(str(path))
    lines = path.read_text(encoding="utf-8").splitlines()
    i = 0
    found_command = False
    while i < len(lines):
        m = re.match(r"^(\s*)command:\s*[>|][-+]?\s*$", lines[i])
        if not m:
            i += 1
            continue
        found_command = True
        indent = len(m.group(1))
        i += 1
        while i < len(lines):
            line = lines[i]
            if line.strip() and (len(line) - len(line.lstrip())) <= indent:
                break
            if line.strip() and FLAG.match(line):
                failures.append(
                    f"{path}:{i + 1}: flag-only continuation line inside a YAML "
                    f"command: block scalar - a fold will NOT remove this "
                    f"newline: {line.strip()!r}"
                )
            i += 1
    if found_command:
        scanned[-1] += "  (has a command: block scalar)"


def main() -> int:
    roots = [Path(a) for a in sys.argv[1:]] or ROOTS
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if path.suffix == ".sh":
                scan_shell(path)
            elif path.suffix in (".yaml", ".yml"):
                scan_yaml_command(path)
    print(f"entry-10 guard: scanned {len(scanned)} files")
    for s in scanned:
        print(f"  {s}")
    if failures:
        print("entry-10 guard: FAIL")
        for f in failures:
            print(f"  {f}")
        return 1
    print("entry-10 guard: PASS - every flag-starting line is one bash joins itself,")
    print("                and no YAML command: scalar has one at all")
    return 0


if __name__ == "__main__":
    sys.exit(main())
