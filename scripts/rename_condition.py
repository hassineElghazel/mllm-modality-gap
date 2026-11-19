"""Rename a condition tag across the outputs/ tree (file names and JSON content).

Used to rename Crank15 to Crank20 after the paper adopted the name of the value
the condition reaches on the evaluation set. Run it from the repository root on
every machine that holds outputs (the laptop and the cluster):

    python scripts/rename_condition.py Crank15 Crank20 --dry-run
    python scripts/rename_condition.py Crank15 Crank20

What it does:
  * renames every file or directory under outputs/ whose name contains OLD
    (outputs/logs/ is left as it is: logs are the record of what ran);
  * rewrites OLD to NEW inside .json / .jsonl files under outputs/ (except logs),
    matching OLD only when it is not followed by a digit, so e.g. "Crank150"
    would be left alone;
  * never touches binary files (.pt checkpoints and embeddings keep their bytes).

It refuses to overwrite an existing target, so running it twice is safe.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path("outputs")
SKIP = ROOT / "logs"
TEXT_SUFFIXES = {".json", ".jsonl"}


def skipped(p: Path) -> bool:
    return SKIP in p.parents or p == SKIP


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if not ROOT.is_dir():
        raise SystemExit("run from the repository root (outputs/ not found)")
    pat = re.compile(re.escape(a.old) + r"(?![0-9])")

    # 1. rewrite text content first, while paths are still the old ones
    edited = 0
    for p in sorted(ROOT.rglob("*")):
        if skipped(p) or not p.is_file() or p.suffix not in TEXT_SUFFIXES:
            continue
        txt = p.read_text(encoding="utf-8")
        new_txt, n = pat.subn(a.new, txt)
        if n:
            edited += 1
            print(f"[edit] {p}  ({n} occurrence{'s' if n > 1 else ''})")
            if not a.dry_run:
                p.write_text(new_txt, encoding="utf-8")

    # 2. rename paths, deepest first so directory renames do not invalidate children
    renamed = 0
    for p in sorted(ROOT.rglob("*"), key=lambda q: len(q.parts), reverse=True):
        if skipped(p) or not pat.search(p.name):
            continue
        target = p.with_name(pat.sub(a.new, p.name))
        if target.exists():
            raise SystemExit(f"refusing to overwrite {target}")
        renamed += 1
        print(f"[move] {p} -> {target}")
        if not a.dry_run:
            p.rename(target)

    mode = "would change" if a.dry_run else "changed"
    print(f"{mode}: {edited} files edited, {renamed} paths renamed")


if __name__ == "__main__":
    main()
